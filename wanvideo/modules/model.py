# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from ...enhance_a_video.enhance import get_feta_scores
from ...enhance_a_video.globals import is_enhance_enabled

from .attention import attention
import numpy as np
__all__ = ['WanModel']

from tqdm import tqdm
import gc
import comfy.model_management as mm
from ...utils import log, get_module_memory_mb

def poly1d(coefficients, x):
    result = torch.zeros_like(x)
    for i, coeff in enumerate(coefficients):
        result += coeff * (x ** (len(coefficients) - 1 - i))
    return result.abs()

def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


def rope_params(max_seq_len, dim, theta=10000, L_test=25, k=0):
    assert dim % 2 == 0
    exponents = torch.arange(0, dim, 2, dtype=torch.float64).div(dim)
    inv_theta_pow = 1.0 / torch.pow(theta, exponents)
    
    if k > 0:
        print(f"RifleX: Using {k}th freq")
        inv_theta_pow[k-1] = 0.9 * 2 * torch.pi / L_test
        
    freqs = torch.outer(torch.arange(max_seq_len), inv_theta_pow)
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs

from comfy.model_management import get_torch_device, get_autocast_device
@torch.autocast(device_type=get_autocast_device(get_torch_device()), enabled=False)
@torch.compiler.disable()
def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x.float()).type_as(x)


class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6,
                 attention_mode='sdpa'):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.attention_mode = attention_mode

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        if self.attention_mode == 'spargeattn_tune' or self.attention_mode == 'spargeattn':
            tune_mode = False
            if self.attention_mode == 'spargeattn_tune':
                tune_mode = True
                
            if hasattr(self, 'inner_attention'):
                #print("has inner attention")
                q=rope_apply(q, grid_sizes, freqs)
                k=rope_apply(k, grid_sizes, freqs)
                q = q.permute(0, 2, 1, 3)
                k = k.permute(0, 2, 1, 3)
                v = v.permute(0, 2, 1, 3)
                x = self.inner_attention(
                    q=q, 
                    k=k,
                    v=v, 
                    is_causal=False, 
                    tune_mode=tune_mode
                    ).permute(0, 2, 1, 3)
                #print("inner attention", x.shape) #inner attention torch.Size([1, 12, 32760, 128])
        else:
            q=rope_apply(q, grid_sizes, freqs)
            k=rope_apply(k, grid_sizes, freqs)
            if is_enhance_enabled():
                feta_scores = get_feta_scores(q, k)

            x = attention(
                q=q,
                k=k,
                v=v,
                k_lens=seq_lens,
                window_size=self.window_size,
                attention_mode=self.attention_mode)

        # output
        x = x.flatten(2)
        x = self.o(x)

        if is_enhance_enabled():
            x *= feta_scores

        return x


class WanT2VCrossAttention(WanSelfAttention):

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        # compute attention
        x = attention(q, k, v, k_lens=context_lens, attention_mode=self.attention_mode)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanI2VCrossAttention(WanSelfAttention):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6,
                 attention_mode='sdpa'):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        # self.alpha = nn.Parameter(torch.zeros((1, )))
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.attention_mode = attention_mode

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        context_img = context[:, :257]
        context = context[:, 257:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)
        img_x = attention(q, k_img, v_img, k_lens=None, attention_mode=self.attention_mode)
        # compute attention
        x = attention(q, k, v, k_lens=context_lens, attention_mode=self.attention_mode)

        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o(x)
        return x


WAN_CROSSATTENTION_CLASSES = {
    't2v_cross_attn': WanT2VCrossAttention,
    'i2v_cross_attn': WanI2VCrossAttention,
}


class WanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 attention_mode='sdpa'):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.attention_mode = attention_mode

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps, self.attention_mode)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps,#attention_mode=attention_mode sageattn doesn't seem faster here
                                                                      )
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        assert e.dtype == torch.float32
        e = (self.modulation.to(torch.float32).to(e.device) + e.to(torch.float32)).chunk(6, dim=1)
        assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(
            self.norm1(x).float() * (1 + e[1]) + e[0], seq_lens, grid_sizes,
            freqs)
        x = x.to(torch.float32) + (y.to(torch.float32) * e[2].to(torch.float32))

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            y = self.ffn(self.norm2(x).float() * (1 + e[4]) + e[3])
            x = x.to(torch.float32) + (y.to(torch.float32) * e[5].to(torch.float32))
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        assert e.dtype == torch.float32
        e_unsqueezed = e.unsqueeze(1).to(torch.float32)
        e = (self.modulation.to(torch.float32).to(e.device) + e_unsqueezed).chunk(2, dim=1)
        normed = self.norm(x).to(torch.float32)
        x = self.head(normed * (1 + e[1].to(torch.float32)) + e[0].to(torch.float32))
        return x


class MLPProj(torch.nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim), torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(), torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim))

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class WanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 attention_mode='sdpa',
                 main_device=torch.device('cuda'),
                 offload_device=torch.device('cpu'),
                 teacache_coefficients=[],):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.attention_mode = attention_mode
        self.main_device = main_device
        self.offload_device = offload_device

        self.blocks_to_swap = -1
        self.offload_txt_emb = False
        self.offload_img_emb = False

        #init TeaCache variables
        self.enable_teacache = False
        self.rel_l1_thresh = 0.15
        self.teacache_start_step= 0
        self.teacache_end_step = -1
        self.teacache_cache_device = main_device
        self.teacache_state = TeaCacheState()
        self.teacache_coefficients = teacache_coefficients
        self.teacache_use_coefficients = False


        self.slg_blocks = None
        self.slg_start_percent = 0.0
        self.slg_end_percent = 1.0
        # self.l1_history_x = []
        # self.l1_history_temb = []
        # self.l1_history_rescaled = []

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            WanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                              window_size, qk_norm, cross_attn_norm, eps,
                              attention_mode=self.attention_mode)
            for _ in range(num_layers)
        ])

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)

        # initialize weights
        #self.init_weights()

    def block_swap(self, blocks_to_swap, offload_txt_emb=False, offload_img_emb=False):
        print(f"Swapping {blocks_to_swap + 1} transformer blocks")
        self.blocks_to_swap = blocks_to_swap
        self.offload_img_emb = offload_img_emb
        self.offload_txt_emb = offload_txt_emb

        total_offload_memory = 0
        total_main_memory = 0
       
        for b, block in tqdm(enumerate(self.blocks), total=len(self.blocks), desc="Initializing block swap"):
            block_memory = get_module_memory_mb(block)
            
            if b > self.blocks_to_swap:
                block.to(self.main_device)
                total_main_memory += block_memory
            else:
                block.to(self.offload_device)
                total_offload_memory += block_memory

        mm.soft_empty_cache()
        gc.collect()
                
            #print(f"Block {b}: {block_memory:.2f}MB on {block.parameters().__next__().device}")
        log.info("----------------------")
        log.info(f"Block swap memory summary:")
        log.info(f"Transformer blocks on {self.offload_device}: {total_offload_memory:.2f}MB")
        log.info(f"Transformer blocks on {self.main_device}: {total_main_memory:.2f}MB")
        log.info(f"Total memory used by transformer blocks: {(total_offload_memory + total_main_memory):.2f}MB")
        log.info("----------------------")

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        is_uncond=False,
        current_step_percentage=0.0,
        clip_fea=None,
        y=None,
        device=torch.device('cuda'),
        freqs=None,
        current_step=0,
        pred_id=None
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """        
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        #device = self.patch_embedding.weight.device
        if freqs.device != device:
            freqs = freqs.to(device)
            
        if y is not None:
            #torch.Size([20, 17, 58, 104]) torch.Size([16, 17, 58, 104])
            #c ,t,h,w
            x = torch.cat([x, y], dim=0)

        # embeddings
        x = [self.patch_embedding(x.unsqueeze(0))]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        if self.offload_txt_emb:
            self.text_embedding.to(self.main_device)
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))
        if self.offload_txt_emb:
            self.text_embedding.to(self.offload_device, non_blocking=True)

        if clip_fea is not None:
            if self.offload_img_emb:
                self.img_emb.to(self.main_device)
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)
            if self.offload_img_emb:
                self.img_emb.to(self.offload_device, non_blocking=True)

        should_calc = True
        accumulated_rel_l1_distance = torch.tensor(0.0, dtype=torch.float32, device=device)
        if self.enable_teacache and self.teacache_start_step <= current_step <= self.teacache_end_step:
            if pred_id is None:
                pred_id = self.teacache_state.new_prediction()
                #log.info(current_step)
                #log.info(f"TeaCache: Initializing TeaCache variables for model pred: {pred_id}")
                should_calc = True                
            else:
                previous_modulated_input = self.teacache_state.get(pred_id)['previous_modulated_input']
                previous_modulated_input = previous_modulated_input.to(device)
                previous_residual = self.teacache_state.get(pred_id)['previous_residual']
                accumulated_rel_l1_distance = self.teacache_state.get(pred_id)['accumulated_rel_l1_distance']

                if self.teacache_use_coefficients:
                    rescale_func = np.poly1d(self.teacache_coefficients)
                    accumulated_rel_l1_distance += rescale_func(((e-previous_modulated_input).abs().mean() / previous_modulated_input.abs().mean()).cpu().item())
                else:
                    temb_relative_l1 = relative_l1_distance(previous_modulated_input, e0)
                    accumulated_rel_l1_distance = accumulated_rel_l1_distance.to(e0.device) + temb_relative_l1

                #print("accumulated_rel_l1_distance", accumulated_rel_l1_distance)

                if accumulated_rel_l1_distance < self.rel_l1_thresh:
                    should_calc = False
                else:
                    should_calc = True
                    accumulated_rel_l1_distance = torch.tensor(0.0, dtype=torch.float32, device=device)

            previous_modulated_input = e.clone() if self.teacache_use_coefficients else e0.clone()
            if not should_calc:
                x += previous_residual.to(x.device)
                #log.info(f"TeaCache: Skipping uncond step {current_step+1}")
                self.teacache_state.update(
                    pred_id,
                    accumulated_rel_l1_distance=accumulated_rel_l1_distance,
                    skipped_steps=self.teacache_state.get(pred_id)['skipped_steps'] + 1,
                )

        if not self.enable_teacache or (self.enable_teacache and should_calc):
            if self.enable_teacache:
                original_x = x.clone()
            # arguments
            kwargs = dict(
                e=e0,
                seq_lens=seq_lens,
                grid_sizes=grid_sizes,
                freqs=freqs,
                context=context,
                context_lens=context_lens)

            for b, block in enumerate(self.blocks):
                if self.slg_blocks is not None:
                    if b in self.slg_blocks and is_uncond:
                        if self.slg_start_percent <= current_step_percentage <= self.slg_end_percent:
                            continue
                if b <= self.blocks_to_swap and self.blocks_to_swap >= 0:
                    block.to(self.main_device)
                x = block(x, **kwargs)
                if b <= self.blocks_to_swap and self.blocks_to_swap >= 0:
                    block.to(self.offload_device, non_blocking=True)

            if self.enable_teacache and pred_id is not None:
                self.teacache_state.update(
                    pred_id,
                    previous_residual=(x - original_x),
                    accumulated_rel_l1_distance=accumulated_rel_l1_distance,
                    previous_modulated_input=previous_modulated_input
                )
                #self.teacache_state.report()

        # head
        x = self.head(x, e)
        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return x, pred_id

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        for v in grid_sizes.tolist():
            x = x[:math.prod(v)].view(*v, *self.patch_size, c)
            x = torch.einsum('fhwpqrc->cfphqwr', x)
            x = x.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
        return x

class TeaCacheState:
    def __init__(self, cache_device='cpu'):
        self.cache_device = cache_device
        self.states = {}
        self._next_pred_id = 0
    
    def new_prediction(self):
        """Create new prediction state and return its ID"""
        pred_id = self._next_pred_id
        self._next_pred_id += 1
        self.states[pred_id] = {
            'previous_residual': None,
            'accumulated_rel_l1_distance': 0,
            'previous_modulated_input': None,
            'skipped_steps': 0
        }
        return pred_id
    
    def update(self, pred_id, **kwargs):
        """Update state for specific prediction"""
        if pred_id not in self.states:
            return None
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                value = value.to(self.cache_device)
            self.states[pred_id][key] = value
    
    def get(self, pred_id):
        return self.states.get(pred_id, {})

    def report(self):
        for pred_id in self.states:
            log.info(f"Prediction {pred_id}: {self.states[pred_id]}")
    
    def clear_prediction(self, pred_id):
        if pred_id in self.states:
            del self.states[pred_id]
    
    def clear_all(self):
        self.states.clear()
        self._next_pred_id = 0

def relative_l1_distance(last_tensor, current_tensor):
    l1_distance = torch.abs(last_tensor.to(current_tensor.device) - current_tensor).mean()
    norm = torch.abs(last_tensor).mean()
    relative_l1_distance = l1_distance / norm
    return relative_l1_distance.to(torch.float32).to(current_tensor.device)

def normalize_values(values):
    min_val = min(values)
    max_val = max(values)
    if max_val == min_val:
        return [0.0] * len(values)
    return [(x - min_val) / (max_val - min_val) for x in values]

def rescale_differences(input_diffs, output_diffs):
    """Polynomial fitting between input and output differences"""
    poly_degree = 4
    if len(input_diffs) < 2:
        return input_diffs
    
    x = np.array([x.item() for x in input_diffs])
    y = np.array([y.item() for y in output_diffs])
    print("x ", x)
    print("y ", y)
    
    # Fit polynomial
    coeffs = np.polyfit(x, y, poly_degree)
    
    # Apply polynomial transformation
    return np.polyval(coeffs, x)