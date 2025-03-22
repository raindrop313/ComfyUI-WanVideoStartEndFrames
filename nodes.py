import os
import torch
import gc
from .utils import log, print_memory
import numpy as np
import math
from tqdm import tqdm

from .wanvideo.modules.clip import CLIPModel
from .wanvideo.modules.model import WanModel, rope_params
from .wanvideo.modules.t5 import T5EncoderModel
from .wanvideo.utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                               get_sampling_sigmas, retrieve_timesteps)
from .wanvideo.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

from .enhance_a_video.globals import enable_enhance, disable_enhance, set_enhance_weight, set_num_frames

from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device

import folder_paths
import comfy.model_management as mm
from comfy.utils import load_torch_file, save_torch_file, ProgressBar, common_upscale
import comfy.model_base
import comfy.latent_formats
from comfy.clip_vision import clip_preprocess, ClipVisionModel

script_directory = os.path.dirname(os.path.abspath(__file__))

def add_noise_to_reference_video(image, ratio=None):
    sigma = torch.ones((image.shape[0],)).to(image.device, image.dtype) * ratio 
    image_noise = torch.randn_like(image) * sigma[:, None, None, None]
    image_noise = torch.where(image==-1, torch.zeros_like(image), image_noise)
    image = image + image_noise
    return image


class WindowTrackerSE:
    def __init__(self, verbose=False):
        self.window_map = {}  # Maps frame sequence to persistent ID
        self.next_id = 0
        self.teacache_states = {}  # Maps persistent ID to teacache state
        self.verbose = verbose

    def get_window_id(self, frames):
        key = tuple(sorted(frames))  # Order-independent frame sequence
        if key not in self.window_map:
            self.window_map[key] = self.next_id
            if self.verbose:
                log.info(f"New window pattern {key} -> ID {self.next_id}")
            self.next_id += 1
        return self.window_map[key]

    def get_teacache(self, window_id, base_state):
        if window_id not in self.teacache_states:
            if self.verbose:
                log.info(f"Initializing persistent teacache for window {window_id}")
            self.teacache_states[window_id] = base_state.copy()
        return self.teacache_states[window_id]

class WanVideoSEModel(comfy.model_base.BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pipeline = {}

    def __getitem__(self, k):
        return self.pipeline[k]

    def __setitem__(self, k, v):
        self.pipeline[k] = v

try:
    from comfy.latent_formats import Wan21
    latent_format = Wan21
except: #for backwards compatibility
    log.warning("Wan21 latent format not found, update ComfyUI for better livepreview")
    from comfy.latent_formats import HunyuanVideo
    latent_format = HunyuanVideo

class WanVideoSEModelConfig:
    def __init__(self, dtype):
        self.unet_config = {}
        self.unet_extra_config = {}
        self.latent_format = latent_format
        self.latent_format.latent_channels = 16
        self.manual_cast_dtype = dtype
        self.sampling_settings = {"multiplier": 1.0}
        # Don't know what this is. Value taken from ComfyUI Mochi model.
        self.memory_usage_factor = 2.0
        # denoiser is handled by extension
        self.unet_config["disable_unet_model_creation"] = True

def filter_state_dict_by_blocks(state_dict, blocks_mapping):
    filtered_dict = {}

    for key in state_dict:
        if 'blocks.' in key:
            block_pattern = key.split('diffusion_model.')[1].split('.', 2)[0:2]
            block_key = f'{block_pattern[0]}.{block_pattern[1]}.'

            if block_key in blocks_mapping:
                filtered_dict[key] = state_dict[key]

    return filtered_dict

def standardize_lora_key_format(lora_sd):
    new_sd = {}
    for k, v in lora_sd.items():
        # Diffusers format
        if k.startswith('transformer.'):
            k = k.replace('transformer.', 'diffusion_model.')
            
        # from finetrainer format
        if '.attn1.' in k:
            k = k.replace('.attn1.', '.cross_attn.')
            k = k.replace('.to_k.', '.k.')
            k = k.replace('.to_q.', '.q.')
            k = k.replace('.to_v.', '.v.')
            k = k.replace('.to_out.0.', '.o.')
        elif '.attn2.' in k:
            k = k.replace('.attn2.', '.cross_attn.')
            k = k.replace('.to_k.', '.k.')
            k = k.replace('.to_q.', '.q.')
            k = k.replace('.to_v.', '.v.')
            k = k.replace('.to_out.0.', '.o.')
            
        if "img_attn.proj" in k:
            k = k.replace("img_attn.proj", "img_attn_proj")
        if "img_attn.qkv" in k:
            k = k.replace("img_attn.qkv", "img_attn_qkv")
        if "txt_attn.proj" in k:
            k = k.replace("txt_attn.proj ", "txt_attn_proj")
        if "txt_attn.qkv" in k:
            k = k.replace("txt_attn.qkv", "txt_attn_qkv")
        new_sd[k] = v
    return new_sd


#region Model loading
class WanVideoSEModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (folder_paths.get_filename_list("diffusion_models"), {"tooltip": "These models are loaded from the 'ComfyUI/models/diffusion_models' -folder",}),

            "base_precision": (["fp32", "bf16", "fp16", "fp16_fast"], {"default": "bf16"}),
            "quantization": (['disabled', 'fp8_e4m3fn', 'fp8_e4m3fn_fast', 'fp8_e5m2', 'torchao_fp8dq', "torchao_fp8dqrow", "torchao_int8dq", "torchao_fp6", "torchao_int4", "torchao_int8"], {"default": 'disabled', "tooltip": "optional quantization method"}),
            "load_device": (["main_device", "offload_device"], {"default": "main_device"}),
            },
            "optional": {
                "attention_mode": ([
                    "sdpa",
                    "flash_attn_2",
                    "flash_attn_3",
                    "sageattn",
                    "spargeattn",
                    "spargeattn_tune",
                    ], {"default": "sdpa"}),
                "compile_args": ("WANCOMPILEARGS", ),
                "block_swap_args": ("BLOCKSWAPARGS", ),
                "lora": ("WANVIDLORA", {"default": None}),
                "vram_management_args": ("VRAM_MANAGEMENTARGS", {"default": None, "tooltip": "Alternative offloading method from DiffSynth-Studio, more aggressive in reducing memory use than block swapping, but can be slower"}),
            }
        }

    RETURN_TYPES = ("WANVIDEOMODEL",)
    RETURN_NAMES = ("model", )
    FUNCTION = "loadmodel"
    CATEGORY = "WanVideoStartEndFrame"

    def loadmodel(self, model, base_precision, load_device,  quantization,
                  compile_args=None, attention_mode="sdpa", block_swap_args=None, lora=None, vram_management_args=None):
        assert not (vram_management_args is not None and block_swap_args is not None), "Can't use both block_swap_args and vram_management_args at the same time"        
        transformer = None
        mm.unload_all_models()
        mm.cleanup_models()
        mm.soft_empty_cache()
        manual_offloading = True
        if "sage" in attention_mode:
            try:
                from sageattention import sageattn
            except Exception as e:
                raise ValueError(f"Can't import SageAttention: {str(e)}")

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

                
        manual_offloading = True
        transformer_load_device = device if load_device == "main_device" else offload_device
        
        base_dtype = {"fp8_e4m3fn": torch.float8_e4m3fn, "fp8_e4m3fn_fast": torch.float8_e4m3fn, "bf16": torch.bfloat16, "fp16": torch.float16, "fp16_fast": torch.float16, "fp32": torch.float32}[base_precision]
        
        if base_precision == "fp16_fast":
            if hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation"):
                torch.backends.cuda.matmul.allow_fp16_accumulation = True
            else:
                raise ValueError("torch.backends.cuda.matmul.allow_fp16_accumulation is not available in this version of torch, requires torch 2.7.0.dev2025 02 26 nightly minimum currently")
        else:
            try:
                if hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation"):
                    torch.backends.cuda.matmul.allow_fp16_accumulation = False
            except:
                pass

        model_path = folder_paths.get_full_path_or_raise("diffusion_models", model)
        sd = load_torch_file(model_path, device=transformer_load_device, safe_load=True)

        first_key = next(iter(sd))
        if first_key.startswith("model.diffusion_model."):
            new_sd = {}
            for key, value in sd.items():
                new_key = key.replace("model.diffusion_model.", "", 1)
                new_sd[new_key] = value
            sd = new_sd

        dim = sd["patch_embedding.weight"].shape[0]
        in_channels = sd["patch_embedding.weight"].shape[1]
        print("in_channels: ", in_channels)
        ffn_dim = sd["blocks.0.ffn.0.bias"].shape[0]
        model_type = "i2v" if in_channels == 36 else "t2v"
        num_heads = 40 if dim == 5120 else 12
        num_layers = 40 if dim == 5120 else 30

        log.info(f"Model type: {model_type}, num_heads: {num_heads}, num_layers: {num_layers}")

        teacache_coefficients_map = {
            "1_3B": [2.39676752e+03, -1.31110545e+03, 2.01331979e+02, -8.29855975e+00, 1.37887774e-01],
            "14B": [-5784.54975374, 5449.50911966, -1811.16591783, 256.27178429, -13.02252404],
            "i2v_480": [-3.02331670e+02, 2.23948934e+02, -5.25463970e+01, 5.87348440e+00, -2.01973289e-01],
            "i2v_720": [-114.36346466, 65.26524496, -18.82220707, 4.91518089, -0.23412683],
        }
        if model_type == "i2v":
            model_variant = "i2v_480" if "480" in model else "i2v_720"
        elif model_type == "t2v":
            model_variant = "14B" if dim == 5120 else "1_3B"
        log.info(f"Model variant detected: {model_variant}")
        
        TRANSFORMER_CONFIG= {
            "dim": dim,
            "ffn_dim": ffn_dim,
            "eps": 1e-06,
            "freq_dim": 256,
            "in_dim": in_channels,
            "model_type": model_type,
            "out_dim": 16,
            "text_len": 512,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "attention_mode": attention_mode,
            "main_device": device,
            "offload_device": offload_device,
            "teacache_coefficients": teacache_coefficients_map[model_variant],
        }

        with init_empty_weights():
            transformer = WanModel(**TRANSFORMER_CONFIG)
        transformer.eval()

        comfy_model = WanVideoSEModel(
            WanVideoSEModelConfig(base_dtype),
            model_type=comfy.model_base.ModelType.FLOW,
            device=device,
        )
          

        if not "torchao" in quantization:
            log.info("Using accelerate to load and assign model weights to device...")
            if quantization == "fp8_e4m3fn" or quantization == "fp8_e4m3fn_fast" or quantization == "fp8_scaled":
                dtype = torch.float8_e4m3fn
            elif quantization == "fp8_e5m2":
                dtype = torch.float8_e5m2
            else:
                dtype = base_dtype
            params_to_keep = {"norm", "head", "bias", "time_in", "vector_in", "patch_embedding", "time_", "img_emb", "modulation"}
            for name, param in transformer.named_parameters():
                #print("Assigning Parameter name: ", name)
                dtype_to_use = base_dtype if any(keyword in name for keyword in params_to_keep) else dtype
                set_module_tensor_to_device(transformer, name, device=transformer_load_device, dtype=dtype_to_use, value=sd[name])

            comfy_model.diffusion_model = transformer
            comfy_model.load_device = transformer_load_device
            
            patcher = comfy.model_patcher.ModelPatcher(comfy_model, device, offload_device)
            
            if lora is not None:
                from comfy.sd import load_lora_for_models
                for l in lora:
                    log.info(f"Loading LoRA: {l['name']} with strength: {l['strength']}")
                    lora_path = l["path"]
                    lora_strength = l["strength"]
                    lora_sd = load_torch_file(lora_path, safe_load=True)
                    lora_sd = standardize_lora_key_format(lora_sd)
                    if l["blocks"]:
                        lora_sd = filter_state_dict_by_blocks(lora_sd, l["blocks"])

                    patcher, _ = load_lora_for_models(patcher, None, lora_sd, lora_strength, 0)
                    del lora_sd

                patcher.patch_model(device)

            del sd
            gc.collect()
            mm.soft_empty_cache()

            if load_device == "offload_device":
                patcher.model.diffusion_model.to(offload_device)

            if quantization == "fp8_e4m3fn_fast":
                from .fp8_optimization import convert_fp8_linear
                #params_to_keep.update({"ffn"})
                print(params_to_keep)
                convert_fp8_linear(patcher.model.diffusion_model, base_dtype, params_to_keep=params_to_keep)

            if vram_management_args is not None:
                from .diffsynth.vram_management import enable_vram_management, AutoWrappedModule, AutoWrappedLinear
                from .wanvideo.modules.model import WanLayerNorm, WanRMSNorm

                total_params_in_model = sum(p.numel() for p in patcher.model.diffusion_model.parameters())
                log.info(f"Total number of parameters in the loaded model: {total_params_in_model}")

                offload_percent = vram_management_args["offload_percent"]
                offload_params = int(total_params_in_model * offload_percent)
                params_to_keep = total_params_in_model - offload_params
                log.info(f"Selected params to offload: {offload_params}")
            
                enable_vram_management(
                    patcher.model.diffusion_model,
                    module_map = {
                        torch.nn.Linear: AutoWrappedLinear,
                        torch.nn.Conv3d: AutoWrappedModule,
                        torch.nn.LayerNorm: AutoWrappedModule,
                        WanLayerNorm: AutoWrappedModule,
                        WanRMSNorm: AutoWrappedModule,
                    },
                    module_config = dict(
                        offload_dtype=dtype,
                        offload_device=offload_device,
                        onload_dtype=dtype,
                        onload_device=device,
                        computation_dtype=base_dtype,
                        computation_device=device,
                    ),
                    max_num_param=params_to_keep,
                    overflow_module_config = dict(
                        offload_dtype=dtype,
                        offload_device=offload_device,
                        onload_dtype=dtype,
                        onload_device=offload_device,
                        computation_dtype=base_dtype,
                        computation_device=device,
                    ),
                )

            #compile
            if compile_args is not None:
                torch._dynamo.config.cache_size_limit = compile_args["dynamo_cache_size_limit"]
                if compile_args["compile_transformer_blocks_only"]:
                    for i, block in enumerate(patcher.model.diffusion_model.blocks):
                        patcher.model.diffusion_model.blocks[i] = torch.compile(block, fullgraph=compile_args["fullgraph"], dynamic=compile_args["dynamic"], backend=compile_args["backend"], mode=compile_args["mode"])
                else:
                    patcher.model.diffusion_model = torch.compile(patcher.model.diffusion_model, fullgraph=compile_args["fullgraph"], dynamic=compile_args["dynamic"], backend=compile_args["backend"], mode=compile_args["mode"])        
        elif "torchao" in quantization:
            try:
                from torchao.quantization import (
                quantize_,
                fpx_weight_only,
                float8_dynamic_activation_float8_weight,
                int8_dynamic_activation_int8_weight,
                int8_weight_only,
                int4_weight_only
            )
            except:
                raise ImportError("torchao is not installed")

            # def filter_fn(module: nn.Module, fqn: str) -> bool:
            #     target_submodules = {'attn1', 'ff'} # avoid norm layers, 1.5 at least won't work with quantized norm1 #todo: test other models
            #     if any(sub in fqn for sub in target_submodules):
            #         return isinstance(module, nn.Linear)
            #     return False

            if "fp6" in quantization:
                quant_func = fpx_weight_only(3, 2)
            elif "int4" in quantization:
                quant_func = int4_weight_only()
            elif "int8" in quantization:
                quant_func = int8_weight_only()
            elif "fp8dq" in quantization:
                quant_func = float8_dynamic_activation_float8_weight()
            elif 'fp8dqrow' in quantization:
                from torchao.quantization.quant_api import PerRow
                quant_func = float8_dynamic_activation_float8_weight(granularity=PerRow())
            elif 'int8dq' in quantization:
                quant_func = int8_dynamic_activation_int8_weight()

            log.info(f"Quantizing model with {quant_func}")
            comfy_model.diffusion_model = transformer
            patcher = comfy.model_patcher.ModelPatcher(comfy_model, device, offload_device)

            for i, block in enumerate(patcher.model.diffusion_model.blocks):
                log.info(f"Quantizing block {i}")
                for name, _ in block.named_parameters(prefix=f"blocks.{i}"):
                    #print(f"Parameter name: {name}")
                    set_module_tensor_to_device(patcher.model.diffusion_model, name, device=transformer_load_device, dtype=base_dtype, value=sd[name])
                if compile_args is not None:
                    patcher.model.diffusion_model.blocks[i] = torch.compile(block, fullgraph=compile_args["fullgraph"], dynamic=compile_args["dynamic"], backend=compile_args["backend"], mode=compile_args["mode"])
                quantize_(block, quant_func)
                print(block)
                #block.to(offload_device)
            for name, param in patcher.model.diffusion_model.named_parameters():
                if "blocks" not in name:
                    set_module_tensor_to_device(patcher.model.diffusion_model, name, device=transformer_load_device, dtype=base_dtype, value=sd[name])

            manual_offloading = False # to disable manual .to(device) calls
            log.info(f"Quantized transformer blocks to {quantization}")
            for name, param in patcher.model.diffusion_model.named_parameters():
                print(name, param.dtype)
                #param.data = param.data.to(self.vae_dtype).to(device)

            del sd
            mm.soft_empty_cache()

        patcher.model["dtype"] = base_dtype
        patcher.model["base_path"] = model_path
        patcher.model["model_name"] = model
        patcher.model["manual_offloading"] = manual_offloading
        patcher.model["quantization"] = "disabled"
        patcher.model["block_swap_args"] = block_swap_args
        patcher.model["auto_cpu_offload"] = True if vram_management_args is not None else False

        for model in mm.current_loaded_models:
            if model._model() == patcher:
                mm.current_loaded_models.remove(model)            

        return (patcher,)

#region load VAE

class WanVideoSEVAELoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("vae"), {"tooltip": "These models are loaded from 'ComfyUI/models/vae'"}),
            },
            "optional": {
                "precision": (["fp16", "fp32", "bf16"],
                    {"default": "bf16"}
                ),
            }
        }

    RETURN_TYPES = ("WANVAE",)
    RETURN_NAMES = ("vae", )
    FUNCTION = "loadmodel"
    CATEGORY = "WanVideoStartEndFrame"
    DESCRIPTION = "Loads Hunyuan VAE model from 'ComfyUI/models/vae'"

    def loadmodel(self, model_name, precision):
        from .wanvideo.wan_video_vae_SE import WanVideoVAE

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
        #with open(os.path.join(script_directory, 'configs', 'hy_vae_config.json')) as f:
        #    vae_config = json.load(f)
        model_path = folder_paths.get_full_path("vae", model_name)
        vae_sd = load_torch_file(model_path, safe_load=True)

        has_model_prefix = any(k.startswith("model.") for k in vae_sd.keys())
        if not has_model_prefix:
            vae_sd = {f"model.{k}": v for k, v in vae_sd.items()}
        
        vae = WanVideoVAE(dtype=dtype)
        vae.load_state_dict(vae_sd)
        vae.eval()
        vae.to(device = offload_device, dtype = dtype)
            

        return (vae,)





class WanVideoSEImageClipEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "clip_vision": ("CLIP_VISION",),
            "start": ("IMAGE", {"tooltip": "Image to encode"}),
            "end": ("IMAGE", {"tooltip": "Image to encode"}),
            "vae": ("WANVAE",),
            "generation_width": (
            "INT", {"default": 832, "min": 64, "max": 2048, "step": 8, "tooltip": "Width of the image to encode"}),
            "generation_height": (
            "INT", {"default": 480, "min": 64, "max": 29048, "step": 8, "tooltip": "Height of the image to encode"}),
            "num_frames": (
            "INT", {"default": 81, "min": 1, "max": 10000, "step": 4, "tooltip": "Number of frames to encode"}),
        },
            "optional": {
                "force_offload": ("BOOLEAN", {"default": True}),
                "noise_aug_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.001,
                                                 "tooltip": "Strength of noise augmentation, helpful for I2V where some noise can add motion and give sharper results"}),
                "latent_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001,
                                              "tooltip": "Additional latent multiplier, helpful for I2V where lower values allow for more motion"}),
                "clip_embed_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001,
                                                  "tooltip": "Additional clip embed multiplier"}),
                "adjust_resolution": ("BOOLEAN", {"default": True,
                                                  "tooltip": "Performs the same resolution adjustment as in the original code"}),
                "start_frame_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01,
                                                "tooltip": "Weight for the start frame. Higher values (>1.0) make the starting image more influential throughout the video, resulting in stronger adherence to the start frame's features. Lower values (<1.0) reduce its influence, allowing more creative freedom. Set to 0 to completely ignore the start frame."}),
                "end_frame_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01,
                                              "tooltip": "Weight for the end frame. Higher values (>1.0) make the ending image more influential, causing the video to transition more strongly toward the end frame's characteristics. Lower values (<1.0) reduce its influence. Balancing with start_frame_weight allows control over transition speed and style."}),
            }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS",)
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "process"
    CATEGORY = "WanVideoStartEndFrame"

    def process(self, clip_vision, vae, start, end, num_frames, generation_width, generation_height, force_offload=True,
                noise_aug_strength=0.0, latent_strength=1.0, clip_embed_strength=1.0, adjust_resolution=True,
                start_frame_weight=1.0, end_frame_weight=1.0):

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        self.image_mean = [0.48145466, 0.4578275, 0.40821073]
        self.image_std = [0.26862954, 0.26130258, 0.27577711]
        patch_size = (1, 2, 2)
        vae_stride = (4, 8, 8)

        H, W = start.shape[1], end.shape[2]
        max_area = generation_width * generation_height

        print(clip_vision)
        clip_vision.model.to(device)
        if isinstance(clip_vision, ClipVisionModel):
            clip_context = clip_vision.encode_image(start).last_hidden_state.to(device)
        else:
            pixel_values = clip_preprocess(start.to(device), size=224, mean=self.image_mean, std=self.image_std,
                                           crop=True).float()
            clip_context = clip_vision.visual(pixel_values)
        if clip_embed_strength != 1.0:
            clip_context *= clip_embed_strength

        if force_offload:
            clip_vision.model.to(offload_device)
            mm.soft_empty_cache()

        if adjust_resolution:
            aspect_ratio = H / W
            lat_h = round(
                np.sqrt(max_area * aspect_ratio) // vae_stride[1] //
                patch_size[1] * patch_size[1])
            lat_w = round(
                np.sqrt(max_area / aspect_ratio) // vae_stride[2] //
                patch_size[2] * patch_size[2])
            h = lat_h * vae_stride[1]
            w = lat_w * vae_stride[2]
        else:
            h = generation_height
            w = generation_width
            lat_h = h // 8
            lat_w = w // 8

        # Step 1: Create initial mask with ones for first frame, zeros for others
        mask = torch.ones(1, num_frames+1, lat_h, lat_w, device=device)
        mask[:, 1:-1] = 0
        
        # No longer applying weights to the mask - this should improve transition smoothness
        # mask[:, 0] *= start_frame_weight
        # mask[:, -1] *= end_frame_weight

        # Step 2: Repeat first frame 4 times and concatenate with remaining frames
        first_frame_repeated = torch.repeat_interleave(mask[:, 0:1], repeats=4, dim=1)
        last_frame_repeated = torch.repeat_interleave(mask[:, -1:], repeats=4, dim=1)
        mask = torch.concat([first_frame_repeated, mask[:, 1:-1], last_frame_repeated], dim=1)

        # Step 3: Reshape mask into groups of 4 frames
        mask = mask.view(1, mask.shape[1] // 4, 4, lat_h, lat_w)
        # Step 4: Transpose dimensions and select first batch
        mask = mask.transpose(1, 2)[0]

        # Calculate maximum sequence length
        frames_per_stride = (num_frames - 1) // vae_stride[0] + 2
        patches_per_frame = lat_h * lat_w // (patch_size[1] * patch_size[2])
        max_seq_len = frames_per_stride * patches_per_frame

        vae.to(device)

        # Step 1: Resize and rearrange the input image dimensions
        # resized_image = image.permute(0, 3, 1, 2)  # Rearrange dimensions to (B, C, H, W)
        # resized_image = torch.nn.functional.interpolate(resized_image, size=(h, w), mode='bicubic')
        resized_start_image = common_upscale(start.movedim(-1, 1), w, h, "lanczos", "disabled")
        resized_start_image = resized_start_image.transpose(0, 1)  # Transpose to match required format
        resized_start_image = resized_start_image * 2 - 1

        resized_end_image = common_upscale(end.movedim(-1, 1), w, h, "lanczos", "disabled")
        resized_end_image = resized_end_image.transpose(0, 1)  # Transpose to match required format
        resized_end_image = resized_end_image * 2 - 1

        if noise_aug_strength > 0.0:
            resized_start_image = add_noise_to_reference_video(resized_start_image, ratio=noise_aug_strength)
            resized_end_image = add_noise_to_reference_video(resized_end_image, ratio=noise_aug_strength)

        # Apply start and end frame weights to image data
        resized_start_image = resized_start_image * start_frame_weight
        resized_end_image = resized_end_image * end_frame_weight

        # Step 2: Create zero padding frames
        zero_frames = torch.zeros(3, num_frames - 1, h, w, device=device)

        # Step 3: Concatenate image with zero frames
        concatenated = torch.concat([resized_start_image.to(device), zero_frames, resized_end_image.to(device)], dim=1).to(
            device=device, dtype=vae.dtype)
        concatenated *= latent_strength
        y = vae.encode([concatenated], device,end_=True)[0]
        # y:[4+c,t,h,w]
        y = torch.concat([mask, y])

        vae.model.clear_cache()
        vae.to(offload_device)
        # clip_context contains the encoding of a single image
        image_embeds = {
            "image_embeds": y,
            "clip_context": clip_context,
            "max_seq_len": max_seq_len,
            "num_frames": num_frames,
            "lat_h": lat_h,
            "lat_w": lat_w,
        }

        return (image_embeds,)




#region Sampler

    



class WanVideoSESampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("WANVIDEOMODEL",),
                "text_embeds": ("WANVIDEOTEXTEMBEDS",),
                "image_embeds": ("WANVIDIMAGE_EMBEDS",),
                "steps": ("INT", {"default": 30, "min": 1}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 30.0, "step": 0.01}),
                "shift": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "force_offload": (
                "BOOLEAN", {"default": True, "tooltip": "Moves the model to the offload device after sampling"}),
                "scheduler": (["unipc", "dpm++", "dpm++_sde", "euler"],
                              {
                                  "default": 'unipc'
                              }),
                "riflex_freq_index": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1,
                                              "tooltip": "Frequency index for RIFLEX, disabled when 0, default 6. Allows for new frames to be generated after without looping"}),

            },
            "optional": {
                "samples": ("LATENT", {"tooltip": "init Latents to use for video2video process"}),
                "denoise_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "feta_args": ("FETAARGS",),
                "context_options": ("WANVIDCONTEXT",),
                "teacache_args": ("TEACACHEARGS",),
                "flowedit_args": ("FLOWEDITARGS",),
                "slg_args": ("SLGARGS",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "process"
    CATEGORY = "WanVideoStartEndFrame"

    def process(self, model, text_embeds, image_embeds, shift, steps, cfg, seed, scheduler, riflex_freq_index,
                force_offload=True, samples=None, feta_args=None, denoise_strength=1.0, context_options=None,
                teacache_args=None, flowedit_args=None,slg_args=None,):

        # assert not (context_options and teacache_args), "Context options cannot currently be used together with teacache."
        patcher = model
        model = model.model
        transformer = model.diffusion_model

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        steps = int(steps / denoise_strength)

        if scheduler == 'unipc':
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=1000,
                shift=shift,
                use_dynamic_shifting=False)
            sample_scheduler.set_timesteps(
                steps, device=device, shift=shift)
            timesteps = sample_scheduler.timesteps
        elif scheduler == 'euler':
            sample_scheduler = FlowMatchEulerDiscreteScheduler(
                num_train_timesteps=1000,
                shift=shift,
                use_dynamic_shifting=False)
            sampling_sigmas = get_sampling_sigmas(steps, shift)
            timesteps, _ = retrieve_timesteps(
                sample_scheduler,
                device=device,
                sigmas=sampling_sigmas)
        elif 'dpm++' in scheduler:
            if scheduler == 'dpm++_sde':
                algorithm_type = "sde-dpmsolver++"
            else:
                algorithm_type = "dpmsolver++"
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=1000,
                shift=shift,
                use_dynamic_shifting=False,
                algorithm_type=algorithm_type)
            sampling_sigmas = get_sampling_sigmas(steps, shift)
            timesteps, _ = retrieve_timesteps(
                sample_scheduler,
                device=device,
                sigmas=sampling_sigmas)
        else:
            raise NotImplementedError("Unsupported solver.")

        if denoise_strength < 1.0:
            steps = int(steps * denoise_strength)
            timesteps = timesteps[-(steps + 1):]

        seed_g = torch.Generator(device=torch.device("cpu"))
        seed_g.manual_seed(seed)
        image_cond = None
        clip_fea = None
        if transformer.model_type == "i2v":
            lat_h = image_embeds.get("lat_h", None)
            lat_w = image_embeds.get("lat_w", None)
            if lat_h is None or lat_w is None:
                raise ValueError("Clip encoded image embeds must be provided for I2V (Image to Video) model")
            noise = torch.randn(
                16,
                (image_embeds["num_frames"] - 1) // 4 + 2,
                lat_h,
                lat_w,
                dtype=torch.float32,
                generator=seed_g,
                device=torch.device("cpu"))
            seq_len = image_embeds["max_seq_len"]
            image_cond = image_embeds.get("image_embeds", None)
            clip_fea = image_embeds.get("clip_context", None)

        else:  # t2v
            target_shape = image_embeds.get("target_shape", None)
            if target_shape is None:
                raise ValueError("Empty image embeds must be provided for T2V (Text to Video")
            seq_len = image_embeds["max_seq_len"]
            noise = torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=torch.device("cpu"),
                generator=seed_g)

        latent_video_length = noise.shape[1]

        if context_options is not None:
            def create_window_mask(noise_pred_context, c, latent_video_length, context_overlap, looped=False):
                window_mask = torch.ones_like(noise_pred_context)

                # Apply left-side blending for all except first chunk (or always in loop mode)
                if min(c) > 0 or (looped and max(c) == latent_video_length - 1):
                    ramp_up = torch.linspace(0, 1, context_overlap, device=noise_pred_context.device)
                    ramp_up = ramp_up.view(1, -1, 1, 1)
                    window_mask[:, :context_overlap] = ramp_up

                # Apply right-side blending for all except last chunk (or always in loop mode)
                if max(c) < latent_video_length - 1 or (looped and min(c) == 0):
                    ramp_down = torch.linspace(1, 0, context_overlap, device=noise_pred_context.device)
                    ramp_down = ramp_down.view(1, -1, 1, 1)
                    window_mask[:, -context_overlap:] = ramp_down

                return window_mask

            context_schedule = context_options["context_schedule"]
            context_frames = (context_options["context_frames"] - 1) // 4 + 2
            context_stride = context_options["context_stride"] // 4
            context_overlap = context_options["context_overlap"] // 4

            self.window_tracker = WindowTrackerSE(verbose=context_options["verbose"])

            # Get total number of prompts
            num_prompts = len(text_embeds["prompt_embeds"])
            log.info(f"Number of prompts: {num_prompts}")
            # Calculate which section this context window belongs to
            section_size = latent_video_length / num_prompts
            log.info(f"Section size: {section_size}")
            is_looped = context_schedule == "uniform_looped"

            seq_len = math.ceil((noise.shape[2] * noise.shape[3]) / 4 * context_frames)

            if context_options["freenoise"]:
                log.info("Applying FreeNoise")
                # code from AnimateDiff-Evolved by Kosinkadink (https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved)
                delta = context_frames - context_overlap
                for start_idx in range(0, latent_video_length - context_frames, delta):
                    place_idx = start_idx + context_frames
                    if place_idx >= latent_video_length:
                        break
                    end_idx = place_idx - 1

                    if end_idx + delta >= latent_video_length:
                        final_delta = latent_video_length - place_idx
                        list_idx = torch.tensor(list(range(start_idx, start_idx + final_delta)),
                                                device=torch.device("cpu"), dtype=torch.long)
                        list_idx = list_idx[torch.randperm(final_delta, generator=seed_g)]
                        noise[:, place_idx:place_idx + final_delta, :, :] = noise[:, list_idx, :, :]
                        break
                    list_idx = torch.tensor(list(range(start_idx, start_idx + delta)), device=torch.device("cpu"),
                                            dtype=torch.long)
                    list_idx = list_idx[torch.randperm(delta, generator=seed_g)]
                    noise[:, place_idx:place_idx + delta, :, :] = noise[:, list_idx, :, :]

            log.info(
                f"Context schedule enabled: {context_frames} frames, {context_stride} stride, {context_overlap} overlap")
            from .context import get_context_scheduler
            context = get_context_scheduler(context_schedule)

        if samples is not None and denoise_strength < 1.0:
            latent_timestep = timesteps[:1].to(noise)
            noise = noise * latent_timestep / 1000 + (1 - latent_timestep / 1000) * samples["samples"].squeeze(0).to(
                noise)

        if samples is not None:
            original_image = samples["samples"].clone().squeeze(0).to(device)
            mask = samples.get("mask", None)
        # latent初始化为一段噪声
        latent = noise.to(device)

        d = transformer.dim // transformer.num_heads
        freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6), L_test=latent_video_length, k=riflex_freq_index),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
            dim=1)

        if not isinstance(cfg, list):
            cfg = [cfg] * (steps + 1)

        print("Seq len:", seq_len)

        pbar = ProgressBar(steps)

        from latent_preview import prepare_callback
        callback = prepare_callback(patcher, steps)

        # blockswap init
        if model["block_swap_args"] is not None:
            for name, param in transformer.named_parameters():
                if "block" not in name:
                    param.data = param.data.to(device)
                elif model["block_swap_args"]["offload_txt_emb"] and "txt_emb" in name:
                    param.data = param.data.to(offload_device)
                elif model["block_swap_args"]["offload_img_emb"] and "img_emb" in name:
                    param.data = param.data.to(offload_device)

            transformer.block_swap(
                model["block_swap_args"]["blocks_to_swap"] - 1,
                model["block_swap_args"]["offload_txt_emb"],
                model["block_swap_args"]["offload_img_emb"],
            )
        elif model["auto_cpu_offload"]:
            for module in transformer.modules():
                if hasattr(module, "offload"):
                    module.offload()
                if hasattr(module, "onload"):
                    module.onload()
        elif model["manual_offloading"]:
            transformer.to(device)
        # feta
        if feta_args is not None:
            set_enhance_weight(feta_args["weight"])
            feta_start_percent = feta_args["start_percent"]
            feta_end_percent = feta_args["end_percent"]
            if context_options is not None:
                set_num_frames(context_frames)
            else:
                set_num_frames(latent_video_length)
            enable_enhance()
        else:
            disable_enhance()

        # Initialize TeaCache if enabled
        if teacache_args is not None:
            transformer.enable_teacache = True
            transformer.rel_l1_thresh = teacache_args["rel_l1_thresh"]
            transformer.teacache_start_step = teacache_args["start_step"]
            transformer.teacache_cache_device = teacache_args["cache_device"]
            transformer.teacache_end_step = len(timesteps) - 1 if teacache_args["end_step"] == -1 else teacache_args[
                "end_step"]
            transformer.teacache_use_coefficients = teacache_args["use_coefficients"]
        else:
            transformer.enable_teacache = False

        if slg_args is not None:
            #assert batched_cfg is not None, "Batched cfg is not supported with SLG"
            transformer.slg_blocks = slg_args["blocks"]
            transformer.slg_start_percent = slg_args["start_percent"]
            transformer.slg_end_percent = slg_args["end_percent"]
        else:
            transformer.slg_blocks = None

        mm.unload_all_models()
        mm.soft_empty_cache()
        gc.collect()

        self.teacache_state = [None, None]
        self.teacache_state_source = [None, None]
        self.teacache_states_context = []

        if "sparge" in transformer.attention_mode:
            from spas_sage_attn.autotune import (
                SparseAttentionMeansim,
                extract_sparse_attention_state_dict,
                load_sparse_attention_state_dict,
            )

            for idx, block in enumerate(transformer.blocks):
                block.self_attn.verbose = True
                block.self_attn.inner_attention = SparseAttentionMeansim(l1=0.06, pv_l1=0.065)
            if transformer.attention_mode == "spargeattn":
                try:
                    saved_state_dict = torch.load("sparge_wan.pt")
                except:
                    raise ValueError("No saved parameters found for sparse attention, tuning is required first")
                load_sparse_attention_state_dict(transformer, saved_state_dict, verbose=True)

        if flowedit_args is not None:
            source_embeds = flowedit_args["source_embeds"]
            source_image_embeds = flowedit_args.get("source_image_embeds", image_embeds)
            source_image_cond = source_image_embeds.get("image_embeds", None)
            source_clip_fea = source_image_embeds["clip_context"]
            skip_steps = flowedit_args["skip_steps"]
            drift_steps = flowedit_args["drift_steps"]
            source_cfg = flowedit_args["source_cfg"]
            if not isinstance(source_cfg, list):
                source_cfg = [source_cfg] * (steps + 1)
            drift_cfg = flowedit_args["drift_cfg"]
            if not isinstance(drift_cfg, list):
                drift_cfg = [drift_cfg] * (steps + 1)

            x_init = samples["samples"].clone().squeeze(0).to(device)
            x_tgt = samples["samples"].squeeze(0).to(device)

            sample_scheduler = FlowMatchEulerDiscreteScheduler(
                num_train_timesteps=1000,
                shift=flowedit_args["drift_flow_shift"],
                use_dynamic_shifting=False)

            sampling_sigmas = get_sampling_sigmas(steps, flowedit_args["drift_flow_shift"])

            drift_timesteps, _ = retrieve_timesteps(
                sample_scheduler,
                device=device,
                sigmas=sampling_sigmas)

            if drift_steps > 0:
                drift_timesteps = torch.cat([drift_timesteps, torch.tensor([0]).to(drift_timesteps.device)]).to(
                    drift_timesteps.device)
                timesteps[-drift_steps:] = drift_timesteps[-drift_steps:]

        def predict_with_cfg(z, cfg_scale, positive_embeds, negative_embeds, timestep, idx, image_cond=None,
                             clip_fea=None, teacache_state=None):
            with torch.autocast(device_type=mm.get_autocast_device(device), dtype=model["dtype"], enabled=True):
                base_params = {
                    'clip_fea': clip_fea,
                    'seq_len': seq_len,
                    'device': device,
                    'freqs': freqs,
                    't': timestep,
                    'current_step': idx,
                    'y': image_cond,
                }

                # Get conditional prediction
                noise_pred_cond, teacache_state_cond = transformer(
                    z,
                    context=[positive_embeds],
                    pred_id=teacache_state[0] if teacache_state else None,
                    is_uncond=False,
                    current_step_percentage=current_step_percentage,
                    **base_params
                )
                noise_pred_cond = noise_pred_cond.to(intermediate_device)

                # If cfg_scale is 1.0, return conditional prediction directly
                if math.isclose(cfg_scale, 1.0):
                    return noise_pred_cond, [teacache_state_cond]

                # Get unconditional prediction and apply cfg
                noise_pred_uncond, teacache_state_uncond = transformer(
                    z,
                    context=negative_embeds,
                    pred_id=teacache_state[1] if teacache_state else None,
                    is_uncond=True,
                    current_step_percentage=current_step_percentage,
                    **base_params
                )
                noise_pred_uncond = noise_pred_uncond.to(intermediate_device)

                return noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond), [teacache_state_cond,
                                                                                               teacache_state_uncond]

        try:
            torch.cuda.reset_peak_memory_stats(device)
        except:
            pass

        log.info(
            f"Sampling {(latent_video_length - 1) * 4 + 2} frames at {latent.shape[3] * 8}x{latent.shape[2] * 8} with {steps} steps")

        intermediate_device = device

        # diff diff prep
        masks = None

        if samples is not None and mask is not None:
            mask = 1 - mask
            thresholds = torch.arange(len(timesteps), dtype=original_image.dtype) / len(timesteps)
            thresholds = thresholds.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1).to(device)
            masks = mask.repeat(len(timesteps), 1, 1, 1, 1).to(device)
            masks = masks > thresholds

        for idx, t in enumerate(tqdm(timesteps)):
            if flowedit_args is not None:
                if idx < skip_steps:
                    continue

            # diff diff
            if masks is not None:
                if idx < len(timesteps) - 1:
                    noise_timestep = timesteps[idx + 1]
                    image_latent = sample_scheduler.scale_noise(
                        original_image, torch.tensor([noise_timestep]), noise.to(device)
                    )
                    mask = masks[idx]
                    mask = mask.to(latent)
                    latent = image_latent * mask + latent * (1 - mask)
                    # end diff diff

            latent_model_input = latent.to(device)
            timestep = torch.tensor([t]).to(device)
            current_step_percentage = idx / len(timesteps)

            # enhance-a-video
            if feta_args is not None:
                if feta_start_percent <= current_step_percentage <= feta_end_percent:
                    enable_enhance()
                else:
                    disable_enhance()
            # flow-edit
            if flowedit_args is not None:
                sigma = t / 1000.0
                sigma_prev = (timesteps[idx + 1] if idx < len(timesteps) - 1 else timesteps[-1]) / 1000.0
                noise = torch.randn(x_init.shape, generator=seed_g, device=torch.device("cpu"))
                if idx < len(timesteps) - drift_steps:
                    cfg = drift_cfg

                zt_src = (1 - sigma) * x_init + sigma * noise.to(t)
                zt_tgt = x_tgt + zt_src - x_init

                # source
                if idx < len(timesteps) - drift_steps:
                    if context_options is not None:
                        counter = torch.zeros_like(zt_src, device=intermediate_device)
                        vt_src = torch.zeros_like(zt_src, device=intermediate_device)
                        context_queue = list(
                            context(idx, steps, latent_video_length, context_frames, context_stride, context_overlap))
                        for c in context_queue:
                            window_id = self.window_tracker.get_window_id(c)

                            if teacache_args is not None:
                                current_teacache = self.window_tracker.get_teacache(window_id, self.teacache_state)
                            else:
                                current_teacache = None

                            prompt_index = min(int(max(c) / section_size), num_prompts - 1)
                            if context_options["verbose"]:
                                log.info(f"Prompt index: {prompt_index}")

                            positive = source_embeds["prompt_embeds"][prompt_index]

                            partial_img_emb = None
                            if source_image_cond is not None:
                                partial_img_emb = source_image_cond[:, c, :, :]
                                partial_img_emb[:, 0, :, :] = source_image_cond[:, 0, :, :].to(intermediate_device)

                            partial_zt_src = zt_src[:, c, :, :]
                            vt_src_context, new_teacache = predict_with_cfg(
                                partial_zt_src, cfg[idx],
                                positive, source_embeds["negative_prompt_embeds"],
                                timestep, idx, partial_img_emb,
                                source_clip_fea, current_teacache)

                            if teacache_args is not None:
                                self.window_tracker.teacache_states[window_id] = new_teacache

                            window_mask = create_window_mask(vt_src_context, c, latent_video_length, context_overlap)
                            vt_src[:, c, :, :] += vt_src_context * window_mask
                            counter[:, c, :, :] += window_mask
                        vt_src /= counter
                    else:
                        vt_src, self.teacache_state_source = predict_with_cfg(
                            zt_src, cfg[idx],
                            source_embeds["prompt_embeds"][0],
                            source_embeds["negative_prompt_embeds"],
                            timestep, idx, source_image_cond,
                            source_clip_fea,
                            teacache_state=self.teacache_state_source)
                else:
                    if idx == len(timesteps) - drift_steps:
                        x_tgt = zt_tgt
                    zt_tgt = x_tgt
                    vt_src = 0
                # target
                if context_options is not None:
                    counter = torch.zeros_like(zt_tgt, device=intermediate_device)
                    vt_tgt = torch.zeros_like(zt_tgt, device=intermediate_device)
                    context_queue = list(
                        context(idx, steps, latent_video_length, context_frames, context_stride, context_overlap))
                    for c in context_queue:
                        window_id = self.window_tracker.get_window_id(c)

                        if teacache_args is not None:
                            current_teacache = self.window_tracker.get_teacache(window_id, self.teacache_state)
                        else:
                            current_teacache = None

                        prompt_index = min(int(max(c) / section_size), num_prompts - 1)
                        if context_options["verbose"]:
                            log.info(f"Prompt index: {prompt_index}")

                        positive = text_embeds["prompt_embeds"][prompt_index]

                        partial_img_emb = None
                        if image_cond is not None:
                            partial_img_emb = image_cond[:, c, :, :]
                            partial_img_emb[:, 0, :, :] = image_cond[:, 0, :, :].to(intermediate_device)

                        partial_zt_tgt = zt_tgt[:, c, :, :]
                        vt_tgt_context, new_teacache = predict_with_cfg(
                            partial_zt_tgt, cfg[idx],
                            positive, text_embeds["negative_prompt_embeds"],
                            timestep, idx, partial_img_emb,
                            clip_fea, current_teacache)

                        if teacache_args is not None:
                            self.window_tracker.teacache_states[window_id] = new_teacache

                        window_mask = create_window_mask(vt_tgt_context, c, latent_video_length, context_overlap)
                        vt_tgt[:, c, :, :] += vt_tgt_context * window_mask
                        counter[:, c, :, :] += window_mask
                    vt_tgt /= counter
                else:
                    vt_tgt, self.teacache_state = predict_with_cfg(
                        zt_tgt, cfg[idx],
                        text_embeds["prompt_embeds"][0],
                        text_embeds["negative_prompt_embeds"],
                        timestep, idx, image_cond, clip_fea,
                        teacache_state=self.teacache_state)
                v_delta = vt_tgt - vt_src
                x_tgt = x_tgt.to(torch.float32)
                v_delta = v_delta.to(torch.float32)
                x_tgt = x_tgt + (sigma_prev - sigma) * v_delta
                x0 = x_tgt
            # context windowing
            elif context_options is not None:
                counter = torch.zeros_like(latent_model_input, device=intermediate_device)
                noise_pred = torch.zeros_like(latent_model_input, device=intermediate_device)
                context_queue = list(
                    context(idx, steps, latent_video_length, context_frames, context_stride, context_overlap))

                for c in context_queue:
                    window_id = self.window_tracker.get_window_id(c)

                    if teacache_args is not None:
                        current_teacache = self.window_tracker.get_teacache(window_id, self.teacache_state)
                    else:
                        current_teacache = None

                    prompt_index = min(int(max(c) / section_size), num_prompts - 1)
                    if context_options["verbose"]:
                        log.info(f"Prompt index: {prompt_index}")

                    # Use the appropriate prompt for this section
                    positive = text_embeds["prompt_embeds"][prompt_index]

                    partial_img_emb = None
                    if image_cond is not None:
                        partial_img_emb = image_cond[:, c, :, :]
                        partial_img_emb[:, 0, :, :] = image_cond[:, 0, :, :].to(intermediate_device)

                    partial_latent_model_input = latent_model_input[:, c, :, :]

                    noise_pred_context, new_teacache = predict_with_cfg(
                        partial_latent_model_input,
                        cfg[idx], positive,
                        text_embeds["negative_prompt_embeds"],
                        timestep, idx, partial_img_emb, clip_fea,
                        current_teacache)
                    if teacache_args is not None:
                        self.window_tracker.teacache_states[window_id] = new_teacache

                    window_mask = create_window_mask(noise_pred_context, c, latent_video_length, context_overlap,
                                                     looped=is_looped)
                    noise_pred[:, c, :, :] += noise_pred_context * window_mask
                    counter[:, c, :, :] += window_mask
                noise_pred /= counter
            # normal inference
            else:
                noise_pred, self.teacache_state = predict_with_cfg(
                    latent_model_input,
                    cfg[idx],
                    text_embeds["prompt_embeds"][0],
                    text_embeds["negative_prompt_embeds"],
                    timestep, idx, image_cond, clip_fea,
                    teacache_state=self.teacache_state)
            if flowedit_args is None:
                latent = latent.to(intermediate_device)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latent.unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latent = temp_x0.squeeze(0)

                x0 = latent.to(device)
                if callback is not None:
                    callback_latent = (latent_model_input - noise_pred.to(t.device) * t / 1000).detach().permute(1, 0,
                                                                                                                 2, 3)
                    callback(idx, callback_latent, None, steps)
                else:
                    pbar.update(1)
                del latent_model_input, timestep
            else:
                if callback is not None:
                    callback_latent = (zt_tgt - vt_tgt.to(t.device) * t / 1000).detach().permute(1, 0, 2, 3)
                    callback(idx, callback_latent, None, steps)
                else:
                    pbar.update(1)

        if teacache_args is not None:
            states = transformer.teacache_state.states
            state_names = {
                0: "conditional",
                1: "unconditional"
            }
            for pred_id, state in states.items():
                name = state_names.get(pred_id, f"prediction_{pred_id}")
                if 'skipped_steps' in state:
                    log.info(f"TeaCache skipped: {state['skipped_steps']} {name} steps")
            transformer.teacache_state.clear_all()

        if transformer.attention_mode == "spargeattn_tune":
            saved_state_dict = extract_sparse_attention_state_dict(transformer)
            torch.save(saved_state_dict, "sparge_wan.pt")
            save_torch_file(saved_state_dict, "sparge_wan.safetensors")

        if force_offload:
            if model["manual_offloading"]:
                transformer.to(offload_device)
                mm.soft_empty_cache()
                gc.collect()

        try:
            print_memory(device)
            torch.cuda.reset_peak_memory_stats(device)
        except:
            pass

        return ({
                    "samples": x0.unsqueeze(0).cpu()
                },)



class WanVideoSEDecode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "vae": ("WANVAE",),
                    "samples": ("LATENT",),
                    "start": ("IMAGE",),
                    "end": ("IMAGE",),
                    "enable_vae_tiling": ("BOOLEAN", {"default": True, "tooltip": "Drastically reduces memory use but may introduce seams"}),
                    "tile_x": ("INT", {"default": 272, "min": 64, "max": 2048, "step": 1, "tooltip": "Tile size in pixels, smaller values use less VRAM, may introduce more seams"}),
                    "tile_y": ("INT", {"default": 272, "min": 64, "max": 2048, "step": 1, "tooltip": "Tile size in pixels, smaller values use less VRAM, may introduce more seams"}),
                    "tile_stride_x": ("INT", {"default": 144, "min": 32, "max": 2048, "step": 32, "tooltip": "Tile stride in pixels, smaller values use less VRAM, may introduce more seams"}),
                    "tile_stride_y": ("INT", {"default": 128, "min": 32, "max": 2048, "step": 32, "tooltip": "Tile stride in pixels, smaller values use less VRAM, may introduce more seams"}),
                    },
                }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "decode"
    CATEGORY = "WanVideoStartEndFrame"

    def decode(self, vae, samples, start, end, enable_vae_tiling, tile_x, tile_y, tile_stride_x, tile_stride_y):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        mm.soft_empty_cache()
        latents = samples["samples"]
        vae.to(device)

        latents = latents.to(device = device, dtype = vae.dtype)

        mm.soft_empty_cache()

        image = vae.decode(latents, device=device, end_=True, tiled=enable_vae_tiling, tile_size=(tile_x, tile_y), tile_stride=(tile_stride_x, tile_stride_y))[0]
        vae.to(offload_device)
        vae.model.clear_cache()
        mm.soft_empty_cache()

        image = (image - image.min()) / (image.max() - image.min())
        image = torch.clamp(image, 0.0, 1.0)
        h,w=image.shape[2],image.shape[3]
        end = common_upscale(end.movedim(-1, 1), w, h, "lanczos", "disabled").squeeze(0).unsqueeze(1)
        start = common_upscale(start.movedim(-1, 1), w, h, "lanczos", "disabled").squeeze(0).unsqueeze(1)
        image=torch.cat([start,image[:,1:-4,:,:],end],dim=1)
        image = image.permute(1, 2, 3, 0).cpu().float()
        #print(image.shape,end.shape)
        return (image,)





NODE_CLASS_MAPPINGS = {
    "WanVideoSEModelLoader": WanVideoSEModelLoader,
    "WanVideoSEVAELoader": WanVideoSEVAELoader,
    "WanVideoSEImageClipEncode": WanVideoSEImageClipEncode,
    "WanVideoSEDecode": WanVideoSEDecode,
    "WanVideoSESampler":WanVideoSESampler,
    }
NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoSEModelLoader": "WanVideo Model Loader(SE)",
    "WanVideoSEVAELoader": "WanVideo VAE Loader(SE)",
    "WanVideoSEImageClipEncode":"WanVideo SEI Clip Encode",
    "WanVideoSEDecode": "Wan Video SEI Decode",
    "WanVideoSESampler": "Wan Video Sampler(SE)",
    }
