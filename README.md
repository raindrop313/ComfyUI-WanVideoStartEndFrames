# ComfyUI-WanVideoStartEndFrames
ComfyUI nodes that support video generation by start and end frames

# Start
This project is a node-based implementation for video generation using the Wan2.1 model, with a focus on start and end frame guidance. The source code is a modification of Kijai's nodes code, so for model download and installation instructions, please refer to [ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper). This project specifically adds the functionality of start and end frame guided video generation.

The nodes support Wan2.1 models in both 720P and 480P versions. It is recommended to generate videos with a frame count of 25 or higher, as a lower frame count may affect the consistency of character identity.

Currently, the start and end frame video generation approach is in its early stages. It primarily implements the start and end frame video generation functionality at the code level and does not yet involve model or LoRA fine-tuning, which is planned for future work. Additionally, incorporating end frame guidance in Image-to-Video (I2V) seems to degrade video generation quality, which is another area for future improvement.

I welcome discussions in the issues section and extend our gratitude to Kijai for the open-source nodes.

Note: Video generation should ideally be accompanied by positive prompts. Currently, the absence of positive prompts can result in severe video distortion.


# Changelog
- 2025.3.20: Added start and end frame weight controls for video transitions
- 2025.3.22: Compatible with SLG functionality in KJ's nodes


# Examples
Start Frame:
![start_frame_](https://github.com/user-attachments/assets/6c301578-56ae-45c7-8d1c-9ac5f727bf53)
End Frame:
![end_frame](https://github.com/user-attachments/assets/97de3844-e974-4be9-9157-0785c564574d)
prompt:
两个角色搀扶着往前走，并看向彼此

Frame count:81+1=82

Video output:

-720P:

https://github.com/user-attachments/assets/948b70c7-172b-4754-8453-cd6f78b0338a


-480P:

https://github.com/user-attachments/assets/09c224e3-ac17-4621-bfcd-a8d449b8720e






