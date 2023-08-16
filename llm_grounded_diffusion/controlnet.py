import cv2
from PIL import Image
import numpy as np
from diffusers import StableDiffusionControlNetPipeline
from diffusers.utils import load_image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import UniPCMultistepScheduler
import torch

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# this command loads the individual model components on GPU on-demand.
pipe.enable_model_cpu_offload()


# Let's load the popular vermeer image
image = load_image(
   'base.png'
)
output_type = 'pil'
num_images_per_prompt = 5

image = np.array(image)

low_threshold = 100
high_threshold = 200

image = cv2.Canny(image, low_threshold, high_threshold)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)

generator = torch.manual_seed(0)

out_image = pipe(
    "dog", num_inference_steps=20, generator=generator, image=canny_image, output_type=output_type, num_images_per_prompt=num_images_per_prompt,
).images

if output_type=='pil' and num_images_per_prompt==1:
    out_image[0].save('controlnet.png')
elif output_type=='pil' and num_images_per_prompt!=1:
    for n in range(num_images_per_prompt):
        out_image[n].save(f'controlnet_{n}.png')