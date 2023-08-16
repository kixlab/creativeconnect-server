import gradio as gr
import numpy as np
import cv2
import torch
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline
from diffusers.utils import load_image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import UniPCMultistepScheduler

from models import sam
from shared import model_dict, sam_model_dict


model_dict.update(sam_model_dict)

# Hyperparams
height = 512  # default height of Stable Diffusion
width = 512  # default width of Stable Diffusion
H, W = height // 8, width // 8 # size of the latent
guidance_scale = 7.5  # Scale for classifier-free guidance

# batch size that is not 1 is not supported
overall_batch_size = 1

# discourage masks with confidence below
discourage_mask_below_confidence = 0.85

# discourage masks with iou (with coarse binarized attention mask) below
discourage_mask_below_coarse_iou = 0.25




def controlnet_run(now, verbose=True):

    '''
        vae.decode할 때, latents를 나눠주는 값: self.vae.config.scaling_factor = 0.18215

        ********** 
        Input: 
            white(or black) background 512*512 image contained segment; 
            bbox coordination from the layout;
            segment mask from SAM;
            text prompt;
        Output:
            latent vector;
            boxes_list (text prompt, bbox coordination); 
                - it should be matched with (object name, [top-left x coordinate, top-left y coordinate, box width, box height]);
            masks_list contains SAM mask;
    '''

    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    # this command loads the individual model components on GPU on-demand.
    pipe.enable_model_cpu_offload()

    # 
    ctrl_latents_list = []
    ctrl_img_list=[]
    boxes_list = []

    # Let's load the popular vermeer image; 
    image = load_image(
        'base.png'
    )
    object_name = 'cute dog'
    box_coord = [[10, 10, 487, 229]]

    output_type = 'pil' # both
    num_images_per_prompt = 1

    image = np.array(image)

    low_threshold = 100
    high_threshold = 300

    canny_image = cv2.Canny(image, low_threshold, high_threshold)
    canny_image = canny_image[:, :, None]
    canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
    canny_image = Image.fromarray(canny_image)

    generator = torch.manual_seed(15)

    # TODO 5개일 때에도 고려해야 함.
    output = pipe(
        object_name, num_inference_steps=20, generator=generator, image=canny_image, output_type=output_type, num_images_per_prompt=num_images_per_prompt,
    )
    
    if verbose:
        merged = Image.new('RGB', (1024, 512), (255, 255, 255))
        for pil in output.images:
            merged.paste(pil, (0, 0))
            merged.paste(canny_image, (512, 0))
        
        merged.save(f'controlnet_{now}.png')
        print(f'Saved controlnet_{now}.png ...')

    try:
        boxes_list.append(
            (object_name, box_coord)
        )
        ctrl_img_list = list(map(lambda x:np.array(x), output.images))
        if output_type in ['both', 'latent']:
            ctrl_latents_list = output.latents.cpu()
            return ctrl_latents_list, ctrl_img_list, boxes_list
    except Exception as err:
        print(f'Exception')


    # TODO implement mask generation
    sam_refine_kwargs = dict(
        discourage_mask_below_confidence=discourage_mask_below_confidence, discourage_mask_below_coarse_iou=discourage_mask_below_coarse_iou,
        height=height, width=width, H=height, W=width
    )

    sam_box_coord = [[tuple(map(lambda x: x/height, coord)) for coord in box_coord]]
    sam_img_arr = np.stack([np.array(img) for img in ctrl_img_list])
    mask_selected, _ = sam.sam_refine_boxes(sam_input_images=ctrl_img_list, boxes=sam_box_coord, model_dict=model_dict, verbose=verbose, **sam_refine_kwargs)

    return sam_img_arr[0], mask_selected[0]

def mask_to_segment(img, masks) -> np.array:
    masked_img = []
    for mask in masks:
        mask_arr = (mask * 255).astype(np.uint8)
        img_arr = img
        masked_img.append(img_arr * (mask_arr[:, :, None] / 255).astype(img_arr.dtype))

    return masked_img

def overlay_segment(img_arr, segment_arr_list, mask_arr_list, verbose=True):
    """
    Overlay a segment image onto a background image at the given position.

    Args:
    - background_path (str): Path to the background image.
    - segment_path (str): Path to the segment image (unstructured shape with 0 background).
    - output_path (str): Path to save the output image.
    - position (tuple): (x, y) coordinates where the segment image should be placed.
    """
    background = Image.fromarray(img_arr)
    segments = [Image.fromarray(segment_arr) for segment_arr in segment_arr_list]

    for segment, mask in zip(segments, mask_arr_list):
        # Create a mask using the alpha channel of the segment image or by its luminance
        mask_int = (mask * 255).astype(np.uint8)
        mask_pil = Image.fromarray(mask_int)
        # Paste the segment onto the background using the mask to identify the area to replace
        background.paste(segment, (0,0), mask_pil)
    
    if verbose:
        background.save('overlay.png')
    
    return background