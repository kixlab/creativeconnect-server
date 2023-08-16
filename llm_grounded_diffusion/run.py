import gradio as gr
import numpy as np
from PIL import Image
import ast
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
from utils.parse import filter_boxes
from generation import run as run_ours
from baseline import run as run_baseline
import torch
from shared import DEFAULT_SO_NEGATIVE_PROMPT, DEFAULT_OVERALL_NEGATIVE_PROMPT
from examples import stage1_examples, stage2_examples

from time import time
from datetime import datetime
from controlnet_pipeline import controlnet_run, mask_to_segment, overlay_segment



print(f"Is CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")

now = datetime.now().microsecond

box_scale = (512, 512)
size = box_scale

bg_prompt_text = "Background prompt: "

default_template = """You are an intelligent bounding box generator. I will provide you with a caption for a photo, image, or painting. Your task is to generate the bounding boxes for the objects mentioned in the caption, along with a background prompt describing the scene. The images are of size 512x512, and the bounding boxes should not overlap or go beyond the image boundaries. Each bounding box should be in the format of (object name, [top-left x coordinate, top-left y coordinate, box width, box height]) and include exactly one object. Make the boxes larger if possible. Do not put objects that are already provided in the bounding boxes into the background prompt. If needed, you can make reasonable guesses. Generate the object descriptions and background prompts in English even if the caption might not be in English. Do not include non-existing or excluded objects in the background prompt. Please refer to the example below for the desired format.

Caption: A realistic image of landscape scene depicting a green car parking on the left of a blue truck, with a red air balloon and a bird in the sky
Objects: [('a green car', [21, 181, 211, 159]), ('a blue truck', [269, 181, 209, 160]), ('a red air balloon', [66, 8, 145, 135]), ('a bird', [296, 42, 143, 100])]
Background prompt: A realistic image of a landscape scene

Caption: A watercolor painting of a wooden table in the living room with an apple on it
Objects: [('a wooden table', [65, 243, 344, 206]), ('a apple', [206, 306, 81, 69])]
Background prompt: A watercolor painting of a living room

Caption: A watercolor painting of two pandas eating bamboo in a forest
Objects: [('a panda eating bambooo', [30, 171, 212, 226]), ('a panda eating bambooo', [264, 173, 222, 221])]
Background prompt: A watercolor painting of a forest

Caption: A realistic image of four skiers standing in a line on the snow near a palm tree
Objects: [('a skier', [5, 152, 139, 168]), ('a skier', [278, 192, 121, 158]), ('a skier', [148, 173, 124, 155]), ('a palm tree', [404, 180, 103, 180])]
Background prompt: A realistic image of an outdoor scene with snow

Caption: An oil painting of a pink dolphin jumping on the left of a steam boat on the sea
Objects: [('a steam boat', [232, 225, 257, 149]), ('a jumping pink dolphin', [21, 249, 189, 123])]
Background prompt: An oil painting of the sea

Caption: A realistic image of a cat playing with a dog in a park with flowers
Objects: [('a playful cat', [51, 67, 271, 324]), ('a playful dog', [302, 119, 211, 228])]
Background prompt: A realistic image of a park with flowers"""

simplified_prompt = """{template}

Caption: {prompt}
Objects: """

prompt_placeholder = "A realistic photo of a gray cat and an orange dog on the grass."

layout_placeholder = """Caption: A realistic photo of a gray cat and an orange dog on the grass.
Objects: [('a gray cat', [67, 243, 120, 126]), ('an orange dog', [265, 193, 190, 210])]
Background prompt: A realistic photo of a grassy area."""

def get_ours_image(response, overall_prompt_override="", seed=0, num_inference_steps=20, dpm_scheduler=True, use_autocast=False, 
                   fg_seed_start=20, fg_blending_ratio=0.1, frozen_step_ratio=0.4, gligen_scheduled_sampling_beta=0.3, 
                   so_negative_prompt=DEFAULT_SO_NEGATIVE_PROMPT, overall_negative_prompt=DEFAULT_OVERALL_NEGATIVE_PROMPT, show_so_imgs=False, scale_boxes=False,
                   obj_latents_list=[], ctrl_img_list=[], boxes_list=[]):
    if response == "":
        response = layout_placeholder
    gen_boxes, bg_prompt = parse_input(response)
    gen_boxes = filter_boxes(gen_boxes, scale_boxes=scale_boxes) # ********** list형태: ('a gray cat', (67, 243, 120, 126))
    spec = {
        # prompt is unused
        'prompt': '',
        'gen_boxes': gen_boxes, 
        'bg_prompt': bg_prompt
    }
    # ********** spec for controlnet objects
    ctrl_spec = {
        # prompt is unused
        'prompt': '',
        'gen_boxes': boxes_list, 
        'bg_prompt': bg_prompt
    }
    
    if dpm_scheduler:
        scheduler_key = "dpm_scheduler"
    else:
        scheduler_key = "scheduler"
        
    image_np, so_img_list = run_ours(
        spec, bg_seed=seed, overall_prompt_override=overall_prompt_override, fg_seed_start=fg_seed_start, 
        fg_blending_ratio=fg_blending_ratio,frozen_step_ratio=frozen_step_ratio, use_autocast=use_autocast,
        gligen_scheduled_sampling_beta=gligen_scheduled_sampling_beta, num_inference_steps=num_inference_steps, scheduler_key=scheduler_key,
        so_negative_prompt=so_negative_prompt, overall_negative_prompt=overall_negative_prompt, so_batch_size=2,
        obj_latents_list=obj_latents_list, ctrl_img_list=ctrl_img_list, boxes_list=boxes_list, ctrl_spec=ctrl_spec
    )
    images = [image_np]
    if show_so_imgs:
        images.extend([np.asarray(so_img) for so_img in so_img_list])
    return images

def parse_input(text=None):
    try:
        if "Objects: " in text:
            text = text.split("Objects: ")[1]
            
        text_split = text.split(bg_prompt_text)
        if len(text_split) == 2:
            gen_boxes, bg_prompt = text_split
        gen_boxes = ast.literal_eval(gen_boxes)    
        bg_prompt = bg_prompt.strip()
    except Exception as e:
        raise gr.Error(f"response format invalid: {e} (text: {text})")
    
    return gen_boxes, bg_prompt

def recombination():
    start_time = time()

    prompt = """Caption: Gray cat and a soccer ball on the grass, line drawing.
Objects: [('a gray cat', [67, 243, 120, 126]), ('a soccer ball', [265, 193, 190, 210])]
Background prompt: A grassy area."""

    img = get_ours_image(response=prompt)
    output = Image.fromarray(img[0])
    
    # ctrl_img, masks = controlnet_run(now, verbose=True) # obj_latents_list, ctrl_img_list, boxes_list = 
    # segments = mask_to_segment(ctrl_img, masks)
    # output = overlay_segment(img[0], segments, masks)

    print(f'Latency: {time()-start_time}')

    output.save(f'{now}_res.png')
    print(f'Saved {now}_res.png...')
    
    return output


if __name__ == '__main__':
    recombination()

