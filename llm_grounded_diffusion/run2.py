import gradio as gr
import numpy as np
import cv2
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

print(f"Is CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")

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
Background prompt: A realistic image of a park with flowers

Caption: 一个客厅场景的油画，墙上挂着电视，电视下面是一个柜子，柜子上有一个花瓶。
Objects: [('a tv', [88, 85, 335, 203]), ('a cabinet', [57, 308, 404, 201]), ('a flower vase', [166, 222, 92, 108])]
Background prompt: An oil painting of a living room scene"""

simplified_prompt = """{template}

Caption: {prompt}
Objects: """

prompt_placeholder = "A realistic photo of a gray cat and an orange dog on the grass."

layout_placeholder = """Caption: A realistic photo of a gray cat and an orange dog on the grass.
Objects: [('a gray cat', [67, 243, 120, 126]), ('an orange dog', [265, 193, 190, 210])]
Background prompt: A realistic photo of a grassy area."""

def get_lmd_prompt(prompt, template=default_template):
    if prompt == "":
        prompt = prompt_placeholder
    if template == "":
        template = default_template
    return simplified_prompt.format(template=template, prompt=prompt)

def get_layout_image(response):
    if response == "":
        response = layout_placeholder
    gen_boxes, bg_prompt = parse_input(response)
    fig = plt.figure(figsize=(8, 8))
    # https://stackoverflow.com/questions/7821518/save-plot-to-numpy-array
    show_boxes(gen_boxes, bg_prompt)
    # If we haven't already shown or saved the plot, then we need to
    # draw the figure first...
    fig.canvas.draw()

    # Now we can save it to a numpy array.
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.clf()
    return data

def get_layout_image_gallery(response):
    return [get_layout_image(response)]

def get_ours_image(response, overall_prompt_override="", seed=0, num_inference_steps=20, dpm_scheduler=True, use_autocast=False, fg_seed_start=20, fg_blending_ratio=0.1, frozen_step_ratio=0.4, gligen_scheduled_sampling_beta=0.3, so_negative_prompt=DEFAULT_SO_NEGATIVE_PROMPT, overall_negative_prompt=DEFAULT_OVERALL_NEGATIVE_PROMPT, show_so_imgs=False, scale_boxes=False):
    if response == "":
        response = layout_placeholder
    gen_boxes, bg_prompt = parse_input(response)
    gen_boxes = filter_boxes(gen_boxes, scale_boxes=scale_boxes)
    spec = {
        # prompt is unused
        'prompt': '',
        'gen_boxes': gen_boxes,
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
        so_negative_prompt=so_negative_prompt, overall_negative_prompt=overall_negative_prompt, so_batch_size=2
    )
    images = [image_np]
    if show_so_imgs:
        images.extend([np.asarray(so_img) for so_img in so_img_list])
    return images

def get_baseline_image(prompt, seed=0):
    if prompt == "":
        prompt = prompt_placeholder
    
    scheduler_key = "dpm_scheduler"
    num_inference_steps = 20
    
    image_np = run_baseline(prompt, bg_seed=seed, scheduler_key=scheduler_key, num_inference_steps=num_inference_steps)
    return [image_np]

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

def draw_boxes(anns):
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in anns:
        c = (np.random.random((1, 3))*0.6+0.4)
        [bbox_x, bbox_y, bbox_w, bbox_h] = ann['bbox']
        poly = [[bbox_x, bbox_y], [bbox_x, bbox_y+bbox_h],
                [bbox_x+bbox_w, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y]]
        np_poly = np.array(poly).reshape((4, 2))
        polygons.append(Polygon(np_poly))
        color.append(c)

        # print(ann)
        name = ann['name'] if 'name' in ann else str(ann['category_id'])
        ax.text(bbox_x, bbox_y, name, style='italic',
                bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 5})

    p = PatchCollection(polygons, facecolor='none',
                        edgecolors=color, linewidths=2)
    ax.add_collection(p)


def show_boxes(gen_boxes, bg_prompt=None):
    anns = [{'name': gen_box[0], 'bbox': gen_box[1]}
            for gen_box in gen_boxes]

    # White background (to allow line to show on the edge)
    I = np.ones((size[0]+4, size[1]+4, 3), dtype=np.uint8) * 255

    plt.imshow(I)
    plt.axis('off')

    if bg_prompt is not None:
        ax = plt.gca()
        ax.text(0, 0, bg_prompt, style='italic',
                bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 5})

        c = np.zeros((1, 3))
        [bbox_x, bbox_y, bbox_w, bbox_h] = (0, 0, size[1], size[0])
        poly = [[bbox_x, bbox_y], [bbox_x, bbox_y+bbox_h],
                [bbox_x+bbox_w, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y]]
        np_poly = np.array(poly).reshape((4, 2))
        polygons = [Polygon(np_poly)]
        color = [c]
        p = PatchCollection(polygons, facecolor='none',
                            edgecolors=color, linewidths=2)
        ax.add_collection(p)

    draw_boxes(anns)


def main():
    prompt = ""
    
    # Stage 1: LLM to bounding box and corresponding label generation
    # formatted_prompt = get_lmd_prompt(prompt=prompt)

    # layout_prompt= chatgpt(formatted_prompt)

    # layout_image = get_layout_image_gallery(layout_prompt)

    # cv2.imwrite('tmp.png', layout_image[0])

    img = get_ours_image(response="",)

    img = cv2.cvtColor(img[0], cv2.COLOR_BGR2RGB)
    cv2.imwrite('res.png', img)
    


    # Stage 2: Composed image generation

if __name__ == '__main__':
    main()
