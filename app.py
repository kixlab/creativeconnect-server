from flask import Flask, request
from flask_cors import CORS
from datetime import datetime
import base64, ast, torch, cv2
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from gpt_prompts import (
    caption_to_layout,
    keywords_to_descriptions,
    keywords_expansion,
    caption_to_keywords,
    caption_layout_matcher,
)
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from diffusers import StableDiffusionGLIGENPipeline
from style_module.style_transfer import line_drawing_predict
from layout_module.layout_metrics import generate_layouts, xywh_to_xyxy

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# blipprocessor = None
# blipmodel = None
# sam = None
# mask_generator = None

# Setup BLIP model
blipprocessor = Blip2Processor.from_pretrained(
    "Salesforce/blip2-opt-2.7b", low_cpu_mem_usage=True
)
blipmodel = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16, low_cpu_mem_usage=True
).to(DEVICE)

gligen = StableDiffusionGLIGENPipeline.from_pretrained(
    "masterful/gligen-1-4-generation-text-box",
    variant="fp16",
    torch_dtype=torch.float16,
).to(DEVICE)

# Setup SAM model
MODEL_TYPE = "vit_l"
CHECKPOINT_PATH = "./checkpoints/sam_vit_l_0b3195.pth"
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(DEVICE)
mask_generator = SamAutomaticMaskGenerator(sam)

app = Flask(__name__, static_folder="./generated")
CORS(app, resources={r"/*": {"origins": "*"}})

from log import create_log_api

app.register_blueprint(create_log_api(), url_prefix="/log")

# Util functions


def keywordlist_to_string(keywords):
    matters = []
    actions = []
    themes = []

    for e in keywords:
        if e["type"] == "Subject matter":
            matters.append(e)
        elif e["type"] == "Action & pose":
            actions.append(e)
        elif e["type"] == "Theme & mood":
            themes.append(e)

    res = ""
    if len(matters) > 0:
        res += (
            "Subject matter: "
            + ", ".join([o["keyword"].split("-")[0] for o in matters])
            + "\n"
        )
    if len(actions) > 0:
        res += "Action & pose: " + ", ".join([a["keyword"] for a in actions]) + "\n"
    if len(themes) > 0:
        res += "Theme & mood: " + ", ".join([c["keyword"] for c in themes])
    res = res.strip()

    return res


def prompt_to_recombined_images(input_prompt, gen_num=1):
    '''
    Input: prompt
        e.g. """Caption: a waterfall and a modern high speed train running through the tunnel in a beautiful forest with fall foliage.
                Objects: [('a waterfall', [0.1387, 0.2051, 0.4277, 0.7090]), ('a modern high speed train running through the tunnel', [0.4980, 0.4355, 0.8516, 0.7266])]
            """
            % Now: input bbox is xywh format, ratio. <-- It can be change!!
    Output:
        - image_path_raw: generated raw image path list
        - image_path_sketch: generated sketch image path list
    '''
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    image_path_sketch_list = [
        "generated/" + f"{timestamp}_{i}_sketch.png" for i in range(gen_num)
    ]
    image_path_raw_list = [
        "generated/" + f"{timestamp}_{i}_raw.png" for i in range(gen_num)
    ]

    def parse_input(text=None):
        try:
            if "Objects: " in text:
                caption, objects = text.split("Objects: ")
            caption = caption.replace("Caption: ", "")
            objects = ast.literal_eval(objects)
        except Exception as e:
            raise Exception(f"response format invalid: {e} (text: {text})")

        return caption, objects

    input_prompt = parse_input(input_prompt)

    prompt = input_prompt[0]
    boxes = xywh_to_xyxy(
        [i[1] for i in input_prompt[1]]
    )  # [[0.1387, 0.2051, 0.4277, 0.7090], [0.4980, 0.4355, 0.8516, 0.7266]]
    phrases = [
        i[0] for i in input_prompt[1]
    ]  # ["a waterfall", "a modern high speed train running through the tunnel"]

    images = gligen(
        prompt=prompt,
        gligen_phrases=phrases,
        gligen_boxes=boxes,
        gligen_scheduled_sampling_beta=1,
        output_type="pil",
        num_inference_steps=15,
        num_images_per_prompt=gen_num,
    ).images

    for image, image_path_raw, image_path_sketch in zip(
        images, image_path_raw_list, image_path_sketch_list
    ):
        image.save(image_path_raw)
        output_sketch = line_drawing_predict(image, ver="Simple Lines")
        output_sketch.save(image_path_sketch)

    return image_path_raw_list, image_path_sketch_list


# Routes


@app.route("/setup", methods=["GET"])
def setup():
    # global blipprocessor, blipmodel, sam, mask_generator

    # # Setup BLIP model
    # blipprocessor = Blip2Processor.from_pretrained(
    #     "Salesforce/blip2-opt-2.7b", low_cpu_mem_usage=True
    # )
    # blipmodel = Blip2ForConditionalGeneration.from_pretrained(
    #     "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16, low_cpu_mem_usage=True
    # ).to(DEVICE)

    # # Setup SAM model
    # MODEL_TYPE = "vit_l"
    # CHECKPOINT_PATH = "./checkpoints/sam_vit_l_0b3195.pth"
    # sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    # sam.to(DEVICE)
    # mask_generator = SamAutomaticMaskGenerator(sam)

    return "setup done"


@app.route("/", methods=["GET"])
def test():
    return "hello, world!"


@app.route("/sendImage", methods=["POST"])
def store_image():
    """
    Input:
        - image: base64 encoded image

    Output:
        - filename: filename of uploaded image
    """
    data = request.get_json()
    image = data.get("image")
    if not image:
        return {"message": "No file received"}, 400

    # Save image
    file_content = image.split(";base64,")[1]
    file_extension = image.split(";")[0].split("/")[1]
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{timestamp}.{file_extension}"

    with open("uploaded/" + filename, "wb") as f:
        f.write(base64.b64decode(file_content))

    return {"filename": filename}, 200


# Image --> Keyword list
@app.route("/imageToKeywords", methods=["POST"])
def extract_element_from_image():
    """
    Input:
        - image: base64 encoded image

    Output:
        - filename: filename of uploaded image
        - keywords: list of keywords
    """
    data = request.get_json()
    image = data.get("image")
    if not image:
        return {"message": "No file received"}, 400

    # Save image
    file_content = image.split(";base64,")[1]
    file_extension = image.split(";")[0].split("/")[1]
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{timestamp}.{file_extension}"

    with open("uploaded/" + filename, "wb") as f:
        f.write(base64.b64decode(file_content))

    # Load image
    IMAGE_PATH = "uploaded/" + filename
    image_bgr = cv2.imread(IMAGE_PATH)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Get whole image description
    segmented_descriptions = []
    inputs = blipprocessor(images=image_rgb, return_tensors="pt").to(
        DEVICE, torch.float16
    )
    generated_ids = blipmodel.generate(**inputs)
    whole_image_description = blipprocessor.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0].strip()
    segmented_descriptions.append(whole_image_description)

    # Get segmented image descriptions
    image_segments = []
    for i in range(3):
        for j in range(3):
            w = int(image_rgb.shape[0] / 2)
            h = int(image_rgb.shape[1] / 2)
            x = int(image_rgb.shape[0] / 4 * i)
            y = int(image_rgb.shape[1] / 4 * j)
            image_segments.append(image_rgb[x : x + w, y : y + h])
    for seg in image_segments:
        inputs = blipprocessor(images=seg, return_tensors="pt").to(
            DEVICE, torch.float16
        )
        generated_ids = blipmodel.generate(**inputs)
        generated_text = blipprocessor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()
        segmented_descriptions.append(generated_text)

    segmented_descriptions = "\n".join(list(set(segmented_descriptions)))
    
    matters, _, _ = caption_to_keywords(whole_image_description)
    _, actions, themes = caption_to_keywords(segmented_descriptions)

    keywords = []

    for matter in matters:
        keywords.append({"type": "Subject matter", "keyword": matter, "position": None})

    for action in actions:
        keywords.append({"type": "Action & pose", "keyword": action, "position": None})
    for theme in themes:
        keywords.append({"type": "Theme & mood", "keyword": theme, "position": None})

    return {"filename": filename, "keywords": keywords}, 200


# Image --> Layout
@app.route("/imageToLayout", methods=["POST"])
def extract_layout_from_image():
    """
    Input:
        - filename: filename of uploaded image

    Output:
        - bboxes: list of bounding boxes (proportion of image size)
    """

    data = request.get_json()
    filename = data.get("filename")
    if not filename:
        return {"message": "No file received"}, 400

    # Load image
    print("Loading image")
    IMAGE_PATH = "uploaded/" + filename
    image_bgr = cv2.imread(IMAGE_PATH)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    width, height = image_rgb.shape[1], image_rgb.shape[0]

    masks = mask_generator.generate(image_rgb)

    # return only bbox
    bboxes = []
    max_mask_size = max([m["area"] for m in masks])
    # sorted_masks = sorted(masks, key=(lambda x: x['predicted_iou']), reverse=True)
    sorted_masks = sorted(
        masks,
        key=(
            lambda x: x["area"]
            - (x["stability_score"] < 0.2) * max_mask_size
            - (x["predicted_iou"] < 0.2) * max_mask_size
        ),
        reverse=True,
    )
    i = 0
    while len(bboxes) < 5 and i < len(sorted_masks):
        mask = sorted_masks[i]
        if mask["area"] < 0.7 * width * height:
            x, y, w, h = mask["bbox"]
            bboxes.append((x / width, y / height, w / width, h / height))
        i += 1

    return {"bboxes": bboxes}, 200


# Layout --> Layout
@app.route("/getRecommendedLayouts", methods=["POST"])
def recommend_layouts():
    """
    Input:
        - layout: list of xywh format bounding boxes (512*512)

    Output:
        - layouts: 2-dim list of 10 recommended layouts (512*512)
    """

    data = request.get_json()
    old_layout = data.get("layout")
    layouts = []
    # for i in range(1, len(old_layout)+1):
    #     layouts.append(generate_layouts(old_layout, recommends_num=10, target_bbox_num=i))
    target_bbox_num = data.get("target_bbox_num")
    layouts = generate_layouts(
        old_layout, recommends_num=10, target_bbox_num=target_bbox_num
    )

    return {"layouts": layouts}, 200


# Layout diffusion
@app.route("/generateImage", methods=["POST"])
def generate_recombined_images():
    '''
    Input:
        - prompt: prompt for layout diffusion
            e.g., """Caption: Gray cat and a soccer ball on the grass, line drawing.
                    Objects: [('a gray cat', [67, 243, 120, 126]), ('a soccer ball', [265, 193, 190, 210])]
                    """
        - gen_num: number of images from single prompt

    Output:
        - image_path_raw: generated raw image path list
        - image_path_sketch: generated sketch image path list
    '''
    data = request.get_json()
    prompt = data.get("prompt")
    generation_image_num = data.get("gen_num")

    image_path_raw_list, image_path_sketch_list = prompt_to_recombined_images(
        prompt, gen_num=generation_image_num
    )

    return {
        "image_path_raw": image_path_raw_list,
        "image_path_sketch": image_path_sketch_list,
    }, 200


# Keyword list --> Expanded keyword list
@app.route("/expandElements", methods=["POST"])
def generate_elements_from_elements():
    """
    Input:
        - originalKeywords: list of original elements

    Output:
        - suggestedKeywords: list of expanded elements
    """
    data = request.get_json()
    originalKeywords = data.get("originalKeywords")
    keywordString = keywordlist_to_string(originalKeywords)

    suggestedKeywords = keywords_expansion(keywordString)

    return {"suggestedKeywords": suggestedKeywords}, 200


# Keyword list --> Descriptions only
@app.route("/mergeKeywordsToDescriptions", methods=["POST"])
def merge_elements_to_desc():
    data = request.get_json()
    keywords = data.get("keywords")
    keywordString = keywordlist_to_string(keywords)
    generatedDescriptions = keywords_to_descriptions(keywordString)

    return {"descriptions": generatedDescriptions}, 200


# Descriptions --> Layouts & Sketches
@app.route("/descriptionToSketch", methods=["POST"])
def descriptionToSketch():
    data = request.get_json()
    description = data.get("description")

    objects = list(description["objects"].keys())
    print(objects)
    layout = caption_to_layout(description["caption"], objects)
    print(layout)
    prompt = (
        "Caption: "
        + description["caption"]
        + ",line drawing illustration"
        + "\nObjects: "
        + str(layout)
    )
    print(prompt)
    image_path_raw, image_path_sketch = prompt_to_recombined_images(prompt, gen_num=1)
    description["layout"] = ast.literal_eval(layout)
    description["image_path_raw"] = image_path_raw
    description["image_path_sketch"] = image_path_sketch

    return {"image_path_sketch": image_path_sketch}, 200


# Keyword list --> Descriptions & Layouts & Sketches
@app.route("/mergeKeywords", methods=["POST"])
def merge_elements():
    data = request.get_json()
    keywords = data.get("keywords")
    keywordString = keywordlist_to_string(keywords)
    generatedDescriptions = keywords_to_descriptions(keywordString)

    for description in generatedDescriptions:
        objects = list(description["objects"].keys())
        print(objects)
        layout = caption_to_layout(description["caption"], objects)
        print(layout)
        prompt = (
            "Caption: "
            + description["caption"]
            + ",line drawing illustration"
            + "\nObjects: "
            + str(layout)
        )
        print(prompt)
        image_path_raw, image_path_sketch = prompt_to_recombined_images(prompt)
        description["layout"] = ast.literal_eval(layout)
        description["image_path_raw"] = image_path_raw
        description["image_path_sketch"] = image_path_sketch

    return {"descriptions": generatedDescriptions}, 200


@app.route("/getMoreSketches", methods=["POST"])
def generate_more_sketches():
    data = request.get_json()
    description = data.get("description")
    old_layout = data.get("layout")
    target_bbox_num = len(description["objects"])

    # Generate layouts
    new_layouts = []
    if old_layout is None or len(old_layout) < len(description["objects"]):
        original_layout = ast.literal_eval(
            caption_to_layout(description["caption"], description["objects"])
        )
        adjusted_layout = []
        for _, bbox in original_layout:
            adjusted_layout.append(
                [bbox[0] * 512, bbox[1] * 512, bbox[2] * 512, bbox[3] * 512]
            )
        layouts = generate_layouts(
            adjusted_layout, recommends_num=5, target_bbox_num=target_bbox_num
        )
    else:
        layouts = generate_layouts(
            old_layout, recommends_num=5, target_bbox_num=target_bbox_num
        )

    for layout in layouts:
        adjusted_layout = []
        for bbox in layout:
            adjusted_layout.append(
                [bbox[0] / 512, bbox[1] / 512, bbox[2] / 512, bbox[3] / 512]
            )
        new_layout = caption_layout_matcher(
            description["caption"], description["objects"], str(adjusted_layout)
        )
        new_layouts.append(new_layout)

    print("new_layouts")
    print(new_layouts)

    image_path_raw_list = []
    image_path_sketch_list = []

    # Generate skteches
    for new_layout in new_layouts:
        prompt = (
            "Caption: "
            + description["caption"]
            + ",line drawing illustration"
            + "\nObjects: "
            + str(new_layout)
        )
        image_path_raw, image_path_sketch = prompt_to_recombined_images(
            prompt, gen_num=1
        )
        image_path_raw_list.append(image_path_raw[0])
        image_path_sketch_list.append(image_path_sketch[0])

    return {
        "image_path_raw": image_path_raw_list,
        "image_path_sketch": image_path_sketch_list,
    }, 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7887, debug=False)
