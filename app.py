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
)
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from llm_grounded_diffusion.run import recombination
from style_module.style_transfer import line_drawing_predict
from layout_module.layout_metrics import cal_layout_sim, perturb_layout

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

blipprocessor = None
blipmodel = None

app = Flask(__name__, static_folder="./generated")
CORS(app, resources={r"/*": {"origins": "*"}})


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


def prompt_to_recombined_images(prompt):
    '''
    Input: prompt
        e.g. """Caption: Gray cat and a soccer ball on the grass, line drawing.
                Objects: [('a gray cat', [67, 243, 120, 126]), ('a soccer ball', [265, 193, 190, 210])]
                Background prompt: A grassy area."""
    Output:
        - image_path_raw: generated raw image path
        - image_path_sketch: generated sketch image path
    '''
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    image_path_sketch = "generated/" + f"{timestamp}_sketch.png"
    image_path_raw = "generated/" + f"{timestamp}_raw.png"

    output_raw = recombination(prompt=prompt)
    output_sketch = line_drawing_predict(output_raw, ver="Simple Lines")
    output_raw.save(image_path_raw)
    output_sketch.save(image_path_sketch)

    return image_path_raw, image_path_sketch


# Routes


@app.route("/setup", methods=["GET"])
def setup():
    global blipprocessor, blipmodel

    # Setup BLIP model
    blipprocessor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    blipmodel = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
    ).to(DEVICE)

    return "setup done"


@app.route("/", methods=["GET"])
def test():
    return "hello, world!"


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
    inputs = blipprocessor(images=image_rgb, return_tensors="pt").to(
        DEVICE, torch.float16
    )
    generated_ids = blipmodel.generate(**inputs)
    whole_image_description = blipprocessor.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0].strip()
    segmented_descriptions = whole_image_description + "\n"

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
        segmented_descriptions += generated_text + "\n"

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
        - bboxes: list of bounding boxes
    """
    # Setup SAM model
    MODEL_TYPE = "vit_l"
    CHECKPOINT_PATH = "./checkpoints/sam_vit_l_0b3195.pth"
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    sam.to(DEVICE)
    mask_generator = SamAutomaticMaskGenerator(sam)

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
            bboxes.append(
                (x * 200 / width, y * 200 / height, w * 200 / width, h * 200 / height)
            )
        i += 1

    sam = None
    mask_generator = None

    return {"bboxes": bboxes}, 200


# Layout --> Layout
@app.route("/getRecommendedLayouts", methods=["POST"])
def recommend_layouts():
    """
    Input:
        - layout: list of bounding boxes

    Output:
        - layouts: list of 10 recommended layouts
    """
    data = request.get_json()
    old_layout = data.get("layout")
    print(old_layout)
    recomms = []
    sample_layouts = [
        perturb_layout(old_layout, position_variation=300, size_variation=300)
        for _ in range(10)
    ]
    # all sample layouts are xywh format.
    for sample_layout in sample_layouts:
        sim = cal_layout_sim(old_layout, sample_layout)
        recomms.append([sim, sample_layout])

    highrecomms = sorted(recomms, key=lambda x: x[0], reverse=True)[:10]
    layouts = list(map(lambda x: x[1], highrecomms))

    return {"layouts": layouts}, 200


# Layout diffusion
@app.route("/generateImage", methods=["POST"])
def generate_recombined_images():
    '''
    Input:
        - prompt: prompt for layout diffusion
            e.g., """Caption: Gray cat and a soccer ball on the grass, line drawing.
                    Objects: [('a gray cat', [67, 243, 120, 126]), ('a soccer ball', [265, 193, 190, 210])]
                    Background prompt: A grassy area."""

    Output:
        - image_path_sketch: generated sketch image path
    '''
    data = request.get_json()
    prompt = data.get("prompt")

    image_path_raw, image_path_sketch = prompt_to_recombined_images(prompt)

    return {
        "image_path_raw": image_path_raw,
        "image_path_sketch": image_path_sketch,
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


# Keyword list --> Descriptions & Layouts & Sketches
@app.route("/mergeKeywords", methods=["POST"])
def merge_elements():
    data = request.get_json()
    keywords = data.get("keywords")
    keywordString = keywordlist_to_string(keywords)
    generatedDescriptions = keywords_to_descriptions(keywordString)

    for description in generatedDescriptions:
        objects = list(description["objects"].keys())
        layout = caption_to_layout(description["scene"], objects)
        prompt = (
            "Caption: "
            + description["scene"]
            + "\nObjects: "
            + str(layout)
            + "\nBackground prompt: "
            + description["background"]
        )
        image_path_raw, image_path_sketch = prompt_to_recombined_images(prompt)
        description["layout"] = ast.literal_eval(layout)
        description["image_path_raw"] = image_path_raw
        description["image_path_sketch"] = image_path_sketch

    return {"descriptions": generatedDescriptions}, 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7887, debug=True)
