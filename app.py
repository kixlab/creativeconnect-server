from flask import Flask, request, send_file
from flask_cors import CORS
from datetime import datetime
import os, base64, random, time, json
import numpy as np
import torch, cv2, openai
from skimage.measure import label, regionprops
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data.detection_utils import read_image

from llm_grounded_diffusion.run import recombination
from style_module.style_transfer import line_drawing_predict
from layout_module.layout_generation import sample_bboxes_gen
from layout_module.layout_metrics import cal_layout_sim

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
openai.api_key = "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"
blipprocessor = None
blipmodel = None
ovsegPredictor = None

# class MyFlaskApp(Flask):
#   def run(self, host=None, port=None, debug=None, load_dotenv=True, **options):
#     if not self.debug or os.getenv('WERKZEUG_RUN_MAIN') == 'true':
#       with self.app_context():
#         setup()
#     super(MyFlaskApp, self).run(host=host, port=port, debug=debug, load_dotenv=load_dotenv, **options)

app = Flask(__name__)
CORS(app, resources={r'/*': {'origins': '*'}})

@app.route("/setupLarge", methods=['GET'])
def setup_2():
    global blipprocessor, blipmodel, ovsegPredictor
    
    # Setup BLIP model
    blipprocessor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    blipmodel = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
    ).to(DEVICE)
    
    # Setup OVSeg model
    CONFIG_FILE = "../ovseg/configs/ovseg_swinB_vitL_demo.yaml"
    OPTS = ['MODEL.WEIGHTS', '../ovseg/checkpoints/ovseg_swinbase_vitL14_ft_mpt.pth']
    def setup_cfg():
        # load config from file and command-line arguments
        cfg = get_cfg()
        # for poly lr schedule
        add_deeplab_config(cfg)
        cfg.merge_from_file(CONFIG_FILE)
        cfg.merge_from_list(OPTS)
        cfg.freeze()
        return cfg
    cfg = setup_cfg()
    return 'setup done'

@app.route("/setup", methods=['GET'])
def setup():
    global blipprocessor, blipmodel
    
    # Setup BLIP model
    blipprocessor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    blipmodel = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
    ).to(DEVICE)
    return 'setup done'


@app.route("/", methods=['GET'])
def test():
    return 'hello, world!'

@app.route('/sendImage', methods=['POST'])
# @cross_origin()
def register_new_image():
    data = request.get_json()
    image = data.get('image')
    if not image:
        return {'message': 'No file received'}, 400

    file_content = image.split(';base64,')[1]
    file_extension = image.split(';')[0].split('/')[1]
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{timestamp}.{file_extension}"

    with open("uploaded/"+filename, 'wb') as f:
        f.write(base64.b64decode(file_content))
        
    # Load image
    print("Loading image")
    IMAGE_PATH = "uploaded/"+filename
    image_bgr = cv2.imread(IMAGE_PATH)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Get whole image description
    print("Getting whole image description")
    inputs = blipprocessor(images=image_rgb, return_tensors="pt").to(DEVICE, torch.float16)
    generated_ids = blipmodel.generate(**inputs)
    whole_image_description = blipprocessor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    segmented_descriptions = whole_image_description + "\n"
    
    # Get segmented image descriptions
    print("Getting segmented image descriptions")
    image_segments = []
    for i in range(3):
        for j in range(3):
            w = int(image_rgb.shape[0] / 2)
            h = int(image_rgb.shape[1] / 2)
            x = int(image_rgb.shape[0] / 4 * i)
            y = int(image_rgb.shape[1] / 4 * j)
            image_segments.append(image_rgb[x:x+w, y:y+h])
    for seg in image_segments:
        inputs = blipprocessor(images=seg, return_tensors="pt").to(DEVICE, torch.float16)
        generated_ids = blipmodel.generate(**inputs)
        generated_text = blipprocessor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        segmented_descriptions += generated_text + "\n"
    
    def get_gpt_response(description):
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {
                "role": "system",
                "content": "You will be provided with multiple sentences to describe an illustration. Your task is to extract a list of objects, actions, and concepts.\n\nObjects are words about the specific physical objects found in the sentences. DO NOT INCLUDE BACKGROUND OR HIGH-LEVEL WORDS SUCH AS \"ITEM\" OR \"OBJECT\".\n\nConcepts are words not found in the given text, but can potentially convey the overall mood or topic of the illustration."
                },
                {
                "role": "user",
                "content": "a card with chinese writing with colorful objects on it\na red and orange background with a blank paper, chinese, pencils, stationery and more\nan image of a classroom scene with various supplies"
                },
                {
                "role": "assistant",
                "content": "Objects: card, Chinese writing, blank paper, pencils, stationery, classroom\n\nActions: \n\nConcepts: education, learning, multiculturalism."
                },
                {
                "role": "user",
                "content": description
                },
            ],
            temperature=0.5,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        res = response.choices[0].message.content
        objects = res.split('\n\n')[0].split(': ')[1].strip('.').split(', ')
        actions = res.split('\n\n')[1].split(': ')[1].strip('.').split(', ')
        concepts = res.split('\n\n')[2].split(': ')[1].strip('.').split(', ')
        
        return objects, actions, concepts
    
    print("Getting GPT response")
    objects, _, _ = get_gpt_response(whole_image_description)
    _, actions, concepts = get_gpt_response(segmented_descriptions)
    
    keywords = []
    
    # OVSeg
    print("Running OVSeg")
    start_time = time.time()
    predictions, _ = ovsegPredictor.run_on_image(image_bgr, objects)
    print(
        "{}: {} in {:.2f}s".format(
            IMAGE_PATH,
            "detected {} instances".format(len(predictions["instances"]))
            if "instances" in predictions
            else "finished",
            time.time() - start_time,
        )
    )
    segmentation_masks = predictions["sem_seg"].cpu()
    for i in range(len(objects)):
        object_mask = segmentation_masks[i, :, :]
        binary_mask = object_mask > 0.8
        labeled_mask = label(binary_mask)
        props = regionprops(labeled_mask)
        for j in range(len(props)):
            prop = props[j]
            area = prop.area
            if area > 0.001 * image_rgb.shape[0]*image_rgb.shape[1]:
                ymin, xmin, ymax, xmax = prop.bbox
                mask = np.zeros((image_rgb.shape[0], image_rgb.shape[1]))
                mask[ymin:ymax, xmin:xmax] = prop.image
                keywords.append({
                    "type": "object", 
                    "keyword": objects[i]+"-"+str(j+1),
                    "position": {
                        "x": ((xmin+xmax) / 2)  / image_rgb.shape[1],
                        "y": ((ymin+ymax) / 2)  / image_rgb.shape[0]
                    },
                    "mask": mask.tolist(),
                    "bbox": prop.bbox
                })
    
    for act in actions:
        keywords.append({
            "type": "action", 
            "keyword": act, 
            "position": None
        })
    for con in concepts:
        keywords.append({
            "type": "concept", 
            "keyword": con, 
            "position": None
        })

    return {'filename': filename, 'keywords': keywords}, 200

@app.route('/getElement', methods=['POST'])
def extract_element_from_image():
    data = request.get_json()
    image = data.get('image')
    if not image:
        return {'message': 'No file received'}, 400

    file_content = image.split(';base64,')[1]
    file_extension = image.split(';')[0].split('/')[1]
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{timestamp}.{file_extension}"

    with open("uploaded/"+filename, 'wb') as f:
        f.write(base64.b64decode(file_content))
        
    # Load image
    print("Loading image")
    IMAGE_PATH = "uploaded/"+filename
    image_bgr = cv2.imread(IMAGE_PATH)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Get whole image description
    print("Getting whole image description")
    inputs = blipprocessor(images=image_rgb, return_tensors="pt").to(DEVICE, torch.float16)
    generated_ids = blipmodel.generate(**inputs)
    whole_image_description = blipprocessor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    segmented_descriptions = whole_image_description + "\n"
    
    # Get segmented image descriptions
    print("Getting segmented image descriptions")
    image_segments = []
    for i in range(3):
        for j in range(3):
            w = int(image_rgb.shape[0] / 2)
            h = int(image_rgb.shape[1] / 2)
            x = int(image_rgb.shape[0] / 4 * i)
            y = int(image_rgb.shape[1] / 4 * j)
            image_segments.append(image_rgb[x:x+w, y:y+h])
    for seg in image_segments:
        inputs = blipprocessor(images=seg, return_tensors="pt").to(DEVICE, torch.float16)
        generated_ids = blipmodel.generate(**inputs)
        generated_text = blipprocessor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        segmented_descriptions += generated_text + "\n"
    
    def get_gpt_response(description):
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {
                "role": "system",
                "content": "You will be provided with multiple sentences to describe an illustration. Your task is to extract a list of objects, actions, and concepts.\n\nObjects are words about the specific physical objects found in the sentences. DO NOT INCLUDE BACKGROUND OR HIGH-LEVEL WORDS SUCH AS \"ITEM\" OR \"OBJECT\".\n\nConcepts are words not found in the given text, but can potentially convey the overall mood or topic of the illustration."
                },
                {
                "role": "user",
                "content": "a card with chinese writing with colorful objects on it\na red and orange background with a blank paper, chinese, pencils, stationery and more\nan image of a classroom scene with various supplies"
                },
                {
                "role": "assistant",
                "content": "Objects: card, Chinese writing, blank paper, pencils, stationery, classroom\n\nActions: \n\nConcepts: education, learning, multiculturalism."
                },
                {
                "role": "user",
                "content": description
                },
            ],
            temperature=0.5,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        res = response.choices[0].message.content
        objects = res.split('\n\n')[0].split(': ')[1].strip('.').split(', ')
        actions = res.split('\n\n')[1].split(': ')[1].strip('.').split(', ')
        concepts = res.split('\n\n')[2].split(': ')[1].strip('.').split(', ')
        
        return objects, actions, concepts
    
    print("Getting GPT response")
    objects, _, _ = get_gpt_response(whole_image_description)
    _, actions, concepts = get_gpt_response(segmented_descriptions)
    
    keywords = []
    
    for obj in objects:
        keywords.append({
            "type": "object", 
            "keyword": obj, 
            "position": None
        })
    
    for act in actions:
        keywords.append({
            "type": "action", 
            "keyword": act, 
            "position": None
        })
    for con in concepts:
        keywords.append({
            "type": "concept", 
            "keyword": con, 
            "position": None
        })

    return {'filename': filename, 'keywords': keywords}, 200

@app.route('/getDescriptions', methods=['POST'])
def generate_descriptions_from_elements():
    data = request.get_json()
    elements = data.get('elements')['elements']
    
    objects = []
    actions = []
    concepts = []
    
    for e in elements:
        if e["type"] == "object":
            objects.append(e)
        elif e["type"] == "action":
            actions.append(e)
        elif e["type"] == "concept":
            concepts.append(e)
    
    elementlist = "Object: " + ", ".join([o["keyword"].split("-")[0] for o in objects]) + "\nAction: " + ", ".join([a["keyword"] for a in actions]) + "\nConcept: " + ", ".join([c["keyword"] for c in concepts])

    def get_gpt_response(elementlist):
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {
                "role": "system",
                "content": "The user wants to draw an illustration, with the assistance of you. You will be provided with multiple keywords users wanted to include in their illustrations. There are three types of keywords: object, action, and concepts.\n\nObjects are words about the specific physical objects that the user wants to include in their illustration.\n\nAction is about the action that the character or animal in the illustration performs.\n\nConcepts are words not found in the illustration directly, but can potentially convey the overall mood or topic of the illustration.\n\nYour task is to generate three descriptions of the illustration that the user can draw based on the given keywords. Each description should include two things: \"Background\" and \"Objects\".\n\n\"Objects\" is a list of the objects depicted in the illustrations, and the short description of them. The objects should be one of the given objects, or closely related to them. The given action and concept should be also considered for generating the detail of the object.\n\nThe background is a one-line concise sentence describing the background of the illustration. It should be well aligned with the objects depicted and the given concepts."
                },
                {
                "role": "user",
                "content": "Object: ball, cat, dog\nAction: jumping\nConcept: playful, peaceful"
                },
                {
                "role": "assistant",
                "content": "Objects: [(cat, a playful white cat), (dog, a joyful dog chasing a bouncing ball)]\nBackground: A sunny park with lush green grass and colorful flowers.\n\nObjects: [(cat, a cat leaping through the air), (dog, a dog jumping after a bouncing ball)]\nBackground: A cozy living room with a warm fireplace and comfortable furniture.\n\nObjects: [(dog, an exuberant dog), (cat, a carefree cat), (ball, rolling on the beach)]\nBackground: A serene beach with soft sand, sparkling blue waves, and seashells scattered along the shore."
                },
                {
                "role": "user",
                "content": "Object: sofa, air balloon, projector\nAction: relaxing\nConcept: skeptical"
                },
                {
                "role": "assistant",
                "content": "Objects: [(sofa, a sofa with a person lounging on it), (air balloon, mini air balloon hovering above the floor), (projector, a projector displaying a sci-fi movie)]\nBackground: A loft-style living room filled with modern, minimalistic furniture and large windows.\n\nObjects: [(sofa, an inviting, plush sofa), (projector, an old projector screening a noir film), (air balloon, a model air balloon)]\nBackground: A dimly lit, cozy media room with posters of classic movies on the walls.\n\nObjects: [(sofa, a vintage leather sofa with a person sitting in contemplation), (projector, a projector casting light and shadow across the room), (air balloon, an air balloon-shaped lamp hanging from the ceiling)]\nBackground: A distinctive, retro-themed lounge filled with a collection of antique items."
                },
                {
                "role": "user",
                "content": "Object: bar graph, cell phone, grad student\nAction: reading, standing\nConcept: gloomy, blue, research"
                },
                {
                "role": "assistant",
                "content": "Objects: [(cell phone, a grad student standing holding a phone), (bar graph, a bar graph on the screen showing statistics)]\nBackground: A dimly lit room barely illuminating a cluttered study desk filled with research papers and stacks of books.\n\nObjects: [(grad student, a grad student in a blue sweater, standing and reading statistical data on a cell phone), (bar graph, a complex bar graph depicts recent research results)]\nBackground: A library corner with towering bookshelves and a gloomy atmosphere, scattered with open books and coffee cups.\n\nObjects: [(bar graph, a large digital bar graph on the phone screen), (grad student, a grad student looking visibly stressed)]\nBackground: A lab filled with computers emitting a soft glow, research papers scattered everywhere."
                },
                {
                "role": "user",
                "content": elementlist
                }
            ],
            temperature=1,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        res = []
        for r in response.choices[0].message.content.split('\n\n'):
            print(r)
            objects = []
            background = ""
            for line in r.split('\n'):
                if line.startswith('Objects'):
                    tmp = line.split(': ')[1].strip('[]').split('), ')
                    for (obj, i) in zip(tmp, range(len(tmp))):
                        obj = obj.strip('()').split(', ')
                        objects.append({
                            "object": obj[0],
                            "detail": obj[1],
                            "id": i+1
                        })
                elif line.startswith('Background'):
                    background = line.split(': ')[1]
            res.append({
                "objects": objects,
                "background": background
            })
        
        return res
    
    print("Getting GPT response")
    res = get_gpt_response(elementlist)

    return {"descriptions": res}, 200

@app.route('/listLayouts', methods=['POST'])
def recommend_layouts():
    data = request.get_json()
    old_layout = data.get('layout')
    recomms = []
    # all sample layouts are xywh format.
    for sample_layout in sample_bboxes_gen():
        sim = cal_layout_sim(old_layout, sample_layout)
        recomms.append([sim, sample_layout])
    
    highrecomms = sorted(lambda x:x[0], recomms)[:10]

    return list(map(lambda x: x[1], highrecomms))


@app.route('/getImages', methods=['POST'])
def generate_recombined_images():
    image_path = "generated/" + "test.png"

    data = request.get_json()
    data = data.get('data')
    print(data)
    
    # output type is pil
    output = recombination(prompt=data)
    output = line_drawing_predict(output, ver='Simple Lines')
    output.save(image_path)

    return send_file(image_path, mimetype='image/png')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=52000, debug=True)
