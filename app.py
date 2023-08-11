from flask import Flask, request
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
from ovseg.open_vocab_seg import add_ovseg_config
from ovseg.open_vocab_seg.utils import VisualizationDemo

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

@app.route("/setup", methods=['GET'])
def setup():
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
        add_ovseg_config(cfg)
        cfg.merge_from_file(CONFIG_FILE)
        cfg.merge_from_list(OPTS)
        cfg.freeze()
        return cfg
    cfg = setup_cfg()
    ovsegPredictor = VisualizationDemo(cfg)
    return 'setup done'

@app.route("/", methods=['GET'])
def test():
    return 'hello, world!'

@app.route('/imageElement/sendImage', methods=['POST'])
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

@app.route('/imageElement/sendElement', methods=['POST'])
def element_merge():
    # if request.method == 'OPTIONS':
    #     return '', 200
    
    data = request.get_json()
    elements = data.get('elements')['elements']
    
    # f = open("dummydata.json", "r")
    # elements = json.loads(f.read())['elements']
    # f.close()
    
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
                "content": "The user wants to draw an illustration, with the assistance of you. You will be provided with multiple keywords users wanted to include in their illustrations. There are three types of keywords: object, action, and concepts.\n\nObjects are words about the specific physical objects that the user wants to include in their illustration.\n\nAction is about the action that the character or animal in the illustration performs.\n\nConcepts are words not found in the illustration directly, but can potentially convey the overall mood or topic of the illustration.\n\nYour task is to generate three descriptions of the illustration that the user can draw based on the given keywords. Each description should be in two concise lines, one for the background, and one for the scene."
                },
                {
                "role": "user",
                "content": "Object: ball, cat, dog\nAction: jumping\nConcept: playful, peaceful"
                },
                {
                "role": "assistant",
                "content": "Background: A sunny park with lush green grass and colorful flowers.\nScene: A playful cat and a joyful dog chasing a bouncing ball together.\n\nBackground: A cozy living room with a warm fireplace and comfortable furniture.\nScene: A peaceful cat gracefully leaping through the air, while a playful dog jumps energetically after a bouncing ball.\n\nBackground: A serene beach with soft sand, sparkling blue waves, and seashells scattered along the shore.\nScene: A carefree cat and an exuberant dog leaping in sync, as they joyfully jump after a ball rolling on the sandy beach."
                },
                {
                "role": "user",
                "content": "Object: sofa, air balloon, projector\nAction: relaxing\nConcept: skeptical"
                },
                {
                "role": "assistant",
                "content": "Background: A softly lit, intriguing living room, with old mystery books scattered across wooden shelves. \nScene: On the comfy sofa, a skeptical person relaxes, gazing at the ghostly silhouette of an air balloon projected onto the wall. \n\nBackground: A spacious, cozy attic with vintage decorations and star-themed wallpaper.\nScene: A person, donned in relaxed loungewear, sits skeptically on a plush sofa, observing a projected air balloon mysteriously floating on the ceiling.\n\nBackground: A circular room adorned with astronomical murals and whimsical twinkling lights for a starry effect. \nScene: A dubious figure lounges restfully on an antique sofa, eyes following the floating journey of a projected air balloon around the room."
                },
                {
                "role": "user",
                "content": "Object: bar graph, cell phone, grad student\nAction: researching\nConcept: sad"
                },
                {
                "role": "assistant",
                "content": "Background: A cluttered grad school dorm room filled with textbooks, sticky notes, and mug-full of half-drunk coffee. \nScene: A tired grad student sits slumped over, sadly researching on their cell phone, with a bar graph displayed despairingly on their cracked screen.\n\nBackground: A dimly lit, silent library corner with towering bookshelves all around.\nScene: A disheartened grad student is hunched over a bar graph, meticulously researching data on their cell phone with an air of melancholy.\n\nBackground: A quiet study room, where the fading sunset casts long, forlorn shadows. \nScene: A downcast grad student deeply engrossed in researching, sadly staring at a bar graph displayed on their cellphone screen."
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
            background, scene = r.split('\n')
            background = background.strip(' ').split(': ')[1]
            scene = scene.strip(' ').split(': ')[1]
            res.append({
                "background": background,
                "scene": scene
            })
        return res
    
    print("Getting GPT response")
    res = get_gpt_response(elementlist)

    return {"descriptions": res}, 200


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=52000, debug=True)
