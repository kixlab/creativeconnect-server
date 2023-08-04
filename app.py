from flask import Flask, request
from flask_cors import CORS
from datetime import datetime
import os, base64, random, time
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
        objects = res.split('\n\n')[0].split(': ')[1].split(', ')
        actions = res.split('\n\n')[1].split(': ')[1].split(', ')
        concepts = res.split('\n\n')[2].split(': ')[1].split(', ')
        
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
                keywords.append({
                    "type": "object", 
                    "keyword": objects[i]+"-"+str(j+1),
                    "position": {
                        "x": ((xmin+xmax) / 2)  / image_rgb.shape[1],
                        "y": ((ymin+ymax) / 2)  / image_rgb.shape[0]
                    },
                    "mask": prop.image.tolist(),
                    "bbox": prop.bbox
                })
    
    for act in actions:
        keywords.append({
            "type": "action", 
            "keyword": act, 
            "position": {
                "x": random.random(), 
                "y": random.random()
            }
        })
    for con in concepts:
        keywords.append({
            "type": "concept", 
            "keyword": con, 
            "position": {
                "x": random.random(), 
                "y": random.random()
            }
        })

    return {'filename': filename, 'keywords': keywords}, 200


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=52000, debug=True)
