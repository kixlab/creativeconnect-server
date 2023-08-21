from flask import Flask, request, send_file
from flask_cors import CORS
from datetime import datetime
import os, base64, random, time, json
import numpy as np
import torch, cv2, openai
from skimage.measure import label, regionprops
from transformers import Blip2Processor, Blip2ForConditionalGeneration

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

from llm_grounded_diffusion.run import recombination
from style_module.style_transfer import line_drawing_predict
from layout_module.layout_generation import sample_bboxes_gen
from layout_module.layout_metrics import cal_layout_sim

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
openai.api_key = "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"
blipprocessor = None
blipmodel = None

app = Flask(__name__)
CORS(app, resources={r'/*': {'origins': '*'}})

@app.route("/setup", methods=['GET'])
def setup():
    global blipprocessor, blipmodel
    
    # Setup BLIP model
    blipprocessor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    blipmodel = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        torch_dtype=torch.float16
    ).to(DEVICE)
    
    return 'setup done'

@app.route("/", methods=['GET'])
def test():
    return 'hello, world!'

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
                "content": "You will be provided with multiple sentences to describe an illustration. Your task is to extract a list of Subject matter, Action & pose, and Theme & mood.\n\nSubject matters are one-word, describing the specific physical objects, characters, or landscape that the user wants to include in their illustration. Example subject matters include pencil, children, or wave. \nFor subject matters, no adjectives should be included. They should always be a noun.\n\nActions & poses are word-level or phrase-level actions that the character or the object in the illustration performs. Example actions & poses include riding a bus, standing still, or traveling.\n\nThemes & moods are words not directly present in the illustration, but those that can potentially convey the overall theme or mood of the illustration. Example themes & moods include imaginative, eco-friendly, or sad. \nThey should be adverbs, preferably one-word."
                },
                {
                "role": "user",
                "content": "a card with chinese writing with colorful objects on it\na red and orange background with a blank paper, chinese, pencils, stationery and more\nan image of a classroom scene with various supplies"
                },
                {
                "role": "assistant",
                "content": "Subject matter: card, Chinese writing, colorful objects, red and orange background, blank paper, Chinese, pencils, stationery, classroom, supplies.\n\nAction & pose: \n\nTheme & mood: education, learning, multiculturalism."
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
        matters = res.split('\n\n')[0].split(': ')[1].strip('.').split(', ')
        actions = res.split('\n\n')[1].split(': ')[1].strip('.').split(', ')
        themes = res.split('\n\n')[2].split(': ')[1].strip('.').split(', ')
        
        return matters, actions, themes
    
    print("Getting GPT response")
    matters, _, _ = get_gpt_response(whole_image_description)
    _, actions, themes = get_gpt_response(segmented_descriptions)
    
    keywords = []
    
    for matter in matters:
        keywords.append({
            "type": "Subject matter", 
            "keyword": matter, 
            "position": None
        })
    
    for action in actions:
        keywords.append({
            "type": "Action & pose", 
            "keyword": action, 
            "position": None
        })
    for theme in themes:
        keywords.append({
            "type": "Theme & mood", 
            "keyword": theme, 
            "position": None
        })

    return {'filename': filename, 'keywords': keywords}, 200

@app.route('/getDescriptions', methods=['POST'])
def generate_descriptions_from_elements():
    data = request.get_json()
    elements = data.get('elements')['elements']
    
    matters = []
    actions = []
    themes = []
    
    for e in elements:
        if e["type"] == "Subject matter":
            matters.append(e)
        elif e["type"] == "Action & pose":
            actions.append(e)
        elif e["type"] == "Theme & mood":
            themes.append(e)
    
    elementlist = "Subject matter: " + ", ".join([o["keyword"].split("-")[0] for o in matters]) + "\nAction & pose: " + ", ".join([a["keyword"] for a in actions]) + "\nTheme & mood: " + ", ".join([c["keyword"] for c in themes])

    def get_gpt_response(elementlist):
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {
                "role": "system",
                "content": "The user wants to draw an illustration, with the assistance of you. You will be provided with multiple keywords users wanted to include in their illustrations. There are three types of keywords: Subject matter, Action & pose, and Theme & mood.\n\nSubject matters are one-word, describing the specific physical objects, characters, or landscape that the user wants to include in their illustration. Example subject matters include pencil, children, or wave. \nFor subject matters, no adjectives should be included. They should always be a noun.\n\nActions & poses are word-level or phrase-level actions that the character or the object in the illustration performs. Example actions & poses include riding a bus, standing still, or traveling.\n\nThemes & moods are words not directly present in the illustration, but those that can potentially convey the overall theme or mood of the illustration. Example themes & moods include imaginative, eco-friendly, or sad. \nThey should be adverbs, preferably one-word.\n\nYour task is to generate three descriptions of the illustration that the user can draw based on the given keywords. The three descriptions should be significantly different with each other. Each description should include three things: \"Scene\", \"Background\", and \"Objects\".\n\n\"Scene\" is a simple description of the overall image of the description. This should include all objects in the \"Objects\" list. Do not make it too long.\n\n\"Objects\" is a list of the objects depicted in the illustrations, and a short description of them. The objects should be one of the given \"Subject matters\", or something related to them. You don't have to include all given subject matters. The given action and concept should also be considered for generating the object's detail.\n\nThe background is a short concise sentence describing the background of the illustration. It should be well aligned with the objects depicted and the given concepts."
                },
                {
                "role": "user",
                "content": "Subject matter: ball, cat, dog\nAction & pose: jumping\nTheme & mood: playful, peaceful"
                },
                {
                "role": "assistant",
                "content": "Scene: A joyous spectacle unfolds in a vibrant backyard as a cat and a dog are engaged in delightful play, with a ball being their favorite toy.\nObjects: [(ball, a brightly colored ball), (dog, a dog jumping with its tongue sticking out), (cat, an agile cat in mid-jump)]\nBackground: A suburban backyard with lush green grass and bright flowers.\n\nScene: A beach where a dog is united in its pursuit of a bounce ball.\nObjects: [(ball, a sandy beach ball), (dog, a spotted dog leaping for the ball)]\nBackground: A peaceful beach vista, with playful seagulls and gentle waves crashing against the shore.\n\nScene: In a residential living room, the cat and dog both defying gravity in their jumps to catch the toy.\nObjects: [(toy, a small rubber toy with vibrant stripes), (dog, a bouncy terrier flying mid-air), (cat, a Siamese cat also in mid-jump)]\nBackground: A warm and cozy living room with sunlight pouring in through the windows."
                },
                {
                "role": "user",
                "content": "Subject matter: woman, drink, hat, chair\nAction & pose: sipping drink, relaxing, sitting\nTheme & mood: sunny, relaxed"
                },
                {
                "role": "assistant",
                "content": "Scene: A relaxed woman wearing a wide-brimmed hat is sitting in a comfortable chair, sipping her cool drink.\nObjects: [(woman, a casual-dressed woman with a relaxed smile on her face), (drink, a sweating glass filled with a luscious summer cocktail), (hat, the woman's wide-brimmed straw hat), (chair, a comfy chair with soft white cushions)]\nBackground: On a sunny outdoor patio with lush, green foliage around. \n\nScene: A stylish woman is peacefully enjoying her drink at an elegant street café.\nObjects: [(woman, an elegantly dressed woman with a serene facial expression), (drink, a classy cocktail in a stemmed glass), (chair, a classic bistro chair with a hat hanging on its backrest)]\nBackground:  An outdoor café in a city street against the backdrop of colorful buildings. \n\nScene: The woman in a panama hat relaxes on a mat with her drink, feeling the warmth of the sun. \nObjects: [(woman, a woman in a breezy summer dress), (drink, a jug of iced lemonade), (hat, a jaunty panama hat rested on her knee)]\nBackground: A porch area with a garden in the background and sunlight filtering through the leaves."
                },
                {
                "role": "user",
                "content": elementlist
                }
            ],
            temperature=1,
            max_tokens=1024,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        print(response.choices[0].message.content)
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
                elif line.startswith('Scene'):
                    scene = line.split(': ')[1]
            try:
                res.append({
                    "objects": objects,
                    "background": background,
                    "scene": scene
                })
            except:
                pass
        
        return res
    
    print("Getting GPT response")
    res = get_gpt_response(elementlist)

    return {"descriptions": res}, 200

@app.route('/getLayout', methods=['POST'])
def extract_layout_from_image():
    # Setup SAM model
    print("Setting up SAM model to ")
    print(DEVICE)
    MODEL_TYPE = "vit_l" # Other option: "vit_h"
    CHECKPOINT_PATH = "./checkpoints/sam_vit_l_0b3195.pth" # Other option: "vit_h"
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    sam.to(DEVICE)
    mask_generator = SamAutomaticMaskGenerator(sam)
    
    data = request.get_json()
    filename = data.get('filename')
    if not filename:
        return {'message': 'No file received'}, 400
        
    # Load image
    print("Loading image")
    IMAGE_PATH = "uploaded/"+filename
    image_bgr = cv2.imread(IMAGE_PATH)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    width, height = image_rgb.shape[1], image_rgb.shape[0]
    
    masks = mask_generator.generate(image_rgb)
    
    # return only bbox
    bboxes = []
    sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
    # If mask is too big, not included in bboxes
    i = 0
    while len(bboxes) < 10 and i < len(sorted_masks):
        mask = sorted_masks[i]
        if mask["area"] < 0.7*width*height:
            x, y, w, h = mask["bbox"]
            bboxes.append((x*200/width, y*200/height, w*200/width, h*200/height))
        i += 1
    
    sam = None
    mask_generator = None
    
    return {"bboxes": bboxes}, 200

@app.route('/listLayouts', methods=['POST'])
def recommend_layouts():
    '''
        recommend_layouts: Recommend most similar layout list to given

        old_layout = [(60, 143, 100, 126),(265, 193, 190, 210)] each tuple is single xywh formatted bounding box
        return => List contained 10 xywh formatted layout
    '''
    data = request.get_json()
    old_layout = data.get('layout')
    print(old_layout)
    recomms = []
    # all sample layouts are xywh format.
    for sample_layout in sample_bboxes_gen():
        sim = cal_layout_sim(old_layout, sample_layout)
        recomms.append([sim, sample_layout])
    
    highrecomms = sorted(lambda x:x[0], recomms, reverse=True)[:10]

    return list(map(lambda x: x[1], highrecomms))

@app.route('/getImage', methods=['POST'])
def generate_recombined_images():
    '''
        generate_recombined_images: following the given prompt, generate image and transform it to line drawing

        prompt = """Caption: Gray cat and a soccer ball on the grass, line drawing.
        Objects: [('a gray cat', [67, 243, 120, 126]), ('a soccer ball', [265, 193, 190, 210])]
        Background prompt: A grassy area."""

        output => PIL
    '''
    
    # test_prompt = """Caption: Gray cat and a soccer ball on the grass, line drawing.\nObjects: [('a gray cat', [67, 243, 120, 126]), ('a soccer ball', [265, 193, 190, 210])]\nBackground prompt: A grassy area."""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{timestamp}.png"
    image_path_sketch = "generated/" + f"{timestamp}_sketch.png"
    image_path_raw = "generated/" + f"{timestamp}_raw.png"

    data = request.get_json()
    prompt = data.get('prompt')
    print(prompt)
    
    # output type is pil
    output_raw = recombination(prompt=prompt)
    output_sketch = line_drawing_predict(output_raw, ver='Simple Lines')
    output_raw.save(image_path_raw)
    output_sketch.save(image_path_sketch)

    return send_file(image_path_sketch, mimetype='image/png')

@app.route('/expandElements', methods=['POST'])
def generate_elements_from_elements():
    data = request.get_json()
    elements = data.get('elements')
    
    matters = []
    actions = []
    themes = []
    
    for e in elements:
        if e["type"] == "Subject matter":
            matters.append(e)
        elif e["type"] == "Action & pose":
            actions.append(e)
        elif e["type"] == "Theme & mood":
            themes.append(e)
    
    elementlist = "Subject matter: " + ", ".join([o["keyword"].split("-")[0] for o in matters]) + "\nAction & pose: " + ", ".join([a["keyword"] for a in actions]) + "\Theme & mood: " + ", ".join([c["keyword"] for c in themes])

    def get_gpt_response(elementlist):
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "We are trying to support novice designers' ideation process by semantically combining different parts of references (illustrations). You will be provided with the topic of the ideation, and multiple keywords users like in the illustrations they found as references. There are three types of keywords: Subject matter, Action & pose, and Theme & mood.\n\nSubject matters are one-word, describing the specific physical objects, characters, or landscape that the user wants to include in their illustration. Example subject matters include pencil, children, or wave. \nFor subject matters, no adjectives should be included. They should always be a noun.\n\nActions & poses are word-level or phrase-level actions that the character or the object in the illustration performs. Example actions & poses include riding a bus, standing still, or traveling.\n\nThemes & moods are words not directly present in the illustration, but those that can potentially convey the overall theme or mood of the illustration. Example themes & moods include imaginative, eco-friendly, or sad. \nThey should be adverbs, preferably one-word.\n\nYour task is to expand on the keywords being given, by combining multiple keywords or looking for synonyms that can inspire new creations or ideas.\n\nFor example, the subject matter \"pencil\" can be combined with the action & pose \"traveling\" to inspire a new action & pose \"writing a diary\". \nThe keywords should be formatted in the following manner: \nwriting a diary (action & pose)  = pencil (subject matter) + traveling (action & pose)\n\nYou can combine as many keywords at once.\nAnother example:\nhair salon (subject matter) = hair dryer (subject matter) + comb (subject matter) + scissors (subject matter)\n\nFor combinations that result in theme & mood, make them as abstract as possible. An example:\nadventurous (theme & mood) = riding on ship (action & pose) + tent (subject matter)\n\nCome up with five new keywords for each keyword type with creative combinations. Only use the original keywords provided in creating new keywords.\n\nImportant: Include at least one subject matter for each combination. Subject matter and theme & mood should be a SINGLE WORD. Combinations among subject matters are highly recommended. "
                },
                {
                    "role": "user",
                    "content": "Subject matter: camping, tent, tree, animals, Eiffel tower, family\nAction & pose: riding on a bus, riding on a ship\nTheme & mood: playful, imaginative"
                },
                {
                    "role": "assistant",
                    "content": "Subject matter:\ncamping site = camping (Subject matter) + tent (Subject matter)\nforest = tree (Subject matter) + animals (Subject matter)\nParis scene = Eiffel tower (Subject matter) + family (Subject matter)\nsafari = animals (Subject matter) + riding on a bus (Action & pose)\n\nAction & pose:\nsetting up camp = camping (Subject matter) + tent (Subject matter)\nwildlife exploring = tree (Subject matter) + animals (Subject matter)\ntouring Paris = Eiffel tower (Subject matter) + riding on a bus (Action & pose)\ngoing on a picnic = family (Subject matter) + camping (Subject matter)\n\nTheme & mood:\nadventurous = riding on a bus (Action & pose) + animals (Subject matter)\nserene = tree (Subject matter) + playful (Theme & mood)\njoyful = family (Subject matter) + playful (Theme & mood)\nromantic = Eiffel tower (Subject matter) + imaginative (Theme & mood)"
                },
                {
                    "role": "user",
                    "content": elementlist
                }
            ],
            temperature=1,
            max_tokens=1066,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        res = []
        for r in response.choices[0].message.content.split('\n\n'):
            lines = r.split('\n')
            elementType = lines[0].split(':')[0]
            for line in lines[1:]:
                keyword = line.split(' = ')[0].strip()
                sources = line.split(' = ')[1].split(' + ')
                res.append({
                    "type": elementType,
                    "keyword": keyword,
                    "sources": [{"keyword": s.split(' (')[0].strip(),
                                    "type": s.split(' (')[1].split(')')[0]} for s in sources]})
        return res
    
    print("Getting GPT response")
    res = get_gpt_response(elementlist)

    return {"descriptions": res}, 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=7887, debug=True)
