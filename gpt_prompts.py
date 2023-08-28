import openai, ast

openai.api_key = "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"


def caption_to_keywords(description):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You will be provided with multiple sentences to describe an illustration. Your task is to extract a list of Subject matter, Action & pose, and Theme & mood.\n\nSubject matters are one-word, describing the specific physical objects, characters, or landscape that the user wants to include in their illustration. Example subject matters include pencil, children, or wave. \nFor subject matters, no adjectives should be included. They should always be a noun.\n\nActions & poses are word-level or phrase-level actions that the character or the object in the illustration performs. Example actions & poses include riding a bus, standing still, or traveling.\n\nThemes & moods are words not directly present in the illustration, but those that can potentially convey the overall theme or mood of the illustration. Example themes & moods include imaginative, eco-friendly, or sad. \nThey should be adverbs, preferably one or two words.\n\nIf you are provided a sentences including some style such as cartoon, illustration, image, photo, exclude it. For other example, 'an illustration of a woman sitting at a table' caption is extracted to 'woman', 'table', 'sitting at a table', 'cozy'. The 'illustration' is not contained. \n\nEliminate the changed forms of the same word, such as plurals. Only include roots. For example of 'trees' and 'tree', only include 'tree'. ",
            },
            {
                "role": "user",
                "content": "a card with chinese writing with colorful objects on it\na red and orange background with a blank paper, chinese, pencils, stationery and more\nan image of a classroom scene with various supplies",
            },
            {
                "role": "assistant",
                "content": "Subject matter: card, Chinese writing, colorful objects, red and orange background, blank paper, Chinese, pencils, stationery, classroom, supplies.\n\nAction & pose: \n\nTheme & mood: education, learning, multiculturalism.",
            },
            {
                "role": "user",
                "content": "a man and a woman sitting at a table eating food\na woman sitting in a chair in a room with potted plants\na man sitting at a table eating a hot dog\na illustration of a woman sitting at a table in a kitchen with potted plants\nan illustration of a woman sitting at a table\nan illustration of a refrigerator next to a chair and a table\nan illustration of two people sitting at a table with their feet on a chair\na cartoon of a person sitting in a chair in a shower",
            },
            {
                "role": "assistant",
                "content": "Subject matter: man, woman, table, food, chair, potted plants, hot dog, kitchen, refrigerator, feet\n\nAction & pose: sitting at a table, eating food.\n\nTheme & mood: cozy, heartwarming",
            },
            {
                "role": "user",
                "content": "a painting of a white barn in a field of flowers\na painting of a blue house with a tree next to it\na painting of a blue house with a roof and trees\na painting of a window of a house with flowers\na painting of a field of flowers in front of a house",
            },
            {
                "role": "assistant",
                "content": "Subject matter: painting, white barn, field, flowers, blue house, tree, roof, window.\n\nAction & pose: \n\nTheme & mood: rural, peaceful, nature.",
            },
            {
                "role": "user",
                "content": "the album cover of the beatles abbey road\na man and a woman standing in front of a car\na man and a woman walking down a street\na group of people walking across a crosswalk",
            },
            {
                "role": "assistant",
                "content": "Subject matter: Beatles, Abbey Road, man, woman, car, street, group, people, crosswalk.\n\nAction & pose: standing in front of a car, walking down a street, walking across a crosswalk.\n\nTheme & mood: urban, nostalgia.",
            },
            {"role": "user", "content": description},
        ],
        temperature=0.3,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    res = response.choices[0].message.content
    matters = res.split("\n\n")[0].split(": ")[1].strip(".").split(", ")
    actions = res.split("\n\n")[1].split(": ")[1].strip(".").split(", ")
    themes = res.split("\n\n")[2].split(": ")[1].strip(".").split(", ")

    return matters, actions, themes


def keywords_expansion(elementlist):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": 'We are trying to support novice designers\' ideation process by semantically combining different parts of references (illustrations). You will be provided with the topic of the ideation, and multiple keywords users like in the illustrations they found as references. There are three types of keywords: Subject matter, Action & Pose, and Theme & Mood.\n\nSubject matters are one-word, describing the specific physical objects, characters, or landscape that the user wants to include in their illustration. Example subject matters include pencil, children, or wave. \nFor subject matters, no adjectives should be included. They should always be a noun.\nCome up with more than four new keywords for Subject matter.\n\nActions & poses are word-level or phrase-level actions that the character or the object in the illustration performs. Example actions & poses include riding a bus, standing still, or traveling.\n\nThemes & moods are words not directly present in the illustration, but those that can potentially convey the overall theme or mood of the illustration. Example themes & moods include imaginative, eco-friendly, or sad. \nThey should be adverbs, preferably one-word.\n\nYour task is to expand on the keywords being given, by combining multiple keywords or looking for synonyms that can inspire new creations or ideas.\n\nFor example, the subject matter "pencil" can be combined with the action & pose "traveling" to inspire a new action & pose "writing a diary". \n\nYou can combine as many keywords at once. Another example is to generate "hair salon" from "hair dryer", "comb", and "scissors"\n\nFor combinations that result in theme & mood, make them as abstract as possible. An example is to make "adventurous", "gusty" from "riding on ship" and "tent"\n\nCome up with new keywords for each keyword type with creative combinations. Only use the original keywords provided for creating new keywords.\nDo not just paraphrasing original keywords. Do not suggest similar keywords with original ones. \n\nImportant: Include at least one subject matter for each combination. Subject matter and theme & mood should be a SINGLE WORD. Combinations among subject matters are highly recommended. New keywords should be \'surprising\' compared to original ones. It means the character of your suggested word should has low similarity.',
            },
            {
                "role": "user",
                "content": "Subject matter: camping, tent, tree, animals, Eiffel tower, family\nAction & pose: riding on a bus, riding on a ship\nTheme & mood: playful, imaginative",
            },
            {
                "role": "assistant",
                "content": "Subject matter: bear, sleeping person, safari, cruise\nAction & pose: traveling, setting up camp, dancing jazz\nTheme & mood: adventurous, serene, joyful, romantic",
            },
            {
                "role": "user",
                "content": "Subject matter: boy, dinosaur, flower\nAction & pose: watching television\nTheme & mood: fantasy, playful",
            },
            {
                "role": "assistant",
                "content": "Subject matter: wind mill, volcano, movie screen\nAction & pose: exploding strongly, riding a dinosaur, flying away to the sky\nTheme & mood: vast, whimsical, rustic, frenetic",
            },
            {
                "role": "user",
                "content": "Subject matter: dreamy scene, boy\nAction & pose: playing with dino toys",
            },
            {
                "role": "assistant",
                "content": "Subject matter: universe, Saturn, astronauts\nAction & pose: imagining adventures, floating on the space, role-playing, daydreaming\nTheme & mood: jolly, imaginative, impactful",
            },
            {
                "role": "user",
                "content": "Subject matter: Christmas tree\nAction & pose: dancing around the Christmas tree\nTheme & mood: family-bonding",
            },
            {
                "role": "assistant",
                "content": "Subject matter: Fireplace, wooden sled, Snowman, jazz, piano\nAction & pose: melting, giving , body-warming\nTheme & mood: jubilant, sparkling, heartwarming",
            },
            {
                "role": "user",
                "content": "Subject matter: sea turtles, Christmas tree, marine life\nAction & pose: swimming, dancing around the Christmas tree\nTheme & mood: fantasy, underwater, family-bonding",
            },
            {
                "role": "assistant",
                "content": "Subject matter: Sea horse, Christmas lights, coral, mermaid\nAction & pose: floating on the wave, blinking eye, singing under the sea\nTheme & mood: ethereal, aquatic, charming, panoramic",
            },
            {
                "role": "user",
                "content": "Subject matter: kid, cat\nAction & pose: laying on top of a suitcase, playing hide and seek\nTheme & mood: Rustic, vivid, exhilarating",
            },
            {
                "role": "assistant",
                "content": "Subject matter: Birdcage, attic, trunk, blue bird\nAction & pose: jumping on boxes, chasing birds, hiding in a suitcase\nTheme & mood: quaint, mischievous, lively, nostalgic",
            },
            {"role": "user", "content": elementlist},
        ],
        temperature=1.2,
        max_tokens=200,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    res = []
    response_content = response.choices[0].message.content
    response_content = response_content.replace("Action & Pose", "Action & pose")
    response_content = response_content.replace("Action & Poses", "Action & pose")
    response_content = response_content.replace("Action & poses", "Action & pose")
    response_content = response_content.replace("Theme & Mood", "Theme & mood")
    response_content = response_content.replace("Theme & Moods", "Theme & mood")
    response_content = response_content.replace("Theme & moods", "Theme & mood")

    for line in response_content.split("\n"):
        keywordType = line.split(": ")[0]
        keywords = line.split(": ")[1]
        for keyword in keywords.split(", "):
            res.append({"type": keywordType, "keyword": keyword})

    return res


def keywords_to_descriptions(elementlist):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": 'The user wants to draw an illustration, with the assistance of you. You will be provided with multiple keywords users wanted to include in their illustrations. There are three types of keywords: Subject matter, Action & pose, and Theme & mood.\n\nSubject matters are one-word, describing the specific physical objects, characters, or landscape that the user wants to include in their illustration. Example subject matters include pencil, children, or wave. \nFor subject matters, no adjectives should be included. They should always be a noun.\n\nActions & poses are word-level or phrase-level actions that the character or the object in the illustration performs. Example actions & poses include riding a bus, standing still, or traveling.\n\nThemes & moods are words not directly present in the illustration, but those that can potentially convey the overall theme or mood of the illustration. Example themes & moods include imaginative, eco-friendly, or sad. \nThey should be adverbs, preferably one-word.\n\nYour task is to generate three descriptions of the illustration that the user can draw based on the given keywords. The three descriptions should be significantly different from each other. Each description should include three things: "Caption" and "Objects".\n\n"Caption" is a simple description of the overall image of the description. This should include some objects in the "Objects" list. Keep it concise. Do not make it long and do not include unnecessary adjectives.\n\n"Objects" is a list of the objects depicted in the illustrations, and a short description of them. The objects should be one of the given "Subject matters", or something related to them. You don\'t have to include all given subject matters. The given action and concept should also be considered for generating the object\'s detail.',
            },
            {
                "role": "user",
                "content": "Subject matter: ball, cat, dog\nAction & pose: jumping\nTheme & mood: playful, peaceful",
            },
            {
                "role": "assistant",
                "content": "1.\nScene: In the backyard, cats and dogs play happily with a ball which is their favorite toy.\nObjects: [(ball, a brightly colored ball), (dog, a dog jumping with its tongue sticking out), (cat, an agile cat in mid-jump)]\n\n2.\nScene: A beach where a dog is united in its pursuit of a bouncing ball.\nObjects: [(ball, a sandy beach ball), (dog, a spotted dog leaping for the ball)]\n\n3.\nScene: In a residential living room, the cat and dog both defy gravity in their jumps to catch the toy.\nObjects: [(toy, a small rubber toy with vibrant stripes), (dog, a bouncy terrier flying mid-air), (cat, a Siamese cat also in mid-jump)]",
            },
            {
                "role": "user",
                "content": "Subject matter: dinosaur, drink, hat, chair\nAction & pose: relaxing\nTheme & mood: sunny, relaxed",
            },
            {
                "role": "assistant",
                "content": "1.\nScene: A dinosaur wearing a hat, lounging with a drink on a chair under the sun.\nObjects: [(dinosaur, a large but friendly looking dinosaur), (drink, a fruity cocktail with a tiny umbrella), (hat, a wide-brimmed straw hat), (chair, a comfortable looking lounge chair)]\n\n2.\nScene: Dinosaur relaxes in a sophisticated environment with a drink and plops a hat on the chair next to it.\nObjects: [(dinosaur, a dinosaur in a suit), (drink, a fancy drink in a crystal glass), (hat, a stylish trilby), (chair, a plush velvet chair)]\n\n3.\nScene: A kid with a hat is playing on a picnic mat with a toy dinosaur.\nObjects: [(kid, a kid with pink shirts), (dinosaur, a small green toy dinosaur)]",
            },
            {
                "role": "user",
                "content": "Subject matter: dog, teeth\nAction & pose: \nTheme & mood: care",
            },
            {
                "role": "assistant",
                "content": "1.\nCaption: A caring veterinarian examining a dog's teeth.\nObjects: [(dog, a dog sitting calmly), (teeth, a pair of dental tools), (vet, a veterinarian wearing a lab coat)]\n\n2.\nCaption: A child diligently brushing their dog's teeth.\nObjects: [(dog, a dog lying on its back), (teeth, a toothbrush and dog toothpaste), (child, a child brushing the dog's teeth)]\n\n3.\nCaption: A dental hygienist showing a dog owner the correct brushing technique for their pet's teeth.\nObjects: [(dog, a dog standing on a dental examination table), (teeth, a dental mirror and toothbrush), (hygienist, a dental hygienist demonstrating the brushing technique)]",
            },
            {"role": "user", "content": elementlist},
        ],
        temperature=1,
        max_tokens=400,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    res = []
    for r in response.choices[0].message.content.split("\n\n"):
        objects = {}
        for line in r.split("\n"):
            if line.startswith("Objects"):
                curr_objs = line.split(": ")[1].strip("[]").split("), ")
                for obj in curr_objs:
                    obj = obj.strip("()").split(", ")
                    objects[obj[0]] = obj[1]
            elif line.startswith("Caption"):
                caption = line.split(": ")[1]
        try:
            res.append({"objects": objects, "caption": caption})
        except:
            pass

    return res


def caption_to_layout(caption, objects):
    input_prompt = caption + "\n[" + ", ".join(objects) + "]"

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are an intelligent bounding box generator. I will provide you with a caption for an illustration and a list of the objects. Your task is to generate the bounding boxes for the objects based on the caption.\nThe images are of size 512x512. The top-left corner has coordinate [0, 0]. The bottom-right corner has coordinnate [512, 512]. The bounding boxes should not go beyond the image boundaries.\nEach bounding box should be in the format of (object name, [top-left x coordinate, top-left y coordinate, box width, box height]). If an object must appear several times by the provided caption, multiple bounding boxes may be added for the object.",
            },
            {
                "role": "user",
                "content": "A realistic image of landscape scene depicting a green car parking on the left of a blue truck, with a red air balloon and a bird in the sky\n[car, truck, air balloon, bird]",
            },
            {
                "role": "assistant",
                "content": "[('car', [0.041, 0.549, 0.412, 0.31]), ('truck', [0.525, 0.553, 0.408, 0.313]), ('balloon', [0.129, 0.016, 0.283, 0.264]), ('bird', [0.578, 0.082, 0.279, 0.195])]",
            },
            {
                "role": "user",
                "content": "A realistic top-down view of a wooden table with two apples on it\n[wooden table, apple]",
            },
            {
                "role": "assistant",
                "content": "[('wooden table', [0.039, 0.289, 0.922, 0.422]), ('apple', [0.293, 0.441, 0.195, 0.195]), ('apple', [0.547, 0.441, 0.195, 0.195])]",
            },
            {
                "role": "user",
                "content": "A realistic scene of three skiers standing in a line on the snow near a palm tree\n[skier, palm tree]",
            },
            {
                "role": "assistant",
                "content": "[('skier', [0.01, 0.297, 0.271, 0.328]), ('skier', [0.543, 0.375, 0.236, 0.308]), ('skier', [0.289, 0.338, 0.242, 0.303]), ('palm tree', [0.789, 0.205, 0.201, 0.49])]",
            },
            {
                "role": "user",
                "content": "An oil painting of a pink dolphin jumping on the left of a steam boat on the sea\n[steam boat, dolphin]",
            },
            {
                "role": "assistant",
                "content": "[('steam boat', [0.453, 0.439, 0.502, 0.291]), ('dolphin', [0.041, 0.486, 0.369, 0.24])]",
            },
            {
                "role": "user",
                "content": "A cute cat and an angry dog without birds\n[cat, dog]",
            },
            {
                "role": "assistant",
                "content": "[('cat', [0.1, 0.131, 0.529, 0.632]), ('dog', [0.589, 0.232, 0.412, 0.445])]",
            },
            {
                "role": "user",
                "content": "Two pandas in a forest without flowers\n[panda]",
            },
            {
                "role": "assistant",
                "content": "[('panda', [0.059, 0.335, 0.414, 0.441]), ('panda', [0.516, 0.338, 0.434, 0.432])]",
            },
            {
                "role": "user",
                "content": "Immersed in his imagination, a boy is indoors enacting a prehistoric tale using two toy dinosaurs.\n[boy, dino toys]",
            },
            {
                "role": "assistant",
                "content": "[('boy', [0.367, 0.076, 0.301, 0.486]), ('dino toys', [0.15, 0.469, 0.201, 0.254]), ('dino toys', [0.543, 0.475, 0.188, 0.248])]",
            },
            {"role": "user", "content": input_prompt},
        ],
        temperature=0,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )

    response_content = response.choices[0].message.content
    # [('car', [21, 281, 211, 159]), ('truck', [269, 283, 209, 160]), ('balloon', [66, 8, 145, 135]), ('bird', [296, 42, 143, 100])]
    # response_content = ast.literal_eval(response_content)
    return response_content


def caption_ref_to_layout(caption, objects, refLayout):
    input_prompt = caption + "\n[" + ", ".join(objects) + "]\n" + str(refLayout)

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are an intelligent bounding box generator. I will provide you with a caption for an illustration, a list of the objects, and reference bounding boxes. Your task is to generate the bounding boxes for the objects based on the caption. The generated bounding boxes should be similar with the reference bounding boxes.\n\nEach bounding box should be in the format of (object name, [top-left x coordinate, top-left y coordinate, box width, box height]). The bounding boxes are represented as a proportion. The top-left corner has coordinate [0, 0]. The bottom-right corner has coordinnate [1, 1]. The bounding boxes should not go beyond the image boundaries. If an object must appear several times by the provided caption, multiple bounding boxes may be added for the object.\n\nThe generated and reference bounding boxes should be similar in terms of the overall arrangement of the objects. For example, if the reference bounding boxes are arranged in a way that makes a big space in the middle of the image, the generated bounding boxes also should leave some space in the middle as well. Another example is that, if the reference bounding boxes are biased toward the left side of the image, the generated bounding boxes should also biased toward left.",
            },
            {
                "role": "user",
                "content": "A realistic image of landscape scene depicting a green car parking on the left of a blue truck, with a red air balloon and a bird in the sky\n[car, truck, air balloon, bird]\n[[0.023, 0.785, 0.172, 0.181], [0.221, 0.692, 0.408, 0.274], [0.699, 0.580, 0.279, 0.244], [0.655, 0.841, 0.323, 0.125]]",
            },
            {
                "role": "assistant",
                "content": "[('car', [0.041, 0.783, 0.442, 0.179]), ('truck', [0.525, 0.699, 0.408, 0.263]), ('balloon', [0.261, 0.458, 0.222, 0.264]), ('bird', [0.525, 0.458, 0.279, 0.195])]",
            },
            {
                "role": "user",
                "content": "A realistic top-down view of a wooden table with two apples on it\n[wooden table, apple]\n[[0.023, 0.044, 0.520, 0.181], [0.570, 0.044, 0.408, 0.274], [0.023, 0.841, 0.581, 0.125], [0.655, 0.841, 0.323, 0.125]]",
            },
            {
                "role": "assistant",
                "content": "[('wooden table', [0.219, 0, 0.562, 1]), ('apple', [0.402, 0.138, 0.195, 0.195]), ('apple', [0.402, 0.667, 0.195, 0.195])]",
            },
            {
                "role": "user",
                "content": "A realistic scene of three skiers standing in a line on the snow near a palm tree\n[skier, palm tree]\n[[0.795, 0.353, 0.183, 0.441], [0.570, 0.044, 0.408, 0.274], [0.655, 0.841, 0.323, 0.125]]",
            },
            {
                "role": "assistant",
                "content": "[('skier', [0.487, 0.131, 0.142, 0.441]), ('skier', [0.661, 0.131, 0.143, 0.441]), ('skier', [0.836, 0.131, 0.142, 0.441]), ('palm tree', [0.795, 0.613, 0.183, 0.387])]",
            },
            {
                "role": "user",
                "content": "An oil painting of a pink dolphin jumping on the left of a steam boat on the sea\n[steam boat, dolphin]\n[[0.032, 0.018, 0.183, 0.285], [0.032, 0.340, 0.135, 0.420], [0.032, 0.787, 0.236, 0.168]]",
            },
            {
                "role": "assistant",
                "content": "[('steam boat', [0.273, 0, 0.245, 1]), ('dolphin', [0.032, 0.455, 0.135, 0.420])]",
            },
            {
                "role": "user",
                "content": "Immersed in his imagination, a boy is indoors enacting a prehistoric tale using four toy dinosaurs.\n[boy, dino toys]\n[[0.023, 0.044, 0.520, 0.181], [0.570, 0.044, 0.408, 0.274], [0.023, 0.841, 0.581, 0.125], [0.655, 0.841, 0.323, 0.125], [0.023, 0.253, 0.149, 0.563], [0.824, 0.33, 0.154, 0.496], [0.283, 0.375, 0.491, 0.407]]",
            },
            {
                "role": "assistant",
                "content": "[('boy', [0.250, 0.218, 0.566, 0.563]), ('dino toys', [0.074, 0.556, 0.137, 0.284]), ('dino toys', [0.074, 0.76, 0.137, 0.284]), ('dino toys', [0.659, 0.041, 0.254, 0.134]), ('dino toys', [0.464, 0.840, 0.195, 0.120])]",
            },
            {
                "role": "user",
                "content": "Two pandas in a forest without flowers\n[panda]\n[[0.795, 0.353, 0.183, 0.441], [0.570, 0.044, 0.408, 0.274], [0.655, 0.841, 0.323, 0.125], [0.032, 0.018, 0.183, 0.285], [0.032, 0.340, 0.135, 0.420], [0.032, 0.787, 0.236, 0.168]]",
            },
            {
                "role": "assistant",
                "content": "[('panda', [0.114, 0.399, 0.183, 0.441]), ('panda', [0.733, 0.106, 0.150, 0.441])]",
            },
            {"role": "user", "content": input_prompt},
        ],
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )

    response_content = response.choices[0].message.content
    # [('car', [21, 281, 211, 159]), ('truck', [269, 283, 209, 160]), ('balloon', [66, 8, 145, 135]), ('bird', [296, 42, 143, 100])]
    # response_content = ast.literal_eval(response_content)
    return response_content


def caption_layout_matcher(caption, objects, layout):
    input_prompt = caption + "\n[" + ", ".join(objects) + "]\n" + str(layout)

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are an intelligent bounding box matcher. I will provide you with a caption that describes an illustration, a list of the objects that are included in the illustration, and a list of bounding boxes. Your task is to match bounding boxes to each object to make the illustration most balanced and realistic.\n\nEach bounding box is in the format of (object name, [top-left x coordinate, top-left y coordinate, box width, box height]). The bounding boxes are represented as a proportion. The top-left corner has coordinate [0, 0]. The bottom-right corner has coordinnate [1, 1]. The bounding boxes should not go beyond the image boundaries.",
            },
            {
                "role": "user",
                "content": "A realistic image of landscape scene depicting a green car parking on the left of a blue truck, with a red air balloon and a bird in the sky\n[air balloon, car, bird, truck]\n[0.041, 0.783, 0.442, 0.179], [0.525, 0.699, 0.408, 0.263], [0.261, 0.458, 0.222, 0.264], [0.525, 0.458, 0.279, 0.195]",
            },
            {
                "role": "assistant",
                "content": "[('car', [0.041, 0.783, 0.442, 0.179]), ('truck', [0.525, 0.699, 0.408, 0.263]), ('balloon', [0.261, 0.458, 0.222, 0.264]), ('bird', [0.525, 0.458, 0.279, 0.195])]",
            },
            {
                "role": "user",
                "content": "A realistic top-down view of a wooden table with two apples on it\n[apple, apple, wooden table]\n[0.219, 0, 0.562, 1], [0.402, 0.138, 0.195, 0.195], [0.402, 0.667, 0.195, 0.195]",
            },
            {
                "role": "assistant",
                "content": "[('wooden table', [0.219, 0, 0.562, 1]), ('apple', [0.402, 0.138, 0.195, 0.195]), ('apple', [0.402, 0.667, 0.195, 0.195])]",
            },
            {
                "role": "user",
                "content": "A realistic scene of three skiers standing in a line on the snow near a palm tree\n[skier, skier, skier, palm tree]\n[0.487, 0.131, 0.142, 0.441], [0.661, 0.131, 0.143, 0.441], [0.836, 0.131, 0.142, 0.441], [0.795, 0.613, 0.183, 0.387]",
            },
            {
                "role": "assistant",
                "content": "[('skier', [0.487, 0.131, 0.142, 0.441]), ('skier', [0.661, 0.131, 0.143, 0.441]), ('skier', [0.836, 0.131, 0.142, 0.441]), ('palm tree', [0.795, 0.613, 0.183, 0.387])]",
            },
            {
                "role": "user",
                "content": "An oil painting of a pink dolphin jumping on the left of a steam boat on the sea\n[dolphin, steam boat]\n[0.273, 0, 0.245, 1], [0.032, 0.455, 0.135, 0.420]",
            },
            {
                "role": "assistant",
                "content": "[('steam boat', [0.273, 0, 0.245, 1]), ('dolphin', [0.032, 0.455, 0.135, 0.420])]",
            },
            {
                "role": "user",
                "content": "Immersed in his imagination, a boy is indoors enacting a prehistoric tale using four toy dinosaurs.\n[dino toys, dino toys, dino toys, boy, dino toys]\n[0.250, 0.218, 0.566, 0.563], [0.074, 0.556, 0.137, 0.284], [0.074, 0.76, 0.137, 0.284], [0.659, 0.041, 0.254, 0.134], [0.464, 0.840, 0.195, 0.120]",
            },
            {
                "role": "assistant",
                "content": "[('boy', [0.250, 0.218, 0.566, 0.563]), ('dino toys', [0.074, 0.556, 0.137, 0.284]), ('dino toys', [0.074, 0.76, 0.137, 0.284]), ('dino toys', [0.659, 0.041, 0.254, 0.134]), ('dino toys', [0.464, 0.840, 0.195, 0.120])]",
            },
            {
                "role": "user",
                "content": "Two pandas in a forest without flowers\n[panda, panda]\n[0.114, 0.399, 0.183, 0.441], [0.733, 0.106, 0.150, 0.441]",
            },
            {
                "role": "assistant",
                "content": "[('panda', [0.114, 0.399, 0.183, 0.441]), ('panda', [0.733, 0.106, 0.150, 0.441])]",
            },
            {"role": "user", "content": input_prompt},
        ],
        temperature=0,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )

    response_content = response.choices[0].message.content
    # [('car', [21, 281, 211, 159]), ('truck', [269, 283, 209, 160]), ('balloon', [66, 8, 145, 135]), ('bird', [296, 42, 143, 100])]
    response_content = ast.literal_eval(response_content)
    return response_content
