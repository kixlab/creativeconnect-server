import openai

openai.api_key = "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"


def caption_to_keywords(description):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You will be provided with multiple sentences to describe an illustration. Your task is to extract a list of Subject matter, Action & pose, and Theme & mood.\n\nSubject matters are one-word, describing the specific physical objects, characters, or landscape that the user wants to include in their illustration. Example subject matters include pencil, children, or wave. \nFor subject matters, no adjectives should be included. They should always be a noun.\n\nActions & poses are word-level or phrase-level actions that the character or the object in the illustration performs. Example actions & poses include riding a bus, standing still, or traveling.\n\nThemes & moods are words not directly present in the illustration, but those that can potentially convey the overall theme or mood of the illustration. Example themes & moods include imaginative, eco-friendly, or sad. \nThey should be adverbs, preferably one-word.",
            },
            {
                "role": "user",
                "content": "a card with chinese writing with colorful objects on it\na red and orange background with a blank paper, chinese, pencils, stationery and more\nan image of a classroom scene with various supplies",
            },
            {
                "role": "assistant",
                "content": "Subject matter: card, Chinese writing, colorful objects, red and orange background, blank paper, Chinese, pencils, stationery, classroom, supplies.\n\nAction & pose: \n\nTheme & mood: education, learning, multiculturalism.",
            },
            {"role": "user", "content": description},
        ],
        temperature=0.5,
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
                "content": 'We are trying to support novice designers\' ideation process by semantically combining different parts of references (illustrations). You will be provided with the topic of the ideation, and multiple keywords users like in the illustrations they found as references. There are three types of keywords: Subject matter, Action & Pose, and Theme & Mood.\n\nSubject matters are one-word, describing the specific physical objects, characters, or landscape that the user wants to include in their illustration. Example subject matters include pencil, children, or wave. \nFor subject matters, no adjectives should be included. They should always be a noun.\n\nActions & poses are word-level or phrase-level actions that the character or the object in the illustration performs. Example actions & poses include riding a bus, standing still, or traveling.\n\nThemes & moods are words not directly present in the illustration, but those that can potentially convey the overall theme or mood of the illustration. Example themes & moods include imaginative, eco-friendly, or sad. \nThey should be adverbs, preferably one-word.\n\nYour task is to expand on the keywords being given, by combining multiple keywords or looking for synonyms that can inspire new creations or ideas.\n\nFor example, the subject matter "pencil" can be combined with the action & pose "traveling" to inspire a new action & pose "writing a diary". \n\nYou can combine as many keywords at once. Another example is to generate "hair salon" from "hair dryer", "comb", and "scissors"\n\nFor combinations that result in theme & mood, make them as abstract as possible. An example is to make "adventurous" from "riding on ship" and "tent"\n\nCome up with four new keywords for each keyword type with creative combinations. Only use the original keywords provided for creating new keywords.\n\nImportant: Include at least one subject matter for each combination. Subject matter and theme & mood should be a SINGLE WORD. Combinations among subject matters are highly recommended. ',
            },
            {
                "role": "user",
                "content": "Subject matter: camping, tent, tree, animals, Eiffel tower, family\nAction & pose: riding on a bus, riding on a ship\nTheme & mood: playful, imaginative",
            },
            {
                "role": "assistant",
                "content": "Subject matter: camping site, forest, Paris scene, safari\nAction & pose: traveling, setting up camp, touring Paris, going on a picnic\nTheme & mood: adventurous, serene, joyful, romantic",
            },
            {
                "role": "user",
                "content": "Subject matter: boy, dinosaur, flower\nAction & pose: watching television\nTheme & mood: fantasy, playful",
            },
            {
                "role": "assistant",
                "content": "Subject matter: child, nature, prehistoric creature, Jurassic Park\nAction & pose: exploring wilderness, sightseeing, reading a book\nTheme & mood: mysterious, whimsical, tranquil, jolly",
            },
            {"role": "user", "content": elementlist},
        ],
        temperature=1,
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
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": 'The user wants to draw an illustration, with the assistance of you. You will be provided with multiple keywords users wanted to include in their illustrations. There are three types of keywords: Subject matter, Action & pose, and Theme & mood.\n\nSubject matters are one-word, describing the specific physical objects, characters, or landscape that the user wants to include in their illustration. Example subject matters include pencil, children, or wave. \nFor subject matters, no adjectives should be included. They should always be a noun.\n\nActions & poses are word-level or phrase-level actions that the character or the object in the illustration performs. Example actions & poses include riding a bus, standing still, or traveling.\n\nThemes & moods are words not directly present in the illustration, but those that can potentially convey the overall theme or mood of the illustration. Example themes & moods include imaginative, eco-friendly, or sad. \nThey should be adverbs, preferably one-word.\n\nYour task is to generate three descriptions of the illustration that the user can draw based on the given keywords. The three descriptions should be significantly different with each other. Each description should include three things: "Scene", "Background", and "Objects".\n\n"Scene" is a simple description of the overall image of the description. This should include all objects in the "Objects" list. Do not make it too long.\n\n"Objects" is a list of the objects depicted in the illustrations, and a short description of them. The objects should be one of the given "Subject matters", or something related to them. You don\'t have to include all given subject matters. The given action and concept should also be considered for generating the object\'s detail.\n\nThe background is a short concise sentence describing the background of the illustration. It should be well aligned with the objects depicted and the given concepts.',
            },
            {
                "role": "user",
                "content": "Subject matter: ball, cat, dog\nAction & pose: jumping\nTheme & mood: playful, peaceful",
            },
            {
                "role": "assistant",
                "content": "1.\nScene: A joyous spectacle unfolds in a vibrant backyard as a cat and a dog are engaged in delightful play, with a ball being their favorite toy.\nObjects: [(ball, a brightly colored ball), (dog, a dog jumping with its tongue sticking out), (cat, an agile cat in mid-jump)]\nBackground: A suburban backyard with lush green grass and bright flowers.\n\n2.\nScene: A beach where a dog is united in its pursuit of a bounce ball.\nObjects: [(ball, a sandy beach ball), (dog, a spotted dog leaping for the ball)]\nBackground: A peaceful beach vista, with playful seagulls and gentle waves crashing against the shore.\n\n3.\nScene: In a residential living room, the cat and dog both defying gravity in their jumps to catch the toy.\nObjects: [(toy, a small rubber toy with vibrant stripes), (dog, a bouncy terrier flying mid-air), (cat, a Siamese cat also in mid-jump)]\nBackground: A warm and cozy living room with sunlight pouring in through the windows.",
            },
            {
                "role": "user",
                "content": "Subject matter: woman, drink, hat, chair\nAction & pose: sipping drink, relaxing, sitting\nTheme & mood: sunny, relaxed",
            },
            {
                "role": "assistant",
                "content": "1.\nScene: A relaxed woman wearing a wide-brimmed hat is sitting in a comfortable chair, sipping her cool drink.\nObjects: [(woman, a casual-dressed woman with a relaxed smile on her face), (drink, a sweating glass filled with a luscious summer cocktail), (hat, the woman's wide-brimmed straw hat), (chair, a comfy chair with soft white cushions)]\nBackground: On a sunny outdoor patio with lush, green foliage around. \n\n2.\nScene: A stylish woman is peacefully enjoying her drink at an elegant street café.\nObjects: [(woman, an elegantly dressed woman with a serene facial expression), (drink, a classy cocktail in a stemmed glass), (chair, a classic bistro chair with a hat hanging on its backrest)]\nBackground:  An outdoor café in a city street against the backdrop of colorful buildings. \n\n3.\nScene: The woman in a panama hat relaxes on a mat with her drink, feeling the warmth of the sun. \nObjects: [(woman, a woman in a breezy summer dress), (drink, a jug of iced lemonade), (hat, a jaunty panama hat rested on her knee)]\nBackground: A porch area with a garden in the background and sunlight filtering through the leaves.",
            },
            {
                "role": "user",
                "content": "Subject matter: boy, dreamy scene\nAction & pose: playing with dino toys",
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
        background = ""
        for line in r.split("\n"):
            if line.startswith("Objects"):
                curr_objs = line.split(": ")[1].strip("[]").split("), ")
                for obj in curr_objs:
                    obj = obj.strip("()").split(", ")
                    objects[obj[0]] = obj[1]
            elif line.startswith("Background"):
                background = line.split(": ")[1]
            elif line.startswith("Scene"):
                scene = line.split(": ")[1]
        try:
            res.append({"objects": objects, "background": background, "scene": scene})
        except:
            pass

    return res


def caption_to_layout(caption, objects):
    input_prompt = caption + "\n[" + ", ".join(objects) + "]"

    response = openai.ChatCompletion.create(
        model="gpt-4",
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
                "content": "[('car', [21, 281, 211, 159]), ('truck', [269, 283, 209, 160]), ('balloon', [66, 8, 145, 135]), ('bird', [296, 42, 143, 100])]",
            },
            {
                "role": "user",
                "content": "A realistic top-down view of a wooden table with two apples on it\n[wooden table, apple]",
            },
            {
                "role": "assistant",
                "content": "[('wooden table', [20, 148, 472, 216]), ('apple', [150, 226, 100, 100]), ('apple', [280, 226, 100, 100])]",
            },
            {
                "role": "user",
                "content": "A realistic scene of three skiers standing in a line on the snow near a palm tree\n[skier, palm tree]",
            },
            {
                "role": "assistant",
                "content": "[('skier', [5, 152, 139, 168]), ('skier', [278, 192, 121, 158]), ('skier', [148, 173, 124, 155]), ('palm tree', [404, 105, 103, 251])]",
            },
            {
                "role": "user",
                "content": "An oil painting of a pink dolphin jumping on the left of a steam boat on the sea\n[steam boat, dolphin]",
            },
            {
                "role": "assistant",
                "content": "[('steam boat', [232, 225, 257, 149]), ('dolphin', [21, 249, 189, 123])]",
            },
            {
                "role": "user",
                "content": "A cute cat and an angry dog without birds\n[cat, dog]",
            },
            {
                "role": "assistant",
                "content": "[('cat', [51, 67, 271, 324]), ('dog', [302, 119, 211, 228])]",
            },
            {
                "role": "user",
                "content": "Two pandas in a forest without flowers\n[panda]",
            },
            {
                "role": "assistant",
                "content": "[('panda', [30, 171, 212, 226]), ('panda', [264, 173, 222, 221])]",
            },
            {
                "role": "user",
                "content": "Immersed in his imagination, a boy is indoors enacting a prehistoric tale using two toy dinosaurs.\n[boy, dino toys]",
            },
            {
                "role": "assistant",
                "content": "[('boy', [188, 39, 154, 249]), ('dino toys', [77, 240, 103, 130]), ('dino toys', [278, 243, 96, 127])]",
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
