import numpy as np
from PIL import Image
import cv2
from diffusers.utils import load_image
from datetime import datetime

img_name = 'overlay.png'.split('.')[0]
image = load_image(
        img_name+'.png'
    )

image = np.array(image)

low_threshold = 200
high_threshold = 600

canny_image = cv2.Canny(image, low_threshold, high_threshold)
canny_image = canny_image[:, :, None]
canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
canny_image_invert = cv2.bitwise_not(canny_image)
canny_image = Image.fromarray(canny_image)
canny_image_invert = Image.fromarray(canny_image_invert)

merged = Image.new('RGB', (1024, 512), (255, 255, 255))
merged.paste(canny_image_invert, (0, 0))
merged.paste(canny_image, (512, 0))
    
merged.save(f'canny_{img_name}_{datetime.now().microsecond}.png')
# print(f'Saved controlnet_{now}.png ...')

# canny_image.save(f'canny_{img_name}_{datetime.now().microsecond}.png')