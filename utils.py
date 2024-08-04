import cv2
import numpy as np
from PIL import Image

def input_image(uploaded_image):
    #file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    # original_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    # global image_counter
    original_image = Image.open(uploaded_image)
    master_id = 'A'
    return original_image, master_id