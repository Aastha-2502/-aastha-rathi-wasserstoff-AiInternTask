import easyocr
import numpy as np
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

def extract_text_from_bbox_easy_ocr_bclip(image, bbox):
    # model
    # easy ocr
    reader = easyocr.Reader(['en'], gpu=False)
    # blip
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = BlipProcessor.from_pretrained("noamrot/FuseCap")
    model = BlipForConditionalGeneration.from_pretrained("noamrot/FuseCap").to(device)

    # cropping the images
    image_array = np.array(image)
    x1, y1, x2, y2 = map(int, bbox)
    cropped_img = image_array[y1:y2, x1:x2]
    #cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
    #cropped_img = Image.fromarray(cropped_img)

    # easy ocr
    results = reader.readtext(cropped_img)

    # Extract and return the text
    text_lines = [text for (_, text, _) in results]
    ocr_result = " ".join(text_lines) if text_lines else "-"

    # Blip
    text = "A picture of"
    inputs = processor(cropped_img, text, return_tensors="pt").to(device)

    out = model.generate(**inputs, num_beams=3)
    blip_result = processor.decode(out[0], skip_special_tokens=True)

    return ocr_result, blip_result
