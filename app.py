import streamlit as st
from PIL import Image
import numpy as np
from object_detection import yoloV8, annotated_image
from utils import input_image
from text_extraction import extract_text_from_bbox_easy_ocr_bclip
from table import output_table

def main():
    st.title("Frame Finder")
    uploaded_file = st.file_uploader("Upload Image", type = ['jpg','jpeg'])
    col1, col2 = st.columns(2)
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        with col1:
            st.image(image, caption='Uploaded Image', use_column_width=True)
        image, master_id = input_image(uploaded_file)

        with col2:
            with st.spinner('Processing...'):
                results, master_dict, boxes, object_ids = yoloV8(image, master_id)
                annotated_image(results, object_ids)
            
        for i, bbox in enumerate(boxes.xyxy):
            text, attribute = extract_text_from_bbox_easy_ocr_bclip(image, bbox)

            # Update each object in the dictionary
            if i < len(master_dict['MasterImage']['objects']):
                master_dict['MasterImage']['objects'][i]['text'] = text
                master_dict['MasterImage']['objects'][i]['attribute'] = attribute

        df = output_table(master_dict)
        
        with st.spinner('Loading the table...'):
            st.subheader('Detected Objects')
            st.table(df)

if __name__ == "__main__":
    main()

