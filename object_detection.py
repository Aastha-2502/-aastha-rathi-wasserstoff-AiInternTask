import ultralytics
from collections import defaultdict
import cv2
from PIL import Image
import streamlit as st

object_counter = defaultdict(int)

def yoloV8(image,master_id):
    model = ultralytics.YOLO('yolov8l.pt')
    results = model.predict(image, conf = 0.6)

    global object_counter
    object_ids = list()
    objectID_label = list()
    object_counter[master_id] = 1
    for result in results:
        # num_objects = len(result.boxes)
        boxes = result.boxes
        for _, box in enumerate(boxes):
            class_index = int(box.cls)
            object_counter[master_id] += 1
            object_id = f'{master_id}{object_counter[master_id]}'
            object_ids.append(object_id)
            objectID_label.append({"objectID": object_id,
                                   "object_type": result.names[class_index]})

            # Mapping Master IDs with Objects IDs
    master_dict = {"MasterImage": {"ID": master_id, "objects": objectID_label}}

    return results, master_dict, boxes, object_ids

def annotated_image(results, object_ids):
    for result in results:
        annotated_image = result.plot()
        for box, obj_id in zip(result.boxes.xyxy, object_ids):
            x1, y1, x2, y2 = map(int, box)  # Box coordinates
            cv2.putText(
                annotated_image,
                f"{obj_id}",
                (x1 + 20, y2 - 20),  # Place text above the top-left corner of the bounding box
                cv2.FONT_HERSHEY_SIMPLEX,
                1,  # Font scale
                (0, 0, 255),  # Font color in BGR (blue)
                3,  # Line thickness
                cv2.LINE_AA)
        rgb_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

        pil_image = Image.fromarray(rgb_image)

        st.image(pil_image, caption=f"Annotated Image", use_column_width=True)


