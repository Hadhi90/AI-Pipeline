import streamlit as st
from PIL import Image
import os
import json
import cv2
import matplotlib.pyplot as plt
import numpy as np
import uuid
import pandas as pd
import torch
from torchvision import models, transforms
from transformers import pipeline

# Define paths
input_image_path = "project_root/data/input_images/sample.jpg"
segmented_objects_dir = "project_root/data/segmented_objects/"
output_dir = "project_root/data/output/"

# Streamlit UI setup
st.title("AI Pipeline for Image Segmentation and Object Analysis")

# File upload
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Save uploaded image
    uploaded_image_path = os.path.join(input_image_path)
    image.save(uploaded_image_path)
    
    st.write("Image uploaded successfully!")

    # Step 1: Image Segmentation
    st.write("Processing image...")

    # Import your segmentation code here
    model = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        predictions = model(image_tensor)

    masks = predictions[0]['masks'].numpy()
    boxes = predictions[0]['boxes'].numpy()

    # Visualization
    image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    for i in range(len(masks)):
        mask = masks[i, 0]
        image_np[mask > 0.5] = [0, 255, 0]

    st.image(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB), caption='Segmented Image', use_column_width=True)

    # Save segmented objects
    os.makedirs(segmented_objects_dir, exist_ok=True)
    master_id = str(uuid.uuid4())
    metadata = {"master_id": master_id, "objects": []}

    for i, (mask, box) in enumerate(zip(masks, boxes)):
        unique_id = str(uuid.uuid4())
        x_min, y_min, x_max, y_max = map(int, box)
        object_img = image_np[y_min:y_max, x_min:x_max]
        object_mask = mask[y_min:y_max, x_min:x_max]
        object_img = cv2.bitwise_and(object_img, object_img, mask=(object_mask > 0.5).astype("uint8"))
        object_filename = f"{unique_id}.png"
        cv2.imwrite(os.path.join(segmented_objects_dir, object_filename), object_img)
        metadata["objects"].append({
            "unique_id": unique_id,
            "file_name": object_filename,
            "bounding_box": [x_min, y_min, x_max, y_max]
        })

    # Save metadata
    metadata_path = os.path.join(segmented_objects_dir, f"{master_id}_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    
    st.write("Image segmentation completed.")

    # Step 2: Object Identification
    st.write("Identifying objects...")

    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    for obj in metadata['objects']:
        object_img_path = os.path.join(segmented_objects_dir, obj['file_name'])
        image = Image.open(object_img_path)
        image_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            predictions = model(image_tensor)

        labels = predictions[0]['labels'].numpy()
        scores = predictions[0]['scores'].numpy()
        label_idx = scores.argmax()
        obj_label = labels[label_idx]
        obj_confidence = scores[label_idx]
        obj['label'] = str(obj_label)
        obj['confidence'] = float(obj_confidence)

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    st.write("Object identification completed.")

    # Step 3: Text/Data Extraction
    st.write("Extracting text/data...")

    import pytesseract
    import easyocr
    
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    USE_TESSERACT = True

    if USE_TESSERACT:
        pass
    else:
        reader = easyocr.Reader(['en'])

    for obj in metadata['objects']:
        object_img_path = os.path.join(segmented_objects_dir, obj['file_name'])
        image = cv2.imread(object_img_path)
        if USE_TESSERACT:
            extracted_text = pytesseract.image_to_string(image)
        else:
            extracted_text = reader.readtext(image, detail=0)
        obj['extracted_text'] = extracted_text

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    st.write("Text extraction completed.")

    # Step 4: Summarize Attributes
    st.write("Summarizing attributes...")

    summarizer = pipeline("summarization")

    for obj in metadata['objects']:
        if 'extracted_text' in obj and obj['extracted_text']:
            text = obj['extracted_text']
            summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
            obj['summary'] = summary[0]['summary_text']
        else:
            obj['summary'] = "No text extracted"

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    st.write("Summarization completed.")

    # Step 5: Data Mapping
    st.write("Mapping data...")

    mapped_data = {"master_image": {"file_name": uploaded_file.name, "objects": []}}

    for obj in metadata['objects']:
        mapped_data['master_image']['objects'].append({
            "unique_id": obj.get('unique_id', 'N/A'),
            "file_name": obj.get('file_name', 'N/A'),
            "label": obj.get('label', 'N/A'),
            "confidence": obj.get('confidence', 'N/A'),
            "extracted_text": obj.get('extracted_text', 'N/A'),
            "summary": obj.get('summary', 'N/A')
        })

    mapped_data_path = os.path.join(output_dir, "mapped_data.json")
    with open(mapped_data_path, "w") as f:
        json.dump(mapped_data, f, indent=4)

    st.write("Data mapping completed.")

    # Display the final output image with annotations
    st.write("Generating final output...")

    image = cv2.imread(input_image_path)

    with open(mapped_data_path, "r") as f:
        mapped_data = json.load(f)

    for obj in mapped_data['master_image']['objects']:
        unique_id = obj.get('unique_id', 'N/A')
        file_name = obj.get('file_name', 'N/A')
        x, y, w, h = 50, 50, 100, 100  # Replace with actual coordinates
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        label = f"ID: {unique_id} | {file_name}"
        cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    annotated_image_path = os.path.join(output_dir, "annotated_image.jpg")
    cv2.imwrite(annotated_image_path, image)

    st.image(annotated_image_path, caption='Annotated Image', use_column_width=True)

    # Display the summary table
    st.write("Summary Table:")

    data = []
    for obj in mapped_data['master_image']['objects']:
        data.append({
            "Unique ID": obj.get('unique_id', 'N/A'),
            "File Name": obj.get('file_name', 'N/A'),
            "Label": obj.get('label', 'N/A'),
            "Confidence": obj.get('confidence', 'N/A'),
            "Extracted Text": obj.get('extracted_text', 'N/A'),
            "Summary": obj.get('summary', 'N/A')
        })

    df = pd.DataFrame(data)
    st.dataframe(df)


