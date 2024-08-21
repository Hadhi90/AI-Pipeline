#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install torch torchvision opencv-python


# In[2]:


# importing the needed packages 

import torch
from torchvision import models, transforms
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Here i am using pre-trained Mask R-CNN model
model = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

# we will define image transformation
transform = transforms.Compose([
    transforms.ToTensor()
])

# Preprocessing the image from the directory
image = Image.open("project_root/data/input_images/sample.jpg")
image_tensor = transform(image).unsqueeze(0)

# we will perform segmentation here, 
with torch.no_grad():
    predictions = model(image_tensor)

# Extract masks and bounding boxes
masks = predictions[0]['masks'].numpy()
boxes = predictions[0]['boxes'].numpy()

# Finally we will visualize the results
image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
for i in range(len(masks)):
    mask = masks[i, 0]
    image_np[mask > 0.5] = [0, 255, 0]  # Apply green color to segmented areas

plt.imshow(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
plt.show()


# ### Step 2 : Object Extractoin and Storage

# In[3]:


# Needed Packages

import os
import cv2
import uuid
import json
from PIL import Image

# Defining paths
input_image_path = "project_root/data/input_images/sample.jpg"
output_dir = "project_root/data/segmented_objects/"
os.makedirs(output_dir, exist_ok=True)

# Load the original image
original_image = cv2.imread(input_image_path)

masks = predictions[0]['masks'].numpy()
boxes = predictions[0]['boxes'].numpy()

# Giving unique ID for the master image
master_id = str(uuid.uuid4())

# Metadata dictionary
metadata = {
    "master_id": master_id,
    "objects": []
}

# Iterate through each object and extract it
for i, (mask, box) in enumerate(zip(masks, boxes)):
    unique_id = str(uuid.uuid4())  # Generate unique ID for each object
    
    # Giving the bounding box coordinates
    x_min, y_min, x_max, y_max = map(int, box)
    
    # Extract the object using the mask
    object_img = original_image[y_min:y_max, x_min:x_max]
    object_mask = mask[y_min:y_max, x_min:x_max]
    
    # Apply the mask to the object image
    object_img = cv2.bitwise_and(object_img, object_img, mask=(object_mask > 0.5).astype("uint8"))
    
    # Saving the object image
    object_filename = f"{unique_id}.png"
    cv2.imwrite(os.path.join(output_dir, object_filename), object_img)
    
    # Update metadata
    metadata["objects"].append({
        "unique_id": unique_id,
        "file_name": object_filename,
        "bounding_box": [x_min, y_min, x_max, y_max]
    })

# Save metadata to a JSON file
metadata_path = os.path.join(output_dir, f"{master_id}_metadata.json")
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=4)

print(f"Extracted and saved {len(masks)} objects with metadata.")


# ### Step 3 : Object identification 
# 

# In[4]:


## Importing the Needed Packages

import torch
from torchvision import models, transforms
from PIL import Image
import json
import os

# We will make use of pre-trained Faster R-CNN model
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Transformation for the input image
transform = transforms.Compose([
    transforms.ToTensor()
])

# Load metadata
output_dir = "project_root/data/segmented_objects/"
metadata_path = os.path.join(output_dir, "your_metadata_file.json")

with open(metadata_path, "r") as f:
    metadata = json.load(f)

# Identifying the objects
for obj in metadata['objects']:
    # Load the object image
    object_img_path = os.path.join(output_dir, obj['file_name'])
    image = Image.open(object_img_path)
    image_tensor = transform(image).unsqueeze(0)
    
    # Perform object detection
    with torch.no_grad():
        predictions = model(image_tensor)
    
    # Extract the labels and scores
    labels = predictions[0]['labels'].numpy()
    scores = predictions[0]['scores'].numpy()
    
    # Get the highest confidence label
    label_idx = scores.argmax()
    obj_label = labels[label_idx]
    obj_confidence = scores[label_idx]
    
    # Update metadata with the object label and confidence
    obj['label'] = str(obj_label)
    obj['confidence'] = float(obj_confidence)

# Save the updated metadata
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=4)

print("Object identification completed and metadata updated.")


# ### Step 4 : Text/Data Extraction

# In[5]:


pip install pytesseract easyocr


# In[8]:


get_ipython().system('pip install pytesseract')


# In[17]:


get_ipython().system('pip install easyocr ')


# In[3]:


# Needed Packages

import pytesseract
import easyocr
import cv2
import os
import json
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'



USE_TESSERACT = True  


if USE_TESSERACT:
    pass
else:
    reader = easyocr.Reader(['en'])

#metadata
output_dir = "project_root/data/segmented_objects/"
metadata_path = os.path.join(output_dir, "your_metadata_file.json")

with open(metadata_path, "r") as f:
    metadata = json.load(f)

# Extract text/data from objects
for obj in metadata['objects']:
    object_img_path = os.path.join(output_dir, obj['file_name'])
    
    # Image Loading
    image = cv2.imread(object_img_path)
    
    if USE_TESSERACT:
        extracted_text = pytesseract.image_to_string(image)
    else:
        extracted_text = reader.readtext(image, detail=0)

    # Metadata Updation
    obj['extracted_text'] = extracted_text

# Saving
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=4)

print("Text extraction completed and metadata updated.")


# In[1]:


pip install --upgrade pip


# ### Step 5 : Summariazation

# In[4]:


pip install transformers torch


# In[1]:


# Needed Packages

from transformers import pipeline
import json
import os

#summarization model
summarizer = pipeline("summarization")

#metadata file
output_dir = "project_root/data/segmented_objects/"
metadata_path = os.path.join(output_dir, "your_metadata_file.json")

with open(metadata_path, "r") as f:
    metadata = json.load(f)

# Summarizing 
for obj in metadata['objects']:
    if 'extracted_text' in obj and obj['extracted_text']:
        text = obj['extracted_text']
        
        
        summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
        obj['summary'] = summary[0]['summary_text']
    else:
        obj['summary'] = "No text extracted"

# Save
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=4)

print("Summarization completed and metadata updated.")


# ### Step 6 : Data Mapping

# In[2]:


# Needed Packages

import json
import os

#metadata file
output_dir = "project_root/data/segmented_objects/"
metadata_path = os.path.join(output_dir, "your_metadata_file.json")

# Loading
with open(metadata_path, "r") as f:
    metadata = json.load(f)

# Create a dictionary
mapped_data = {
    "master_image": {
        "file_name": "input_image.jpg",  # Replace with actual master image file name
        "objects": []
    }
}

# Map object data
for obj in metadata['objects']:
    mapped_data['master_image']['objects'].append({
        "unique_id": obj.get('unique_id', 'N/A'),
        "file_name": obj.get('file_name', 'N/A'),
        "label": obj.get('label', 'N/A'),
        "confidence": obj.get('confidence', 'N/A'),
        "extracted_text": obj.get('extracted_text', 'N/A'),
        "summary": obj.get('summary', 'N/A')
    })

# Save
mapped_data_path = os.path.join(output_dir, "mapped_data.json")
with open(mapped_data_path, "w") as f:
    json.dump(mapped_data, f, indent=4)

print("Data mapping completed and saved to mapped_data.json.")


# ### Step 7 : Output Generation

# In[4]:


# Needed Packages

import cv2
import json
import os

# Paths
output_dir = "project_root/data/output/"
input_image_path = "project_root/data/input_images/sample.jpg"
metadata_path = os.path.join(output_dir, "mapped_data.json")

# Input Image
image = cv2.imread(input_image_path)

with open(metadata_path, "r") as f:
    mapped_data = json.load(f)

#annotations
for obj in mapped_data['master_image']['objects']:
    unique_id = obj.get('unique_id', 'N/A')
    file_name = obj.get('file_name', 'N/A')
    x, y, w, h = 50, 50, 100, 100  
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    label = f"ID: {unique_id} | {file_name}"
    cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

# Save
annotated_image_path = os.path.join(output_dir, "annotated_image.jpg")
cv2.imwrite(annotated_image_path, image)

print("Annotated image saved.")


# #### Summary Table

# In[5]:


## Packages

import pandas as pd

#mapped data
with open(metadata_path, "r") as f:
    mapped_data = json.load(f)

#Data Frame for Sumamry Table
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

# Save
summary_table_path = os.path.join(output_dir, "summary_table.csv")
df.to_csv(summary_table_path, index=False)

print("Summary table saved as summary_table.csv.")


# In[9]:


pip install streamlit


# In[ ]:




