from PIL import Image
from torchvision import transforms
import torch
import torch.nn as nn

# Import config and load_model_for_testing from Utils

import sys
sys.path.append('./')
import Config.config as config
from Utils.utils import load_model_for_testing
from Model.model import MultiLayerFCNet

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize the model
model = MultiLayerFCNet(config.INPUT_SIZE, config.HIDDEN_SIZE, config.OUTPUT_SIZE)

# Wrap the model with nn.DataParallel if using multiple GPUs
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

# Load the saved model
model = load_model_for_testing(model, config.MODEL_SAVE_PATH).to(device)

# Set the model to evaluation mode
model.eval()

# Transform for the single image
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5836, 0.4212, 0.3323], std=[0.2325, 0.1985, 0.1722])
])

# Function to test a single image
def test_single_image(image_path, model, device, transform):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)
    
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    
    return predicted.item()

# Path to your single image
image_path = './Dataset/Test/Angry/3370Exp0angry_worker_52.jpg'

#plot the image and set the original lebel and the predicted label

# Get prediction for the single image
predicted_class_label = test_single_image(image_path, model, device, transform)

print(f'Predicted class label:  {config.CLASS_NAMES[predicted_class_label] }')

import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Plot the image
#plot in small size
plt.figure(figsize=(3, 3))
plt.imshow(img)
# Get the original class label
original_class_label = os.path.basename(os.path.dirname(image_path))
#title should be the predicted label and the original label
plt.title(f'Predicted: {config.CLASS_NAMES[predicted_class_label]}, Original: {original_class_label}')
plt.show()
