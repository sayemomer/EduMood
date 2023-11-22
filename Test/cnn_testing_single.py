import random
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
def get_random_images(test_dir, num_images=10):
    all_images = []
    for class_dir in os.listdir(test_dir):
        class_path = os.path.join(test_dir, class_dir)
        if os.path.isdir(class_path):
            images = [os.path.join(class_path, img) for img in os.listdir(class_path)]
            all_images.extend(images)
    return random.sample(all_images, num_images)

# Get 10 random images
random_images = get_random_images(config.TEST_DATA_PATH, 10)

plt.title(f'Predicted: {config.CLASS_NAMES[predicted_class_label]}, Original: {original_class_label}')
plt.show()

# Plot and predict for each image on a grid
num_rows = 2
num_cols = 5

fig, axes = plt.subplots(num_rows, num_cols, figsize=(1.5*num_cols,2*num_rows))
for i in range(len(random_images)):
    predicted_class_label = test_single_image(random_images[i], model, device, transform)
    
    # Load and plot the image
    img = cv2.imread(random_images[i])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    axes.ravel()[i].imshow(img)
    
    # Get the original class label
    original_class_label = os.path.basename(os.path.dirname(random_images[i]))
    
    # Show predicted and original labels
    axes.ravel()[i].set_title(f'Predicted: {config.CLASS_NAMES[predicted_class_label]}, Original: {original_class_label}',fontsize=6)
    axes.ravel()[i].set_axis_off()
plt.tight_layout()
plt.show()



