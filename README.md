
# Applied Artificial Intelligence and Classroom Activity Recognition 

![Project Banner](images/output.png)

## Overview
A Deep Learning Convolutional Neural Network (CNN) using PyTorch that analyses images of students in a classroom or online meeting setting and categorizes them into distinct states or activities.

1. **Facial Expression Recognition:**  Convolutional Neural Networks (CNN) for facial expression recognition. The dataset for this project includes images from a few publicly available datasets containing nu-
merous human facial expressions.

2. **Classroom Activity Recognition:** A Deep Learning CNN using PyTorch designed to analyze images of students in a classroom or online meeting setting, categorizing them into distinct states or activities.

## Classification Labels

- Angry
- Neutral
- Engaged
- Bored


# Data Collection & Preprocessing


## Dataset Summary
Here is a summary of the datasets used in the project:

    IMDB-WIKI
        Total Images: ~500,000
        Image/Class: ~10,000
        Features: Age / Gender labels

    MMA FACIAL EXPRESSION
        Total Images: ~128,000
        Image/Class: ~6,500
        Features: Compact images, Only frontal faces, RGB images

    UTKFace
        Total Images: ~20,000
        Image/Class: Unknown
        Features: Diverse images, Only frontal faces, Duplicate-free

    Real and Fake Face Detection
        Total Images: ~2,000
        Image/Class: ~1,000
        Features: High resolution, Only frontal faces, Duplicate-free

    Flickr-Faces-HQ Dataset (FFHQ)
        Total Images: 70,000
        Image/Class: ~7,000
        Features: High quality images, Only frontal faces, Duplicate-free

## Contents
1. Dependencies for running the Python script.
2. Python code for pre-processing, visualizing data, training, evaluating, and testing models.
3. Signed Originality form by all team members.
4. Structured Project report per guidelines.
5. Provenance information for datasets used.

## Data Cleaning Process for Facial Expression Recognition

The data cleaning process is a critical step in preparing the datasets for training the facial expression recognition model. It involves standardizing the images, reducing complexity, and augmenting the dataset for better generalization.

## Cleaning Techniques

### Resizing Images
All images are resized to a standard dimension to ensure consistency across the dataset.

```python
from keras.preprocessing.image import ImageDataGenerator

# Assuming images are in a directory 'data/train'
datagen = ImageDataGenerator(rescale=1./255)

# Standard dimensions for all images
standard_size = (150, 150)

# Generator will resize images automatically
train_generator = datagen.flow_from_directory(
    'data/train',
    target_size=standard_size,
    color_mode='grayscale', # Convert images to grayscale
    batch_size=32,
    class_mode='categorical'
)
```
### Grayscale Conversion

Images are converted to grayscale to focus on the important features without the distraction of color.
### Brightness Normalization

Uniform lighting conditions are applied to images to mitigate the effects of varying illumination.

```
# Additional configuration for ImageDataGenerator to adjust brightness

datagen = ImageDataGenerator(
    rescale=1./255,
    brightness_range=[0.8,1.2] # Adjust brightness
)
```
### Cropping

Images are cropped to remove background noise and focus on the face, the most important part for emotion detection.

# CNN Architecture , Training, & Evaluation

## Architecture

- Input: 48x48 RGB images.
- Convolutional Layers: Multiple layers with 4x4 kernels, batch normalization, leaky ReLU activation.
- Pooling Layers: Max pooling layers with 2x2 kernels.
- Fully Connected Layers: 2 fully connected layers with dropout and softmax activation.
- Parallel Processing: nn.DataParallel for multi-GPU support.
- Training Monitoring: Tracks validation set performance to optimize model parameters.

# Bias Analysis, Model refinement, & deep evaluation



## Steps for Running the Python File

### Prerequisites
- Python3
- Pip3

### Setup the Datasets
- Download and unzip the [Facial Expression Dataset](link-to-dataset).
- Download and unzip the [Classroom Activity Dataset](https://drive.google.com/drive/folders/15KX23UhhYKx6UGpm-GAEtIsPpweVRHJd?usp=drive_link) in the parent folder.

### Setup Virtual Environment
```bash
pip3 install --upgrade pip
pip3 install virtualenv
python3 -m venv venv
source ./venv/bin/activate
```

### Install Dependencies
```bash
pip3 install -r requirements.txt
```

### Execution Steps
- **Preprocess and Visualize:** `python3 preprocessor.py`
- **Train/Validate and Test Main Model:** `python3 cnn_training_early_stop.py; python3 cnn_testing.py`
- **Variants Training and Testing:** `python3 cnn_training_variant1.py; python3 cnn_training_variant2.py; python3 cnn_testing_variant_1.py; python3 cnn_testing_variant_2.py`
- **K-fold Cross Validation:** `python3 cnn_training_kfold.py`

### Expected Output
- Classification of images into respective classes.
- Display of training/testing dataset classification.
- Data visualizations using Matplotlib.
- Training over epochs with accuracy and loss metrics.
- Saved models under Model folder.
- K-fold analysis with training and validation metrics.

---

For further details on methodology, datasets, and findings, refer to the complete project reports.
