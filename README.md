
# Applied Artificial Intelligence and Classroom Activity Recognition 

![Project Banner](images/output.png)

## Overview
A Deep Learning Convolutional Neural Network (CNN) using PyTorch that analyses images of students in a classroom or online meeting setting and categorizes them into distinct states or activities.

1. **Facial Expression Recognition:**  Convolutional Neural Networks (CNN) for facial expression recognition. The dataset for this project includes images from a few publicly available datasets containing nu-
merous human facial expressions.

2. **Classroom Activity Recognition:** A Deep Learning CNN using PyTorch designed to analyze images of students in a classroom or online meeting setting, categorizing them into distinct states or activities.

## Contents
  - [Classification Labels](#classification-labels)

  - [Data Collection & Preprocessing](#data-collection--preprocessing)
    - [Dataset Summary](#dataset-summary)
    - [Images/Class Distribution](#imagesclass-distribution)
    - [Data Cleaning Process for Facial Expression Recognition](#data-cleaning-process-for-facial-expression-recognition)
      - [Labeling](#labeling)
      - [Resizing Images](#resizing-images)
      - [Grayscale Conversion](#grayscale-conversion)
      - [Brightness Normalization](#brightness-normalization)
      - [Cropping](#cropping)
    - [Class Distribution](#class-distribution-1)
    - [Sample Images](#sample-images)
    - [Pixel Intensity Distribution](#pixel-intensity-distribution)
  - [CNN Architecture , Training, & Evaluation](#cnn-architecture--training--evaluation)
    - [Architecture](#architecture)
  - [Bias Analysis, Model refinement, & deep evaluation](#bias-analysis-model-refinement--deep-evaluation)
    - [Performance Metrics](#performance-metrics)
    - [Variants comparison](#variants-comparison)
    - [Confusion Matrix Analysis](#confusion-matrix-analysis)
      - [Main Model](#main-model)
      - [Variants](#variants)
    - [Impact of Architectural Variations](#impact-of-architectural-variations)
    - [Bias Analysis](#bias-analysis)
      - [Bias detection result](#bias-detection-result)
      - [Bias Mitigation](#bias-mitigation)
      - [Bias detection result after mitigation](#bias-detection-result-after-mitigation)
    - [K-fold Cross Validation](#k-fold-cross-validation)
      - [Original Model](#original-model)
      - [K-fold Model](#k-fold-model)
      - [original vs k-fold](#original-vs-k-fold)
  - [Steps for Running the Python File](#steps-for-running-the-python-file)
    - [Prerequisites](#prerequisites)
    - [Setup the Datasets](#setup-the-datasets)
    - [Setup Virtual Environment](#setup-virtual-environment)
    - [Install Dependencies](#install-dependencies)
    - [Execution Steps](#execution-steps)
    - [Expected Output](#expected-output)

 - [Refecence to the original project](#refecence-to-the-original-project)
  - [Conclusion and Future Work](#conclusion-and-future-work)

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
        Source: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/

    MMA FACIAL EXPRESSION
        Total Images: ~128,000
        Image/Class: ~6,500
        Features: Compact images, Only frontal faces, RGB images
        Source: https://www.kaggle.com/datasets/mahmoudima/mma-facial-expression

    UTKFace
        Total Images: ~20,000
        Image/Class: Unknown
        Features: Diverse images, Only frontal faces, Duplicate-free
        Source: https://susanqq.github.io/UTKFace/

    Real and Fake Face Detection
        Total Images: ~2,000
        Image/Class: ~1,000
        Features: High resolution, Only frontal faces, Duplicate-free
        Source: https://www.kaggle.com/ciplab/real-and-fake-face-detection`

    Flickr-Faces-HQ Dataset (FFHQ)
        Total Images: 70,000
        Image/Class: ~7,000
        Features: High quality images, Only frontal faces, Duplicate-free
        Source:https://github.com/NVlabs/ffhq-dataset

## Images/Class Distribution


## Data Cleaning Process for Facial Expression Recognition

### Labeling

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

## Class Distribution

## Sample Images

## Pixel Intensity Distribution


# CNN Architecture , Training, & Evaluation

## Architecture

- Input: 48x48 RGB images.
- Convolutional Layers: Multiple layers with 4x4 kernels, batch normalization, leaky ReLU activation.
- Pooling Layers: Max pooling layers with 2x2 kernels.
- Fully Connected Layers: 2 fully connected layers with dropout and softmax activation.
- Parallel Processing: nn.DataParallel for multi-GPU support.
- Training Monitoring: Tracks validation set performance to optimize model parameters.

# Bias Analysis, Model refinement, & deep evaluation

## Performance Metrics

## Variants comparison

## Confusion Matrix Analysis

### Main Model
### Variants

## Impact of Architectural Variations

## Bias Analysis

### Bias detection result

### Bias Mitigation

### Bias detection result after mitigation


## K-fold Cross Validation

### Original Model

### K-fold Model

### original vs k-fold






# Steps for Running the Python File

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

# Refecence to the original project

For further details on methodology, datasets, and findings, refer to the complete project reports.

# Conclusion and Future Work
