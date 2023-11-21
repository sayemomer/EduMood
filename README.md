# Part 1 & 2
Course project for COMP 6721 - A Deep Learning Convolutional Neural Network (CNN) using PyTorch that analyses images of students in a classroom or online meeting setting and categorizes them into distinct states or activities.

# Submission Contents

1. A file containing the dependencies for running the Python script 
2. Python code for pre-processing and visualizing the data
3. Python code for training, evaluating and testing the models
4. Originality form - signed by all team members
5. Project report - structured per the guidelines
6. Provenance information for the dataset used

# Steps for Running the Python File

### Prerequisites
-  Python3
-  Pip3 
### Setup the dataset

Download and unzip the [dataset](https://drive.google.com/drive/folders/15KX23UhhYKx6UGpm-GAEtIsPpweVRHJd?usp=drive_link) in the parent folder.

### Setup Virtual Environment

```bash
pip3 install --upgrade pip

# install virtualenv using pip3
pip3 install virtualenv

# venv folder will be created in the root directory
python3 -m venv venv

# activate the virtual env - for Unix/Linux
source ./venv/bin/activate
```

### Install Dependencies
```bash
pip3 install -r requirements.txt
```

### Execution - preprocess and visualize

```python
python3 preprocessor.py
```

### Expected Output
First, the images in the dataset will be classifed into 4 classes. The number of images classifed in the training and testing dataset will be displayed. Then, the data visualizations from Matplotlib will pop-up as images automatically. Finally, the program terminates gracefully.


### Execution - train/validate and test
```python
python3 cnn_training_main.py
python3 cnn_testing.py
```
### Expected Output
The Main Model will be trained over 50 epochs with the accuracy and loss indicated over each epoch. The model with the best accuracy will be saved under Model/

### Execution - variants train/validate and test
```python
python3 cnn_training_variant1.py
python3 cnn_training_variant2.py
python3 cnn_testing_variant_1.py
python3 cnn_testing_variant_2.py
```
### Expected Output
The variants will be trained over 50 epochs with the accuracy and loss indicated over each epoch. The best variant models are available under the Model folder
