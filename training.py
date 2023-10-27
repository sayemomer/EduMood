# importing all the required libraries


from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import numpy as np
from keras.preprocessing import image
from matplotlib import pyplot



# standardizing all the images to fixed size and initializing the dataset existing directories


img_width,img_height =150,150
train_data_dir='dataset/train'
validation_data_dir='dataset/test'
batch_size=8


# checks the color channel of the given images and sets the input shape accordingly


if K.image_data_format() == 'channels_first' :
    input_shape = (3,img_width,img_height)
else:
    input_shape = (img_width,img_height,3)


# assigning all the preprocessing metrics on both the training and testing dataset


train_datagen = ImageDataGenerator(
     rescale=1.0/255,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    featurewise_center=True,
    featurewise_std_normalization=True,
    brightness_range=[0.5, 1.5] 
)

test_datagen = ImageDataGenerator(rescale=1. / 255)


# actual preprocessing happens


train_generator = train_datagen.flow_from_directory(
     train_data_dir,
     target_size=(img_width,img_height),
     batch_size=batch_size,
     class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
     validation_data_dir,
     target_size=(img_width,img_height),
     batch_size=batch_size,
     class_mode='categorical')







