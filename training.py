#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import numpy as np
from keras.preprocessing import image
from matplotlib import pyplot



# In[2]:


img_width,img_height =800,600
train_data_dir='dataset/train'
validation_data_dir='dataset/test'
batch_size=8


# In[3]:


if K.image_data_format() == 'channels_first' :
    input_shape = (3,img_width,img_height)
else:
    input_shape = (img_width,img_height,3)


# In[4]:


train_datagen = ImageDataGenerator(
     rescale=1.0/255,
    rotation_range=40,
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


# In[6]:


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


# In[ ]:




