# importing all the required libraries


from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import numpy as np
from keras.preprocessing import image
from matplotlib import pyplot
import random
import scipy
import os


# standardizing all the images to fixed size and initializing the dataset existing directories


img_width,img_height =128,128
train_data_dir='Dataset/Train'
batch_size=8


# checks the color channel of the given images and sets the input shape accordingly

if K.image_data_format() == 'channels_first' :
    input_shape = (3,img_width,img_height)
else:
    input_shape = (img_width,img_height,3)


# assigning all the preprocessing metrics on both the training and testing dataset


train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    # width_shift_range=0.2,
    # height_shift_range=0.2,
    # shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    featurewise_center=True,
    featurewise_std_normalization=True,
    brightness_range=[0.5, 1.5]
)

# test_datagen = ImageDataGenerator(rescale=1.0 / 255)


# actual preprocessing happens


train_generator = train_datagen.flow_from_directory(
     train_data_dir,
     target_size=(img_width,img_height),
     batch_size=batch_size,
     class_mode='categorical',
     shuffle=False ,# Do not shuffle to keep labels aligned with images
    #  color_mode='grayscale'  # Convert images to grayscale
     )



# Initialize parameters
output_dir_train = 'dataset/preprocessed/train'

# Create output directories for each class
for class_name in train_generator.class_indices.keys():
    os.makedirs(os.path.join(output_dir_train, class_name), exist_ok=True)

# Save images
for i in range(len(train_generator.filenames)):
    # Get image and its class label
    img_path = train_generator.filenames[i]
    class_label = img_path.split('/')[0]

    # Load the image
    img = train_generator._get_batches_of_transformed_samples(np.array([i]))[0]

    # Save the image
    save_path = os.path.join(output_dir_train, class_label, os.path.basename(img_path))
    image.save_img(save_path, img[0])

print("Images have been saved by class in", output_dir_train)


# validation_generator = test_datagen.flow_from_directory(
#      validation_data_dir,
#      target_size=(img_width,img_height),
#      batch_size=batch_size,
#      class_mode='categorical',
#     #  color_mode='grayscale',  # Convert images to grayscale
#      )

# Initialize parameters
# output_dir_test = 'dataset/preprocessed/validation'

# # Create output directories for each class
# for class_name in validation_generator.class_indices.keys():
#     os.makedirs(os.path.join(output_dir_test, class_name), exist_ok=True)

# # Save images
# for i in range(len(validation_generator.filenames)):
#     # Get image and its class label
#     img_path = validation_generator.filenames[i]
#     class_label = img_path.split('/')[0]

#     # Load the image
#     img = validation_generator._get_batches_of_transformed_samples(np.array([i]))[0]

#     # Save the image
#     save_path = os.path.join(output_dir_test, class_label, os.path.basename(img_path))
#     image.save_img(save_path, img[0])

# print("Images have been saved by class in", output_dir_test)




# visualization of the distribution of processed images



# create a bar graph of the number of images per class in the training dataset

def plot_distribution(train_generator):
    # get the total number of images per class
    counts = [train_generator.classes.tolist().count(i) for i in range(train_generator.num_classes)]
    # get the class labels
    labels = list(train_generator.class_indices.keys())
    # create the figure
    fig = pyplot.figure(figsize=(10,8))
    pyplot.bar(labels, counts)
    pyplot.xticks(rotation=90)
    pyplot.show()

plot_distribution(train_generator)

# create a bar graph of the number of images per class in the validation dataset

# plot_distribution(validation_generator)


# Present a collection of 25 images in a 5 × 5 grid,
# with the entire grid sized to fit within the confines of a standard letter-sized page.
# Ensure that the images are randomly chosen from each class upon every code execution. 
# This visual representation aids in understanding the dataset’s content and helps identify any noticeable anomalies or potential mislabelings.


# randomly select 25 images from each training class

random_images = []

def plot_random_images(train_generator):
    # get the class labels
    labels = list(train_generator.class_indices.keys())
    # get the total number of images per class
    counts = [train_generator.classes.tolist().count(i) for i in range(train_generator.num_classes)]
    # create the figure
    fig = pyplot.figure(figsize=(10,8))
    # define number of rows and columns
    rows = 5
    columns = 5
    # plot images
    for i in range(1, rows*columns +1):
        # select a random image
        index = random.randint(0, len(train_generator.filenames))
        
        random_images.append(train_generator.filepaths[index])

        # get the image path
        path = train_generator.filepaths[index]
        # read the image data and resize to the required size
        img = image.load_img(path, target_size=(img_width, img_height))
        # convert the image to a numpy array
        img = image.img_to_array(img)
        # display the image
        fig.add_subplot(rows, columns, i)
        pyplot.imshow(img.astype(np.uint8))
        pyplot.axis('off')
    # show the figure
    pyplot.show()

plot_random_images(train_generator)


#for the same random images, plot a histogram showing distribution of pixel densities 
# this can provide insight into variations in lighting conditions among the images 
# for RBG images , provide intensity distribution of each colour in separate plots

# plot a histogram showing distribution of pixel densities

#show the first random image
pyplot.imshow(image.load_img(random_images[0]))
#show the details of the image

print(image.load_img(random_images[0]).size)
print(image.load_img(random_images[0]).format)
print(image.load_img(random_images[0]).mode)


# plot pixel intensity distribution for the random_images

def plot_pixel_intensity_distribution(random_images):
    # create the figure
    fig = pyplot.figure(figsize=(10,8))
    # define number of rows and columns
    rows = 5
    columns = 5
    # plot images
    for i in range(1, rows*columns +1):
        # get the image path
        path = random_images[i-1]
        # read the image data and resize to the required size
        img = image.load_img(path, target_size=(img_width, img_height))
        # convert the image to a numpy array
        img = image.img_to_array(img)
        # plot pixel intensity distribution for each color channel
        fig.add_subplot(rows, columns, i)
        pyplot.hist(img[:,:,0].flatten(), bins=256, color='red', alpha=0.5)
        pyplot.hist(img[:,:,1].flatten(), bins=256, color='green', alpha=0.5)
        pyplot.hist(img[:,:,2].flatten(), bins=256, color='blue', alpha=0.5)
    # show the figure
    pyplot.show()

plot_pixel_intensity_distribution(random_images)
