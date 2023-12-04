import shutil
import os
import runpy

def copy_files(source_folder, destination_folder):
    # Create the destination folder if it does not exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Copy all files from the source folder to the destination folder
    for file_name in os.listdir(source_folder):
        # Construct full file path
        source_file = os.path.join(source_folder, file_name)
        destination_file = os.path.join(destination_folder, file_name)
        # Copy the file if it is an image
        shutil.copy2(source_file, destination_file)



if __name__ == '__main__':
    # Define the source and destination folder paths

    list_of_subfolders = ['Gender/Man','Gender/Woman' , 'Age/Young', 'Age/Senior', 'Age/Mid']
    list_of_emotions = ['Angry', 'Bored', 'Engaged', 'Neutral']
    for subfolder in list_of_subfolders:
        for emotion in list_of_emotions:
            source_folder = './Dataset copy/Test/' + emotion + '/' + subfolder
            destination_folder = './Dataset/Test/' + emotion
            copy_files(source_folder, destination_folder)
        print("Testing for " + subfolder)
        runpy.run_path('Test/cnn_testing.py')
        #delete the Dataset/Test folder
        shutil.rmtree('./Dataset/Test')
            # run the cnn_testing.py file

