import os
import random
import shutil

def split_files(source_folder, new_parent_folder, split_percentage=15):
    try:
        # Ensure the new parent folder exists
        if not os.path.exists(new_parent_folder):
            os.makedirs(new_parent_folder)

        # Get the list of all child folders in the source folder
        child_folders = [f for f in os.listdir(source_folder) if os.path.isdir(os.path.join(source_folder, f))]

        for child_folder in child_folders:
            # Create corresponding child folders in the new parent folder
            train_category_path = os.path.join(new_parent_folder, 'Train', child_folder)
            test_category_path = os.path.join(new_parent_folder, 'Test', child_folder)
            val_category_path = os.path.join(new_parent_folder, 'Validation', child_folder)

            # Make sure the necessary directories exist
            for folder_path in [train_category_path, test_category_path, val_category_path]:
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)

            # Get the list of all files in the current child folder
            all_files = os.listdir(os.path.join(source_folder, child_folder))
            total_files = len(all_files)

            # Calculate the number of files to move
            num_files_to_move_test = int(total_files * (split_percentage / 100.0))
            num_files_to_move_val = int(total_files * (split_percentage / 100.0))

            # Convert set to list before using random.sample
            files_to_move_test = random.sample(list(set(all_files)), min(num_files_to_move_test, len(all_files)))
            files_to_move_val = random.sample(list(set(all_files) - set(files_to_move_test)), min(num_files_to_move_val, len(all_files)))

            # Move selected files to the destination folders
            for file_name in files_to_move_test:
                source_path = os.path.join(source_folder, child_folder, file_name)
                destination_path = os.path.join(test_category_path, file_name)
                shutil.move(source_path, destination_path)

            for file_name in files_to_move_val:
                source_path = os.path.join(source_folder, child_folder, file_name)
                destination_path = os.path.join(val_category_path, file_name)
                shutil.move(source_path, destination_path)

            # The remaining files stay in the train folder
            for file_name in set(all_files) - set(files_to_move_test) - set(files_to_move_val):
                source_path = os.path.join(source_folder, child_folder, file_name)
                destination_path = os.path.join(train_category_path, file_name)
                shutil.move(source_path, destination_path)

        print("Data split successfully!")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    source_folder = "./dataset"
    destination_folder = "./dataset_split"
    split_percentage = 15

    split_files(source_folder, destination_folder, split_percentage)