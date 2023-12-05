from deepface import DeepFace
import os

# Path to the folder containing images
folder_path = "./Dataset/dataset-split/Train/Bored"

# List to store the results
results = []
total_file=0

# Iterate over each file in the folder
for file in os.listdir(folder_path):
    # if file.endswith(".png"):  # Check for PNG files, change or expand as needed
        file_path = os.path.join(folder_path, file)
        total_file+=1

        # Analyze the image
        objs = DeepFace.analyze(img_path=file_path, actions=[ 'gender'], enforce_detection=False)

        # Store or print the results
        # print(f"Image: {file}, Dominant Gender: {objs[0]['gender']}")
        results.append(objs[0]['dominant_gender'])

# Now results contains all the analysis data
print(len(results))
print(total_file)
print(results)

#print how many is man and women

man = 0
women = 0

for i in results:
    if i == "Man":
          man += 1
    else:
          women += 1

print(man)
print(women)