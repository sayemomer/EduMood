import os
import sys
import cv2
import torch
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F


# Load the model
model_path = './model/best_model.pth'  # Change to your model path
input_size = 48  # The input size that your model expects

class MultiLayerFCNet(nn.Module):
    def __init__(self,input_size, hidden_size, output_size):
        super().__init__()

        self.layer1=nn.Conv2d(3,32,4,padding=1,stride=1)
        self.B1 = nn.BatchNorm2d(32)
        self.layer2 = nn.Conv2d(32, 32, 4, padding=1, stride=1)
        self.B2 = nn.BatchNorm2d(32)
        self.Maxpool=nn.MaxPool2d(2)
        self.layer3 = nn.Conv2d(32, 64, 4, padding=1, stride=1)
        self.B3 = nn.BatchNorm2d(64)
        self.layer4 = nn.Conv2d(64, 64, 4, padding=1, stride=1)
        self.B4 = nn.BatchNorm2d(64)
        self.dropout = nn.Dropout(0.5)
        
        # New layers
        self.layer5 = nn.Conv2d(64, 128, 4, padding=1, stride=1)
        self.B5 = nn.BatchNorm2d(128)
        self.layer6 = nn.Conv2d(128, 128, 4, padding=1, stride=1)
        self.B6 = nn.BatchNorm2d(128)
        self.layer7 = nn.Conv2d(128, 256, 4, padding=1, stride=1)
        self.B7 = nn.BatchNorm2d(256)
        self.layer8 = nn.Conv2d(256, 256, 4, padding=1, stride=1)
        self.B8 = nn.BatchNorm2d(256)
        
        # Calculate the size for the fully connected layer after additional max-pooling layers
        # Assuming two max-pooling operations in the existing layers
        self.fc_size = 256   # Now this is 256 * 3 * 3
        self.fc = nn.Linear(self.fc_size, output_size)

    def forward(self, x):
        # Pass through existing layers
        x = F.leaky_relu(self.B1(self.layer1(x)))
        x = self.Maxpool(F.leaky_relu(self.B2(self.layer2(x))))
        x = F.leaky_relu(self.B3(self.layer3(x)))
        x = self.Maxpool(F.leaky_relu(self.B4(self.layer4(x))))
        
        # Pass through new layers
        x = F.leaky_relu(self.B5(self.layer5(x)))
        x = F.leaky_relu(self.B6(self.layer6(x)))
        x = self.Maxpool(F.leaky_relu(self.B7(self.layer7(x))))
        x = self.Maxpool(F.leaky_relu(self.B8(self.layer8(x))))
        
        x = x.view(x.size(0), -1)  # Flatten the tensor for the fully connected layer
        return self.fc(x)

input_size = 3 * 48 * 48  # 1 channels, 48x48 image size
hidden_size = 50  # Number of hidden units
output_size = 4  # Number of output classes 4

model = MultiLayerFCNet(input_size,hidden_size,output_size)  # Instantiate your model (fill in args)

state_dict = torch.load('./model/best_model_stop.pth')
new_state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict)
model.eval()

# Define the transformation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5836, 0.4212, 0.3323],std=[0.2325, 0.1985, 0.1722])  # Your normalization params
])

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Suppress the warning by redirecting stderr
# devnull = open(os.devnull, 'w')
# sys.stderr = devnull

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess the image
    image = transform(frame).unsqueeze(0)  # Add batch dimension

    # Get model predictions
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    # Display the results
    cv2.imshow('Webcam', frame)

    #display the predicted label
    cv2.putText(frame, str(predicted.item()), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    class_names = ['Angry', 'Bored', 'Engaged', 'Neutral']  # Define the class names

    #display the predicted label with bigger font size
    cv2.putText(frame, class_names[predicted.item()], (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Webcam', frame)

    #print the predicted class
    print(class_names[predicted.item()])

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Restore stderr
cap.release()
cv2.destroyAllWindows()