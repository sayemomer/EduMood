#!/usr/bin/env python
# coding: utf-8

# In[12]:


import torch
import numpy as np

from torch import optim, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset


import torch
from torch.utils.data import random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as td
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def custom_loader(batch_size, shuffle_test=False, data_path='./Dataset/Train'):
    # Add the necessary transforms
    transform = transforms.Compose([
        transforms.Resize((48, 48)),  
        transforms.ToTensor(),
    ])

    # Load your dataset using ImageFolder
    master_dataset = datasets.ImageFolder(root=data_path, transform=transform)

    # Calculate the sizes of the splits
    total_size = len(master_dataset)
    train_size = int(0.85 * total_size)
    val_size = total_size - train_size

    # Use random_split to create datasets for training, testing, and validation
    train_dataset, val_dataset = random_split(master_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader



class MultiLayerFCNet(nn.Module):
    def __init__(self,input_size, hidden_size, output_size):
        super().__init__()

        #initializing 10 layers with kernel size 5 and padding as 2
        self.layer1 = nn.Conv2d(3, 32, 5, padding=2, stride=1)
        self.B1 = nn.BatchNorm2d(32)
        self.layer2 = nn.Conv2d(32, 32, 5, padding=2, stride=1)
        self.B2 = nn.BatchNorm2d(32)
        self.Maxpool = nn.MaxPool2d(2)
        
        self.layer3 = nn.Conv2d(32, 64, 5, padding=2, stride=1)
        self.B3 = nn.BatchNorm2d(64)
        self.layer4 = nn.Conv2d(64, 64, 5, padding=2, stride=1)
        self.B4 = nn.BatchNorm2d(64)
        
        self.layer5 = nn.Conv2d(64, 128, 5, padding=2, stride=1)
        self.B5 = nn.BatchNorm2d(128)
        self.layer6 = nn.Conv2d(128, 128, 5, padding=2, stride=1)
        self.B6 = nn.BatchNorm2d(128)
        
        self.layer7 = nn.Conv2d(128, 256, 5, padding=2, stride=1)
        self.B7 = nn.BatchNorm2d(256)
        self.layer8 = nn.Conv2d(256, 256, 5, padding=2, stride=1)
        self.B8 = nn.BatchNorm2d(256)
        self.layer9 = nn.Conv2d(256, 512, 5, padding=2, stride=1)
        self.B9 = nn.BatchNorm2d(512)
        self.layer10 = nn.Conv2d(512, 512, 5, padding=2, stride=1)
        self.B10 = nn.BatchNorm2d(512)
        self.Maxpool3 = nn.MaxPool2d(2)
        self.Maxpool4 = nn.MaxPool2d(2)
        
        self.fc_size = 512 * 3 * 3  # Adjusted based on the added pooling layers
        self.fc = nn.Linear(self.fc_size, output_size)

    def forward(self, x):
        # Pass through existing layers
        x = F.leaky_relu(self.B1(self.layer1(x)))
        x = F.leaky_relu(self.B2(self.layer2(x)))
        x = self.Maxpool(x)
        
        x = F.leaky_relu(self.B3(self.layer3(x)))
        x = F.leaky_relu(self.B4(self.layer4(x)))
        x = self.Maxpool(x)
        
        x = F.leaky_relu(self.B5(self.layer5(x)))
        x = F.leaky_relu(self.B6(self.layer6(x)))
        x = self.Maxpool3(x)
        x = F.leaky_relu(self.B7(self.layer7(x)))
        x = F.leaky_relu(self.B8(self.layer8(x)))
        x = self.Maxpool4(x)
        
        x = F.leaky_relu(self.B9(self.layer9(x)))
        x = F.leaky_relu(self.B10(self.layer10(x)))
    
        
        x = x.view(x.size(0), -1)  # Flatten the tensor for the fully connected layer
        return self.fc(x)

if __name__ == '__main__':

    batch_size = 64
    test_batch_size = 64
    input_size = 3 * 48 * 48  # 1 channels, 48x48 image size
    hidden_size = 50  # Number of hidden units
    output_size = 4  # Number of output classes 4
    num_epochs = 10
    train_loader, test_loader = custom_loader(batch_size, data_path='./dataset/Train')
  

    epochs = 50
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MultiLayerFCNet(input_size, hidden_size, output_size)
    model = nn.DataParallel(model)
    model.to(device)
  

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    BestACC=0.3
    for epoch in range(epochs):
        running_loss = 0
        for instances, labels in train_loader:
            optimizer.zero_grad()

            output = model(instances)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(running_loss / len(train_loader))

        model.eval()
        with torch.no_grad():
            allsamps=0
            rightPred=0

            for instances, labels in test_loader:

                output = model(instances)
                predictedClass=torch.max(output,1)
                allsamps+=output.size(0)
                rightPred+=(torch.max(output,1)[1]==labels).sum()


            ACC=rightPred/allsamps
            print("epoch=",epoch)
            print('Accuracy is=',ACC*100)
            #if the acc is greater than the best acc, save the model
            
            if ACC>BestACC:
                torch.save(model.state_dict(), './model/model_variant2.pth')
                BestACC=ACC
        model.train()


