#!/usr/bin/env python
# coding: utf-8

# In[19]:


import torch
import numpy as np
from torch import optim, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data import random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as td
import torchvision.transforms as transforms
import torchvision.datasets as datasets

#splits the entire dataset as training and testing and returns their loader variables  
def custom_loader(batch_size, shuffle_test=False, data_path='./Dataset/Train'):
     # Add the necessary transforms
    transform = transforms.Compose([
        transforms.Resize((48, 48)),  
        transforms.ToTensor(),
    ])
    master_dataset = datasets.ImageFolder(root=data_path, transform=transform)
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
        #initializing 4 layers with kernel size 3
        self.layer1=nn.Conv2d(3,32,3,padding=1,stride=1)
        self.B1 = nn.BatchNorm2d(32)
        self.layer2 = nn.Conv2d(32, 32, 3, padding=1, stride=1)
        self.B2 = nn.BatchNorm2d(32)
        self.Maxpool=nn.MaxPool2d(2)
        self.layer3 = nn.Conv2d(32, 64, 3, padding=1, stride=1)
        self.B3 = nn.BatchNorm2d(64)
        self.layer4 = nn.Conv2d(64, 64, 3, padding=1, stride=1)
        self.B4 = nn.BatchNorm2d(64)
        self.dropout = nn.Dropout(0.5) 
        self.fc_size = 64 * 12 * 12  
        self.fc = nn.Linear(self.fc_size, output_size)

    def forward(self, x):
        # Pass through existing layers
        x = F.leaky_relu(self.B1(self.layer1(x)))
        x = self.Maxpool(F.leaky_relu(self.B2(self.layer2(x))))
        x = F.leaky_relu(self.B3(self.layer3(x)))
        x = self.Maxpool(F.leaky_relu(self.B4(self.layer4(x))))
        # Flatten the tensor for the fully connected layer
        x = x.view(x.size(0), -1) 
        return self.fc(x)

if __name__ == '__main__':

    batch_size = 64
    test_batch_size = 64
    # 1 channels, 48x48 image size
    input_size = 3 * 48 * 48  
    # Number of hidden units
    hidden_size = 50  
    # Number of output classes 4
    output_size = 4  
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
    #running through epochs to train the model
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
            #saving the best accuracy variant
            if ACC>BestACC:
                torch.save(model.state_dict(), './model/model_variant1.pth')
                BestACC=ACC

        model.train()


