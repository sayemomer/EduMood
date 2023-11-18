from torch import optim, nn
import torch.nn.functional as F


class MultiLayerFCNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.layer1 = nn.Conv2d(3, 32, 4, padding=1, stride=1)
        self.B1 = nn.BatchNorm2d(32)
        self.layer2 = nn.Conv2d(32, 32, 4, padding=1, stride=1)
        self.B2 = nn.BatchNorm2d(32)
        self.Maxpool = nn.MaxPool2d(2)
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

        # Flatten the tensor for the fully connected layer
        x = x.view(x.size(0), -1)
        return self.fc(x)
