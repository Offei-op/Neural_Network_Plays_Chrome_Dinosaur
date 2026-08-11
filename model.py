import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self,in_channels = 1,num_classes=3,channels = 32,final_pool=1):
        super().__init__()
        conv1_in_channnels = in_channels
        conv1_out_channels = channels
        conv2_in_channels = channels
        conv2_out_channels = channels * 2
        #Convolution layers
        self.conv1 = nn.Conv2d(in_channels=conv1_in_channnels,
                               out_channels=conv1_out_channels,
                               kernel_size=5,
                               stride=1)

        self.conv2 = nn.Conv2d(in_channels=conv2_in_channels,
                               out_channels=conv2_out_channels,
                               kernel_size=5,
                               stride=1)

        #Max=Pooling layers
        self.max1 = nn.MaxPool2d(kernel_size=2)
        self.max2 = nn.AdaptiveMaxPool2d((final_pool,final_pool))

        #Fully Connected layers
        self.fc1 = nn.Linear(conv2_out_channels*final_pool*final_pool,32)
        self.fc2 = nn.Linear(32,num_classes)

        #Activation Layer
        self.relu =nn.ReLU()

        #Regularisation
        self.bn1 = nn.BatchNorm2d(num_features=conv1_out_channels)
        self.bn2 = nn.BatchNorm2d(num_features=conv2_out_channels)


    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.max2(x)
        x = torch.flatten(x,start_dim =1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


