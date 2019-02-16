## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # self.conv1 = nn.Conv2d(1, 32, 5)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch 
        # normalization) to avoid overfitting
        
        
        #input image - 224 x 224 x 1  (grayscale)
        
        self.cv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=1)
        self.b1 = nn.BatchNorm2d(32)
        self.mp1 = nn.MaxPool2d(2,2)
        self.dp1 = nn.Dropout(p=0.1)
        
        self.cv2 = nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=1)
        self.b2 = nn.BatchNorm2d(64)
        self.mp2 = nn.MaxPool2d(2,2)
        self.dp2 = nn.Dropout(p=0.2)
        
        self.cv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.b3 = nn.BatchNorm2d(128)
        self.mp3 = nn.MaxPool2d(2,2)
        self.dp3 = nn.Dropout(p=0.3)
        
        self.cv4 = nn.Conv2d(128, 256, kernel_size=2, stride=1, padding=0)
        self.b4 = nn.BatchNorm2d(256)
        self.mp4 = nn.MaxPool2d(2,2)
        self.dp4 = nn.Dropout(p=0.4)
        
        self.cv5 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.b5 = nn.BatchNorm2d(512)
        self.mp5 = nn.MaxPool2d(2,2)
        self.dp5 = nn.Dropout(p=0.5)
           
        self.fc1 = nn.Linear(18432, 1000)   #input 6 x 6 x 256 = 18,432 
        self.b6 = nn.BatchNorm1d(1000)
        self.dp6 = nn.Dropout(p=0.5)
        
        self.fc2 = nn.Linear(1000, 1000)
        self.b7 = nn.BatchNorm1d(1000)
        self.dp7 = nn.Dropout(p=0.5)
        
        self.fc3 = nn.Linear(1000, 136)
        
        
        I.xavier_uniform(self.fc1.weight.data)
        I.xavier_uniform(self.fc2.weight.data)
        I.xavier_uniform(self.fc3.weight.data)

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        x = self.dp1(self.mp1(F.elu(self.b1(self.cv1(x)))))
        x = self.dp2(self.mp2(F.elu(self.b2(self.cv2(x)))))
        x = self.dp3(self.mp3(F.elu(self.b3(self.cv3(x)))))
        x = self.dp4(self.mp4(F.elu(self.b4(self.cv4(x)))))
        x = self.dp5(self.mp5(F.elu(self.b5(self.cv5(x)))))
        
        x = x.view(x.size(0), -1)
        
        x = self.dp6(F.elu(self.b6(self.fc1(x))))
        x = self.dp7(F.elu(self.b7(self.fc2(x))))
        x = self.fc3(x)
        
        return x
