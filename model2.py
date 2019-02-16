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
        
        self.cv1x1 = nn.Conv2d(1, 256, kernel_size=1, stride=1, padding=0)
        
        self.cv3x3 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=0)  #output 222 x 222 x 1 =  49,284
        self.fc3x3 = nn.Linear(49284, 500)
        self.b3x3 = nn.BatchNorm1d(500)
        self.dp3x3 = nn.Dropout(p=0.3)
        
        self.cv5x5 = nn.Conv2d(256, 64, kernel_size=5, stride=1, padding=0) #output 220 x 220 x 1 = 48,400
        self.fc5x5 = nn.Linear(48400, 500)
        self.b5x5 = nn.BatchNorm1d(500)
        self.dp5x5 = nn.Dropout(p=0.3)
        
        
        self.avp4 = nn.AvgPool2d(4,4)
        self.cvavp4 = nn.Conv2d(55, 16, kernel_size=1, stride=1, padding=0) #output 55 x 55 x 16 = 48,400
        self.fcavp = nn.Linear(48400, 500)
        self.bavp = nn.BatchNorm1d(500)
        self.dpavp = nn.Dropout(p=0.3)
        
        
        self.fc2  = nn.Linear(1500, 1500)
        self.b2 = nn.BatchNorm1d(1500)
        self.dp2 = nn.Dropout(p=0.4)
          
        self.fc3  = nn.Linear(1500, 136)
        
        
        I.xavier_uniform(self.fc3x3.weight.data)
        I.xavier_uniform(self.fc5x5.weight.data)
        I.xavier_uniform(self.fcavp.weight.data)
        I.xavier_uniform(self.fc2.weight.data)
        I.xavier_uniform(self.fc3.weight.data)

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        a = self.cv1x1(x)
        
        x = F.elu(self.cv3x3(F.elu((a))))
        x = x.view(x.size(0), -1)
        x = self.dp3x3(F.elu(self.b3x3((self.fc3x3(x)))))
        
        y = F.elu(self.cv5x5(F.elu((a))))
        y = y.view(y.size(0), -1)
        y = self.dp5x5(F.elu(self.b3x3((self.fc5x5(y)))))
        
        z = F.elu(self.cvavp4(self.avp4(x)))
        z = z.view(z.size(0), -1)
        z = self.dpavp(F.elu(self.bavp(self.fcavp(z))))
        
        s = x + y + z
        fc2 = self.dp2(F.elu(self.b2(self.fc2(s))))
        
        fc3 = self.fc3(fc2)
        
       
        return fc3
