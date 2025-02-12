import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        # this function defines the nn structure
        super(self, CNN).__init__()
        self.cnn_model = nn.Sequential(
            # conv layer 1
            # 1 input as each pixel is level of gray
            nn.Conv2d(in_channels = 1, out_channels=6, kernel_size=5),

            #Relu activation, more commonly used than tanh
            nn.ReLU(),

            # pool elements
            # might want to do kernel_size of 3
            nn.AvgPool2d(kernel_size = 2, stride = 6),

            # conv layer 2
            nn.Conv2d(in_channel=6, out_channels=16, kernel_size = 6),

             #Relu activation, more commonly used than tanh
            nn.ReLU(),

            # pool elements
            # might want to do kernel_size of 3
            nn.AvgPool2d(kernel_size = 2, stride = 5),
        )

        self.dense_model = nn.Sequential(
            # adjust in_features based on performance
            # first linear layer
            nn.Linear(in_features=576, out_features = 128),
            nn.ReLU(),
            # second
            nn.Linear(in_features=128, out_features = 64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features = 4),
            # softmax is already applied for cross entropy loss 
        )


    def forward(self, x):
        # this function defines how the information goes through the nn

        # send through cnn
        x = self.cnn_model(x)

        # flatten output
        x = x.view(x.size[0], -1)

        # send through dense 
        x = self.dense_model(x)

        # soft max is applied for us in cross entropy loss
        
        return x