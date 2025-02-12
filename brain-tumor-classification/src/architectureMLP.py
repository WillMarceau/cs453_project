import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

class MLP(nn.Module):
    def __init__(self, size, classes):
        # this function defines the nn structure
        super().__init__()

        # Hidden dense layer 1 
        # could try 1024 if underfitting or 256 if overfitting w/ dropout
        self.dense1 = nn.Linear(size, 512)

        # Hidden dense layer 2 reduce size from first
        self.dense2 = nn.Linear(512, 256)

        # Hidden dense layer 3 reduce size from second
        self.dense3 = nn.Linear(256, 128)

        # Output layer, map to # of classes
        self.dense4 = nn.Linear(128, classes)

        # add droput if overfitting to increase generalization
        #self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # this function defines how the information goes through the nn

        # flatten image 
        # -1 tells pytorch to figure out the shape
        # might not be needed not sure
        x = x.view(x.shape[0], -1)

        # use RelU function for non-linearlity 
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))

        # do a dropout here if overfitting 
        #x = self.dropout(F.relu(self.dense3(x)))
        x = F.relu(self.dense3(x))

        # No activation here, apply softmax later for classification
        x = self.dense4(x)

        return x