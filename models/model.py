import torch
import torch.nn as nn
from models.layers import *
from torchvision.ops import SqueezeExcitation


class Model(nn.Module):
    def __init__(self, nc=None, att=None, p=0.2): #nc = number of classes, att = attention module, p = probability of the units retained
        super().__init__()
        self.conv1 = Conv(1, 16, 3, att=att)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = Conv(16, 32, 3, att=att)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = Conv(32, 48, 3, att=att)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = Conv(48, 64, 3, att=att)
        self.fpn = FeaturePyramidBlock((64, 48), att=att)
        self.drop = nn.Dropout(p)
        self.fc = nn.Linear(2048, nc)


    def forward(self, x):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x1 = self.conv3(x)
        x = self.pool3(x1)
        x0 = self.conv4(x)
        x = self.fpn(x0,x1)
        x = self.drop(x)
        x = self.fc(x)
        return x