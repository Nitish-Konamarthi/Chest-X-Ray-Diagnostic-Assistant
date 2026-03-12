import os
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

import torchvision

class DenseNet121(nn.Module):

    def __init__(self, classCount, isTrained):

        super(DenseNet121, self).__init__()

        # Use weights parameter to avoid deprecation warning
        if isTrained:
            try:
                from torchvision.models import DenseNet121_Weights
                self.densenet121 = torchvision.models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
            except (ImportError, TypeError):
                self.densenet121 = torchvision.models.densenet121(pretrained=True)
        else:
            self.densenet121 = torchvision.models.densenet121(weights=None)

        kernelCount = self.densenet121.classifier.in_features

        self.densenet121.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())

    def forward(self, x):
        x = self.densenet121(x)
        return x

class DenseNet169(nn.Module):
    
    def __init__(self, classCount, isTrained):
        
        super(DenseNet169, self).__init__()
        
        # Use weights parameter to avoid deprecation warning
        if isTrained:
            try:
                from torchvision.models import DenseNet169_Weights
                self.densenet169 = torchvision.models.densenet169(weights=DenseNet169_Weights.IMAGENET1K_V1)
            except (ImportError, TypeError):
                self.densenet169 = torchvision.models.densenet169(pretrained=True)
        else:
            self.densenet169 = torchvision.models.densenet169(weights=None)
        
        kernelCount = self.densenet169.classifier.in_features
        
        self.densenet169.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())
        
    def forward (self, x):
        x = self.densenet169(x)
        return x
    
class DenseNet201(nn.Module):
    
    def __init__ (self, classCount, isTrained):
        
        super(DenseNet201, self).__init__()
        
        # Use weights parameter to avoid deprecation warning
        if isTrained:
            try:
                from torchvision.models import DenseNet201_Weights
                self.densenet201 = torchvision.models.densenet201(weights=DenseNet201_Weights.IMAGENET1K_V1)
            except (ImportError, TypeError):
                self.densenet201 = torchvision.models.densenet201(pretrained=True)
        else:
            self.densenet201 = torchvision.models.densenet201(weights=None)
        
        kernelCount = self.densenet201.classifier.in_features
        
        self.densenet201.classifier = nn.Sequential(nn.Linear(kernelCount, classCount), nn.Sigmoid())
        
    def forward (self, x):
        x = self.densenet201(x)
        return x


        