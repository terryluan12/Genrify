# mfcc model using resnet pretrained
from torchvision import models
import torch.nn as nn
import torch

class MFCC_Resnet(nn.Module):
    def __init__(self, num_classes=10):
        super(MFCC_Resnet, self).__init__()
        self.name = "MFCC_Resnet"
        self.num_classes = num_classes
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        return self.resnet(x)