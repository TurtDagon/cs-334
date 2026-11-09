'''
Challenge - Model
    Constructs a pytorch model for a convolutional neural network
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class Challenge(nn.Module):
    def __init__(self, block):
        super(Challenge, self).__init__()

    def forward(self, x):
        z = x
        return z
