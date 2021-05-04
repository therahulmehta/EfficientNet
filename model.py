#import torch, torch graph blocks, and python math
import torch 
import torch.nn as nn
import math

# foundational model [expandratio, channels, repeats, stride, kernal_size]
# taken verbatim from paper MobileNet  
model_primatives = [
    [1, 16, 1, 1, 3],
    [6, 24, 2, 2, 3],
    [6, 40, 2, 2, 5],
    [6, 80, 3, 2, 3],
    [6, 112, 3, 1, 5],
    [6, 192, 4, 2, 5],
    [6, 320, 1, 1, 3],
]

# synthetic values for selecting reasource avaliblity for model scaling
# depth = alpha ** phi
# structure: (phi, res, drop rate) 
phi_values = {
    "b0": (0, 224, 0.2), 
    "b1": (0.5, 240, 0.2),
    "b2": (1, 260, 0.3),
    "b3": (2, 300, 0.3),
    "b4": (3, 380, 0.4),
    "b5": (4, 456, 0.4),
    "b6": (5, 528, 0.5),
    "b7": (6, 600, 0.5),
}


class CNNBlock(nn.Module):
    #depth-wise conv configuration, groups retains the # channels of the input. 
    def __init__(self, in_channels, out_channels, kernal_size, stride, padding, groups=1):
        super(CNNBlock, self).__init__()
        self.cnn = nn.Conv2d(
            in_channels,
            out_channels,
            kernal_size, 
            stride,
            padding, 
            groups=groups,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU() 

    def forward(self, x): 
        return self.silu(self.bn(self.cnn(x))) 

# Hu et al. 2018 -- Squeeze and Excitation
# Used in between the inverted residual block layers to select high priority channels
class SqueezeExcitation(nn.Module): 
    def __init__(self, in_channels, reduced_dim): 
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), 
            nn.Conv2d(in_channels, reduced_dim, 1),
            nn.SiLU(),
            nn.Conv2d(reduced_dim, in_channels, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x): 
        return x*self.se(x)
     

# MobileNetV2
class InvertedResidualBlock(nn.Module):
    pass

class EfficientNet(nn.Module):
    pass