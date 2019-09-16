import torch
import torch.nn as nn
from collections import OrderedDict
from actionInstruction import Action
from copy import deepcopy

import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

def apply_drop_path(x, drop_path_keep_prob, layer_id, layers, step, steps):
    layer_ratio = float(layer_id+1) / (layers)
    drop_path_keep_prob = 1.0 - layer_ratio * (1.0 - drop_path_keep_prob)
    step_ratio = float(step + 1) / steps
    drop_path_keep_prob = 1.0 - step_ratio * (1.0 - drop_path_keep_prob)
    if drop_path_keep_prob < 1.:
        mask = torch.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(drop_path_keep_prob).to(device)
        #x.div_(drop_path_keep_prob)
        #x.mul_(mask)
        x = x / drop_path_keep_prob * mask
    return x

# This code refers to NAO_pytorch
class SepConv(nn.Module):
    def __init__(self,C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.C_out = C_out
        self.C_in = C_in
        self.padding = padding
        
        self.relu1 = nn.ReLU(inplace=True)
        self.W1_depthwise = nn.Conv2d(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=self.padding, groups=self.C_in)
        self.W1_pointwise = nn.Conv2d(C_in, C_out, kernel_size=kernel_size, padding=0, groups=self.C_in)
        self.bn1 = nn.BatchNorm2d(C_out)

        self.relu2 = nn.ReLU(inplace=True)
        self.W2_depthwise = nn.Conv2d(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=self.padding, groups=self.C_in)
        self.W2_pointwise = nn.Conv2d(C_in, C_out, kernel_size=kernel_size, padding=0, groups=self.C_in)
        self.bn2 = nn.BatchNorm2d(C_out)
    
    def forward(self, x):
        x = self.relu1(x)
        x = self.W1_depthwise(x)
        x = self.W1_pointwise(x)
        x = self.bn1(x)

        x = self.relu2(x)
        x = self.W2_depthwise(x)
        x = self.W2_pointwise(x)
        x = self.bn2(x)

        return x

class ConvNode(nn.Module):
    def __init__(self,
                channels,
                kernelSize=3,
                stride=1,
                padding=1,
                dropPathKeepProb=None,
                layerId=0,
                layers=0,
                steps=0):
        super(ConvNode, self).__init__()
        self.channels = channels
        self.stride = stride 
        self.kernelSize = kernelSize
        self.stride = stride
        self.dropPathKeepProb = dropPathKeepProb 
        self.layerId = layerId
        self.layers = layers
        self.steps = steps
        self.op = SepConv(channels,channels,kernelSize,stride,padding)

    def forward(self, x, step):
        y = self.op(x)
        if self.dropPathKeepProb is not None:
            y = apply_drop_path(y, self.dropPathKeepProb,layer_id = self.layerId, layers= self.layers, step=step, steps=self.steps)
        return y

class AvgPool(nn.Module):
    def __init__(self, kernelSize, padding):
        super(AvgPool, self).__init__()
        self.op = nn.AvgPool2d(kernel_size=kernelSize, padding=padding)
    
    def forward(self, x):
        return self.op(x)

class MaxPool(nn.Module):
    def __init__(self, kernelSize, padding):
        super(AvgPool, self).__init__()
        self.op = nn.MaxPool2d(kernel_size=kernelSize, padding=padding)
    
    def forward(self, x):
        return self.op(x)


if __name__ == "__main__":
    model = ConvNode(1,3,1,1,0.9,0,1,100)
    inputs = torch.Tensor(np.random.rand(1,1, 12,12))
    model, inputs = model.to(device), inputs.to(device)
    y = model(inputs,1)
    print(y)