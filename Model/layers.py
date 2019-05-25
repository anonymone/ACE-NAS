import torch
import torch.nn as nn
from copy import copy
from collections import OrderedDict


# Basic Node Type
class ConvNode(nn.Module):
    '''
    Basic Convolution layer with the same input and output size.
    '''

    def __init__(self, inChannels, outChannels, kernelSize=3, stride=1, bias=True):
        '''
        :Param inChannels: int, the number of input channels.
        :Param outChannels: int, the number of output channels.
        '''
        super(ConvNode, self).__init__()
        # Calculate paading
        padding = int((kernelSize-1)/2)
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=inChannels, out_channels=outChannels,
                      kernel_size=kernelSize, stride=stride, padding=padding, bias=bias)
            nn.BatchNorm2d(outChannels)
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.model(x)

class PoolNode(nn.Module):
    '''
    Basic Pooling layer.
    This maybe useless
    '''
    def __init__(self, kernelSize=2, stride=1):
        '''
        :Param kernelSize: the size of sampling size.
        :Param stride: the length of stride.
        '''
        super(PoolNode, self).__init__()
        self.model = nn.Sequential(
            nn.MaxPool2d(kernel_size=kernelSize, stride=stride)
        )
    
    def forward(self, x):
        return self.model(x)

class SEENetworkGenerator(nn.Module):
    def __init__(self, code, repeat=None):
        super(SEENetworkGenerator, self).__init__()
    
    @staticmethod
    def decoder(self, code):
        pass
    
    def getModle(self):
        pass
    
    def forward(self,x):
        pass