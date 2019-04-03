import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvolutionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, active_function = F.relu):
        super(ConvolutionLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,out_channels= out_channels,kernel_size=kernel_size, stride=stride, padding = padding)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.af = active_function
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x) 
        if self.af is None:
            return x
        else:
            return self.af(x)

class PoolingLayer(nn.Module):
    def __init__(self, kernel_size, stride, padding, pooling_type= 0):
        super(PoolingLayer, self).__init__()
        if pooling_type%2 == 0:
            self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        return self.pool(x)

class SkipContainer(nn.Module):
    def __init__(self, stack_layers):
        super(SkipContainer, self).__init__()
        self.stack_layers = nn.Sequential(*stack_layers)
        self.af = nn.ReLU(inplace=True)
    
    def forward(self,x ):
        Fx = self.stack_layers(x)
        return self.af(torch.add(Fx,x))
        
class MultiBranchsContainer(nn.Module):
    def __init__(self, branchs):
        super(MultiBranchsContainer,self).__init__()
        for index, branch in zip(range(len(branchs)),branchs):
            self.branchs['branch{0}'.format(index)] = nn.Sequential(*branch)
    
    def forward(self, x):
        outputs = list()
        for branch in self.branchs:
            outputs.append(branch(x))
        return torch.cat(outputs,1)

class FullConnectionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(FullConnectionLayer, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(in_features= in_features, out_features=out_features),
            nn.Dropout(),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.feature(x)

