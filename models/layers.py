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
        
# In this module init_gpu is a compromising method sending weights to GPU
# I need to fix it in the future work.
class MultiBranchsContainer(nn.Module):
    def __init__(self, branchs):
        super(MultiBranchsContainer,self).__init__()
        self.branch_num = len(branchs)
        for index in range(self.branch_num):
            # self.__dict__['branch{0}'.format(index)] = nn.Sequential(*branchs[index])
            exec('self.branch{0} = nn.Sequential(*branchs[{0}])'.format(index))

    # def init_gpu(self):
    #     for index in range(self.branch_num):
    #         self.__dict__['branch{0}'.format(index)].to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    
    def forward(self, x):
        # self.init_gpu()
        outputs = list()
        for index in range(self.branch_num):
            exec('outputs.append(self.branch{0}(x))'.format(index))
        # for index in range(self.branch_num):
        return torch.cat(outputs,1)

class FullConnectionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout =False):
        super(FullConnectionLayer, self).__init__()
        if dropout:
            model = [nn.Linear(in_features= in_features, out_features=out_features), nn.Dropout(), nn.ReLU(inplace=True)]
        else:
            model = [nn.Linear(in_features= in_features, out_features=out_features), nn.ReLU(inplace=True)]
        self.feature = nn.Sequential(*model)

    def forward(self, x):
        return self.feature(x)

