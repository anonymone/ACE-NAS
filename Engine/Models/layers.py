import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvolutionLayer(nn.Module):
    def __init__(self, parameters):
        super(ConvolutionLayer, self).__init__()
        self.conv = nn.Conv2d(  in_channels=parameters['in_channels'],
                                out_channels= parameters['out_channels'],
                                kernel_size= parameters['kernel_size'], 
                                stride= parameters['stride'], 
                                padding = parameters['padding'])

        self.bn = nn.BatchNorm2d(num_features=parameters['out_channels'])
        self.af = parameters['active_function']
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x) 
        if self.af is None:
            return x
        else:
            return self.af(x)

class PoolingLayer(nn.Module):
    def __init__(self,parameters):
        super(PoolingLayer, self).__init__()
        if parameters['poolingLayerType']%2 == 0:
            self.pool = nn.MaxPool2d(kernel_size=parameters['kernel_size'],
                                     stride=parameters['stride'], 
                                     padding=parameters['padding'])
        else:
            self.pool = nn.MaxPool2d(kernel_size=parameters['kernel_size'],
                                     stride=parameters['stride'], 
                                     padding=parameters['padding'])

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

# Just for testing usage.
class linear(nn.Module):
    def __init__(self,inSize,channelSize):
        super(linear, self).__init__()
        self.inSize = inSize
        self.channelSize = channelSize
        self.fc1 = nn.Linear(inSize*inSize*channelSize, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = x.view(-1, self.inSize*self.inSize*self.channelSize)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

