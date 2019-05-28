import torch
import torch.nn as nn
from copy import copy
from collections import OrderedDict

from actionInstruction import Action


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
                      kernel_size=kernelSize, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(outChannels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.model(x)

class PoolNode(nn.Module):
    '''
    Basic Pooling layer.
    This maybe is useless
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
    def __init__(self, code, inChannel, outChannel, repeat=None):
        super(SEENetworkGenerator, self).__init__()
        self.nodeList = nn.ModuleList()
        self.nodeGenerator = ConvNode
        self.node_graph = self.decoder(code)
        self.actionIns = Action()
        
    def decoder(self, code):
        node = []
        node_graph = dict()
        backbone, actionCode = code[0],code[1:]
        # build Backbone
        _,numberOfNode, nodeType = backbone[0],backbone[1],backbone[2]
        for i in range(numberOfNode-1):
            if i not in node_graph.keys():
                node_graph[i] = list()
            node_graph[i].append(i)
            node.append(1)
        # build topological structure
        actioncode_iterator = iter(actionCode)
        for i in range(int(len(actionCode)/2)):
            # build graph
            From, Action, To = actioncode_iterator.__next__()
            if Action == self.actionIns.ADD_EDGE:
                node_graph[To%len(node_graph)].append(Action%len(node_graph))
            elif Action == self.actionIns.ADD_NODE:
                node_graph[len(node_graph)] = [From]
                node_graph[to].append(len(node_graph))
            else:
                raise Exception('Unknown action code.')
                
    def getModle(self):
        pass
    
    def forward(self,x):
        pass
    
if __name__ == "__main__":
    a = SEENetworkGenerator('code')
    print('hello')