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
        self.nodeParam = None
        self.actionIns = Action()
        self.nodeGraph,self.nodeParam = self.decoder(code)
        self.nodeGenerator = ConvNode
        node = []
        assert len(self.nodeParam) == len(self.nodeGraph), 'nodeParam length {0} and nodeGraph length {1} is not match.'.format(
            len(self.nodeParam), len(self.nodeGraph))
        for i in range(len(self.nodeGraph)):
            Type, Kernel, Stride = self.nodeParam[i]
            # select the node type
            if Type == 0:
                self.nodeGenerator = ConvNode
            else:
                self.nodeGenerator = ConvNode
            if i == 0:
                node.append(self.nodeGenerator(
                    inChannel, outChannel, Kernel, Stride))
                continue
            node.append(self.nodeGenerator(outChannel*len(self.nodeGraph[i]),outChannel,Kernel,Stride))
        self.nodeList = nn.ModuleList(node)
            

    def decoder(self, code):
        node = []
        nodeGraph = dict()
        backbone, actionCode = code[0], code[1:]
        # build Backbone
        _, numberOfNode, nodeType = backbone[0], backbone[1], backbone[2]
        for i in range(numberOfNode):
            if len(nodeGraph) == 0:
                nodeGraph[0] = list()
                node.append((0, 3, 1))
                continue
            if i not in nodeGraph.keys():
                nodeGraph[i] = list()
            nodeGraph[i].append(i-1)
            # note the type of node
            node.append((0, 3, 1))
        # build topological structure
        actioncode_iterator = iter(actionCode)
        while actioncode_iterator.__length_hint__() != 0:
            # build graph
            From, Action, To = actioncode_iterator.__next__()
            To = To % len(nodeGraph)
            From = From % (To+1)
            if Action == self.actionIns.ADD_EDGE:
                if To == From:
                    continue
                if From not in nodeGraph[To]:
                    nodeGraph[To].append(From)
            elif Action == self.actionIns.ADD_NODE:
                newNode = len(nodeGraph)
                if To == From:
                    for nodeId in nodeGraph:
                        if To in nodeGraph[nodeId]:
                            nodeGraph[nodeId].append(newNode)
                            nodeGraph[nodeId].remove(To)
                    nodeGraph[newNode] = [From]
                else:
                    nodeGraph[To].append(len(nodeGraph))
                    nodeGraph[len(nodeGraph)] = [From]
                # build real nodeList
                Type, Kernel, Stride = actioncode_iterator.__next__()
                Type = Type % 2
                if Kernel % 2 == 0:
                    Kernel = 3
                else:
                    Kernel = 5
                if Stride % 2 == 0:
                    Stride = 1
                else:
                    Stride = 1
                node.append((Type, Kernel, Stride))
            else:
                raise Exception('Unknown action code : {0}'.format(Action))
        return nodeGraph,node

    def forward(self, x):
        return x


if __name__ == "__main__":
    import sys
    sys.path.append('./Model')
    from individual import SEEIndividual
    ind = SEEIndividual(3,2)
    ind.setDec([[0,5,1],
                [1,0,5],
                [1,1,3],[1,3,1],
                [1,1,3],[1,3,1],
                [6,1,4],[1,3,1],
                [7,0,5],
                [7,1,5],[1,3,1],
                [8,1,0],[1,3,1]])
    a = SEENetworkGenerator(ind.getDec(),3,32)
    print('hello')
