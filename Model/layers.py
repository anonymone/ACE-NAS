import torch
import torch.nn as nn
from copy import deepcopy
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

class SEEPhase(nn.Module):
    def __init__(self, code, inChannel, outChannel, repeat=None):
        super(SEEPhase, self).__init__()
        self.nodeParam = None
        self.actionIns = Action()
        self.nodeGraph, self.nodeParam = self.decoder(code)
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
            node.append(self.nodeGenerator(
                outChannel*len(self.nodeGraph[i]), outChannel, Kernel, Stride))
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
            # Normalize the code
            To = To % len(nodeGraph)
            From = From % (To+1)
            # Check the graph for loops
            isloop, _ = SEEPhase.isLoop(nodeGraph, (From, To))
            if isloop:
                To, From = From, To

            Action = self.actionIns.ActionNormlize(Action)
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
        return nodeGraph, node

    @staticmethod
    def isLoop(graph, newEdge=None):
        graph = deepcopy(graph)
        topologicalStructure = []
        if newEdge is not None:
            a, b = newEdge
            # if a == b:
            #     return False,None
            if a not in graph[b]:
                graph[b].append(a)
        # find root
        for i in graph:
            F = False
            for j in graph.values():
                F = i in j
                if F:
                    break
            if not F:
                break
        if F:
            return True, None
        else:
            topologicalStructure.append(i)
        del graph[i]
        while len(graph) != 0:
            for i in graph:
                F = False
                for j in graph.values():
                    F = i in j
                    if F:
                        break
                if not F:
                    break
            if F:
                return True, None
            else:
                topologicalStructure.append(i)
            del graph[i]
        topologicalStructure.reverse()
        return False, topologicalStructure

    def forward(self, x):
        _, topoList = SEEPhase.isLoop(self.nodeGraph)
        outputs = [self.nodeList[topoList[0]](
            x)] + [None for _ in range(len(self.nodeGraph)-1)]
        for i in topoList[1:-1]:
            outputs[i] = self.nodeList[i](
                torch.cat([outputs[j] for j in self.nodeGraph[i]], dim=1))

        return self.nodeList[topoList[-1]](torch.cat([outputs[j] for j in self.nodeGraph[topoList[-1]]], dim=1))


class SEENetworkGenerator(nn.Module):
    def __init__(self, codeList, channelsList):
        super(SEENetworkGenerator, self).__init__()
        phases = []
        for code, (inChannel, outChannel) in zip(codeList, channelsList):
            phases.append(SEEPhase(code, inChannel, outChannel))


if __name__ == "__main__":
    import sys
    sys.path.append('./Model')
    from individual import SEEIndividual
    ind = SEEIndividual(3, 2)
    ind.setDec([[0, 5, 1],
                [0, 0, 4],
                [0, 1, 2], [1, 3, 1],
                [5, 5, 2],
                [0, 0, 4],
                [4, 1, 5], [1, 3, 1],
                [0, 1, 2], [1, 3, 1],
                [5, 1, 3], [1, 3, 1],
                [6, 0, 4],
                [6, 1, 4], [1, 3, 1],
                [7, 1, 0], [1, 3, 1]])
    model = SEEPhase(ind.getDec(), 3, 32)
    data = torch.randn(16, 3, 32, 32)
    out = model(torch.autograd.Variable(data))
    print(out)
    # test isLoop
    # a = {
    #     0 :[],
    #     1 :[0],
    #     2 :[1,0],
    #     3 :[2,4],
    #     4 :[0,1,2]
    # }
    # print(SEEPhase.isLoop(a))