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
        # if inChannels != outChannels:
        #     # this node is used to deal with the depth-wise concatenations.
        #     self.depthWiseNode = nn.Sequential(
        #         nn.Conv2d(in_channels=inChannels,out_channels=outChannels,kernel_size=1,bias=False),
        #         nn.BatchNorm2d(outChannels),
        #         nn.ReLU(inplace=True)
        #     )
        # else:
        #     self.depthWiseNode = Identity()
        # Calculate paading
        padding = int((kernelSize-1)/2)
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=inChannels, out_channels=outChannels,
                      kernel_size=kernelSize, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(outChannels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x = self .depthWiseNode(x)
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


class Identity(nn.Module):
    """
    Adding an identity allows us to keep things general in certain places.
    This code is inspired by NSGA-net
    """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class SEEPhase(nn.Module):
    '''
    :Param nodeParam, list, recive the parameters of each node from decoder
    :Param actionIns, class, action tag of each action
    :Param nodeGraph, dict, the topological information of network
    :Param nodeGenerator, to generate specific node.
    '''

    def __init__(self, code, inChannel, outChannel, repeat=None):
        super(SEEPhase, self).__init__()
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
            # 0 means to insert a node between two near node.
            if i == 0:
                node.append(self.nodeGenerator(
                    inChannel, outChannel, Kernel, Stride))
            else:
                node.append(self.nodeGenerator(
                    outChannel*len(self.nodeGraph[i]), outChannel, Kernel, Stride))
        self.nodeList = nn.ModuleList(node)
        # self.out = nn.Sequential(
        #     nn.BatchNorm2d(outChannel),
        #     nn.ReLU(inplace=True)
        # )

    def decoder(self, code):
        '''
        transform the code into specific network.
        '''
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
            Action, From, To = actioncode_iterator.__next__()
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
                    continue
                    # continue
                    # for nodeId in nodeGraph:
                    #     if To in nodeGraph[nodeId]:
                    #         nodeGraph[nodeId].append(newNode)
                    #         nodeGraph[nodeId].remove(To)
                    # nodeGraph[newNode] = [From]
                else:
                    nodeGraph[To].append(len(nodeGraph))
                    nodeGraph[len(nodeGraph)] = [From]
                # build real nodeList
                try:
                    Type, Kernel, Stride = actioncode_iterator.__next__()
                except:
                    Type, Kernel, Stride = From, Action, To
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
        '''
        to check if graph is including loops and return topological sort. if not.
        '''
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

    def toDot(self, tag=''):
        nodeTemplate = '#ID[shape=circle, color=pink, fontcolor=red, fontsize=10,label=#ID];\n'
        graphTemplate = "digraph {#nodeList#topology}"
        edgeTemplate = "#preID->#recentID;\n"
        nodeList = ''
        topology = ''
        for nodeID in self.nodeGraph:
            nodeList = nodeList + nodeTemplate.replace('#ID', tag+str(nodeID))
            for preNode in self.nodeGraph[nodeID]:
                topology = topology + \
                    edgeTemplate.replace(
                        '#preID', tag+str(preNode)).replace('#recentID', tag+str(nodeID))
        return graphTemplate.replace('#nodeList', nodeList).replace('#topology', topology)

    def forward(self, x):
        _, topoList = SEEPhase.isLoop(self.nodeGraph)
        outputs = [self.nodeList[topoList[0]](
            x)] + [None for _ in range(len(self.nodeGraph)-1)]
        for i in topoList[1:-1]:
            outputs[i] = self.nodeList[i](
                torch.cat([outputs[j] for j in self.nodeGraph[i]], dim=1))
        return self.nodeList[topoList[-1]](torch.cat([outputs[j] for j in self.nodeGraph[topoList[-1]]], dim=1))


class SEENetworkGenerator(nn.Module):
    def __init__(self, codeList, channelsList, out_features, data_shape, repeats=None):
        super(SEENetworkGenerator, self).__init__()
        self._repeats = repeats
        phases = []
        for code, (inChannel, outChannel) in zip(codeList, channelsList):
            phases.append(SEEPhase(code, inChannel, outChannel))
            if self._repeats is not None:
                for i in range(self._repeats-1):
                    phases.append(SEEPhase(code, outChannel, outChannel))
        phases = self.buildNetwork(phases)
        self.model = nn.Sequential(*phases)

        # After the evolved part of the network, we would like to do global average pooling and a linear layer.
        # However, we don't know the output size so we do some forward passes and observe the output sizes.
        # This code refers from  NSGA-NET https://github.com/ianwhale/nsga-net
        out = self.model(torch.autograd.Variable(
            torch.zeros(1, channelsList[0][0], *data_shape)))
        shape = out.data.shape
        self.gap = nn.AvgPool2d(kernel_size=(shape[-2], shape[-1]), stride=1)
        shape = self.gap(out).data.shape
        self.linear = nn.Linear(shape[1] * shape[2] * shape[3], out_features)
        # We accumulated some unwanted gradient information data with those forward passes.
        self.model.zero_grad()

    def buildNetwork(self, phases):
        """
        Build up the layers with transitions.
        :param phases: list of phases
        :return: list of layers (the model).
        """
        layers = []
        last_phase = phases.pop()
        for i in range(1,len(phases)+1):
            # for _ in range(repeat):
            #     layers.append(phase)
            layers.append(phases[i-1])
            if self._repeats is not None:
                if i% self._repeats != 0:
                    continue
            # TODO: Generalize this, or consider a new genome.
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        layers.append(last_phase)
        return layers

    def toDot(self):
        graphTemplate = "digraph SEE_Network{#phase#edge#classifier}"
        subgraphTemplate = "subgraph Phase#PhaseID {#graph}\n"
        edgeTemplate = "#preID->MaxPool#PoolID;\nMaxPool#PoolID->#recentID;\n"
        classifierTemplate = "#nodeID->GlobalAveragePooling"
        phase = []
        phaseedge = []
        model = self.model[0]
        _, order = model.isLoop(model.nodeGraph)
        subgraph = model.toDot(
            tag="phase"+str(0)).replace('digraph {', '').replace('}', '')
        phase.append(subgraphTemplate.replace(
            '#graph', subgraph).replace("#PhaseID", str(0)))
        for PhaseID in range(1, len(self.model)):
            model = self.model[PhaseID]
            if type(model) != SEEPhase:
                edge = edgeTemplate.replace("#PoolID", str(PhaseID))
                continue
            edge = edge.replace(
                "#preID", "phase{0}".format(PhaseID-2)+str(order[-1]))
            _, order = model.isLoop(model.nodeGraph)
            edge = edge.replace(
                "#recentID", "phase{0}".format(PhaseID)+str(order[0]))
            phaseedge.append(edge)
            subgraph = model.toDot(
                tag="phase"+str(PhaseID)).replace('digraph {', '').replace('}', '')
            phase.append(subgraphTemplate.replace(
                '#graph', subgraph).replace("#PhaseID", str(PhaseID)))
        return graphTemplate.replace("#phase", "".join(phase)).replace("#edge", "".join(phaseedge)).replace("#classifier", classifierTemplate.replace("#nodeID", "phase{0}".format(PhaseID)+str(order[-1])))

    def forward(self, x):
        '''
        Forward propagation.
        :param x: Variable, input to network.
        :return: Variable.
        '''
        x = self.gap(self.model(x))
        x = x.view(x.size(0), -1)
        return self.linear(x), None


if __name__ == "__main__":
    import sys
    sys.path.append('./Model')
    from individual import SEEIndividual
    ind = SEEIndividual(2, (4, 13, 3))
    initChannel = 12
    channels = [(3, initChannel),
                (initChannel, 2*initChannel),
                (2*initChannel, 4*initChannel)]
    model = SEENetworkGenerator(ind.getDec(), channels, 10, (32, 32))
    # data = torch.randn(16, 3, 32, 32)
    # out = model(torch.autograd.Variable(data))
    print(model.toDot())
    print("hello Layers.")
    # test isLoop
    # a = {
    #     0 :[],
    #     1 :[0],
    #     2 :[1,0],
    #     3 :[2,4],
    #     4 :[0,1,2]
    # }
    # print(SEEPhase.isLoop(a))
