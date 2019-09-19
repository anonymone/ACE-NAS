import sys
sys.path.append("./")
import logging
import numpy as np
from misc import utils
from copy import deepcopy
from actionInstruction import Action, Operations
from collections import OrderedDict
import torch.nn as nn
import torch


device = "cuda" if torch.cuda.is_available() else "cpu"


def apply_drop_path(x, drop_path_keep_prob, layer_id, layers, step, steps):
    layer_ratio = float(layer_id+1) / (layers)
    drop_path_keep_prob = 1.0 - layer_ratio * (1.0 - drop_path_keep_prob)
    step_ratio = float(step + 1) / steps
    drop_path_keep_prob = 1.0 - step_ratio * (1.0 - drop_path_keep_prob)
    if drop_path_keep_prob < 1.:
        mask = torch.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(
            drop_path_keep_prob).to(device)
        # x.div_(drop_path_keep_prob)
        # x.mul_(mask)
        x = x / drop_path_keep_prob * mask
    return x


class Identity(nn.Module):
    """
    Adding an identity allows us to keep things general in certain places.
    """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

# This code refers to NAO_pytorch


class SepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.C_out = C_out
        self.C_in = C_in
        self.padding = padding

        self.relu1 = nn.ReLU(inplace=False)
        self.W1_depthwise = nn.Conv2d(
            C_in, C_out, kernel_size=kernel_size, stride=stride, padding=self.padding, groups=self.C_in)
        self.W1_pointwise = nn.Conv2d(
            C_in, C_out, kernel_size=1, padding=0, groups=self.C_in)
        self.bn1 = nn.BatchNorm2d(C_out)

        self.relu2 = nn.ReLU(inplace=False)
        self.W2_depthwise = nn.Conv2d(
            C_in, C_out, kernel_size=kernel_size, stride=1, padding=self.padding, groups=self.C_in)
        self.W2_pointwise = nn.Conv2d(
            C_in, C_out, kernel_size=1, padding=0, groups=self.C_in)
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


class SEECell(nn.Module):
    def __init__(self,
                 code,
                 channels,
                 reduction,
                 # [(Size1), (Size2), ...] in my search space, it is no more than 2.
                 preLayerSize,
                 layerId=1,
                 layers=1,
                 steps=1,
                 dropPathKeepProb=None):
        super(SEECell, self).__init__()
        self.channels = channels
        self.reduction = reduction
        self.calibrate_size = utils.MaybeCalibrateSize(preLayerSize, channels)
        self.ops = nn.ModuleList()
        self.actionIns = Action()
        self.nodeGraph = self.decoder(code)
        _, self.topoList = utils.isLoop(self.nodeGraph)
        self.preOps = nn.ModuleList(self.getPreOps())
        # self.layerId = layerId
        # self.layers = layers
        # self.steps = steps
        # self.dropPathKeepProb = dropPathKeepProb
        self.used = []
        wh = min([shape[0] for i, shape in enumerate(preLayerSize)])
        self.out_shape = [wh/2, wh/2, self.channels] if self.reduction else [wh, wh, self.channels] 

        # debug used
        # self.code = code

    def forward(self, x0, x1):
        # to adjust the size of feature maps.
        x0 = self.calibrate_size(x0, x1)
        outputs = [self.ops[self.topoList[0]](
            x0)] + [None for _ in range(len(self.nodeGraph)-1)]
        for i in self.topoList[1:-1]:
            outputs[i] = self.ops[i](
                self.preOps[i](torch.cat([outputs[j] for j in self.nodeGraph[i]], dim=1)))
        # [OPTION] may change it to concat.
        x = self.ops[self.topoList[-1]](self.preOps[self.topoList[-1]](
            torch.cat([outputs[j] for j in self.nodeGraph[self.topoList[-1]]], dim=1)))
        return x

    def getPreOps(self):
        conv1x1Nodes = [Identity() for _ in self.nodeGraph]
        for node, preNode in self.nodeGraph.items():
            if len(preNode) > 1:
                conv1x1Nodes[node] = nn.Conv2d(
                    len(preNode)*self.channels, self.channels, kernel_size=1, bias=False)
        return conv1x1Nodes

    def decoder(self, code):
        '''
        transform the code into specific network.
        '''
        node = []
        nodeGraph = dict()
        opsDecoder = Operations()
        backbone, actionCode = code[0], code[1:]
        actioncode_iterator = iter(actionCode)
        # Set stride
        strideID = 2 if self.reduction else 1

        # build Backbone
        _, numberOfNode, nodeType = backbone[0], backbone[1], backbone[2]
        for i in range(numberOfNode):
            if len(nodeGraph) == 0:
                nodeGraph[0] = list()
                actionType, kernelID, bias = actioncode_iterator.__next__()
                self.ops.append(opsDecoder.getOps(opID=actionType,
                                                  channels=self.channels,
                                                  kernelSizeID=kernelID,
                                                  strideID=strideID,
                                                  bias=bias))
                strideID = 1
                continue
            if i not in nodeGraph.keys():
                nodeGraph[i] = list()
            nodeGraph[i].append(i-1)
            # Add new node into backbone.
            actionType, kernelID, bias = actioncode_iterator.__next__()
            self.ops.append(opsDecoder.getOps(opID=actionType,
                                              channels=self.channels,
                                              kernelSizeID=kernelID,
                                              strideID=strideID,
                                              bias=bias))
        # build topological structure
        for Action, From, To in actioncode_iterator:
            # build graph
            # Normalize the code
            To = To % len(nodeGraph)
            From = From % (To+1)
            # Check the graph for loops
            isloop, _ = utils.isLoop(nodeGraph, (From, To))
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
                else:
                    nodeGraph[To].append(len(nodeGraph))
                    nodeGraph[len(nodeGraph)] = [From]
                # build real ops
                try:
                    # if no more code unit for modifying node
                    # use the last code unit.
                    actionType, kernelID, bias = actioncode_iterator.__next__()
                except:
                    actionType, kernelID, bias = From, Action, To
                self.ops.append(opsDecoder.getOps(opID=actionType,
                                                  channels=self.channels,
                                                  kernelSizeID=kernelID,
                                                  strideID=strideID,
                                                  bias=bias))
            else:
                raise Exception('Unknown action code : {0}'.format(Action))
        return nodeGraph
    
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


class SEEArchitecture(nn.Module):
    def __init__(self, args, classes, layers, channels, code, keepProb, dropPathKeepProb, useAuxHead, steps):
        super(SEEArchitecture,self).__init__()
        self.args = args
        self.classes = classes
        self.layers = layers
        self.channels = channels
        self.keepProb = keepProb
        self.dropPathKeepProb = dropPathKeepProb
        self.useAuxHead = useAuxHead
        self.steps = steps
        # Normal Cell code
        self.N_code = code[0]
        # Reduction Cell code
        self.R_code = code[1]

        self.reductionLayer = [self.layers, 2 * self.layers+1]
        if self.useAuxHead:
            self.auxHeadIndex = self.reductionLayer[-1]
        self.layers = self.layers*3
        # self.multiAdds = 0 # prepared for extention
        stem_multiplier = 3
        channels = stem_multiplier * self.channels
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        outs = [[32, 32, channels], [32, 32, channels]]
        # self.multi_adds += 3 * 3 * 3 * channels * 32 * 32
        channels = self.channels
        self.cells = nn.ModuleList()
        for i in range(self.layers+2):
            if i not in self.reductionLayer:
                cell = SEECell(
                    code=self.N_code,
                    channels=channels,
                    reduction=False,
                    # [(Size1), (Size2), ...] in my search space, it is no more than 2.
                    preLayerSize=outs,
                    layerId=i,
                    layers=self.layers+2,
                    steps=self.steps,
                    dropPathKeepProb=self.dropPathKeepProb)
            else:
                channels *= 2
                cell = SEECell(
                    code=self.N_code,
                    channels=channels,
                    reduction=True,
                    # [(Size1), (Size2), ...] in my search space, it is no more than 2.
                    preLayerSize=outs,
                    layerId=i,
                    layers=self.layers+2,
                    steps=self.steps,
                    dropPathKeepProb=self.dropPathKeepProb)
            # self.multi_adds += cell.multi_adds
            self.cells.append(cell)
            outs = [outs[-1], cell.out_shape]

            if self.useAuxHead and i == self.auxHeadIndex:
                self.auxiliary_head = utils.AuxHeadCIFAR(outs[-1][-1], classes)

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(1 - self.keepProb)
        self.classifier = nn.Linear(outs[-1][-1], classes)

        self.init_parameters()
    
    def init_parameters(self):
        for w in self.parameters():
            if w.data.dim() >= 2:
                nn.init.kaiming_normal_(w.data)

    def forward(self, input, step=None):
        aux_logits = None
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1)
            if self.useAuxHead and i == self.auxHeadIndex and self.training:
                aux_logits = self.auxiliary_head(s1)
        out = s1
        out = self.global_pooling(out)
        out = self.dropout(out)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, aux_logits
    
    def toDot(self):
        graphTemplate = "digraph SEE_Network{#phase#edge#classifier}"
        subgraphTemplate = "subgraph Phase#PhaseID {#graph}\n"
        edgeTemplate = "#preID->MaxPool#PoolID;\nMaxPool#PoolID->#recentID;\n"
        classifierTemplate = "#nodeID->GlobalAveragePooling"
        phase = []
        phaseedge = []
        model = self.cells[0]
        _, order = utils.isLoop(model.nodeGraph)
        subgraph = model.toDot(
            tag="phase"+str(0)).replace('digraph {', '').replace('}', '')
        phase.append(subgraphTemplate.replace(
            '#graph', subgraph).replace("#PhaseID", str(0)))
        for PhaseID in range(1, len(self.cells)):
            model = self.cells[PhaseID]
            if PhaseID == 1:
                edge = edgeTemplate.replace("#PoolID", str(PhaseID))
                continue
            edge = edge.replace(
                "#preID", "phase{0}".format(PhaseID-2)+str(order[-1]))
            _, order = utils.isLoop(model.nodeGraph)
            edge = edge.replace(
                "#recentID", "phase{0}".format(PhaseID)+str(order[0]))
            phaseedge.append(edge)
            subgraph = model.toDot(
                tag="phase"+str(PhaseID)).replace('digraph {', '').replace('}', '')
            phase.append(subgraphTemplate.replace(
                '#graph', subgraph).replace("#PhaseID", str(PhaseID)))
        return graphTemplate.replace("#phase", "".join(phase)).replace("#edge", "".join(phaseedge)).replace("#classifier", classifierTemplate.replace("#nodeID", "phase{0}".format(PhaseID)+str(order[-1])))


if __name__ == "__main__":
    import sys
    sys.path.append('./Model')
    from individual import SEEIndividual
    ind = SEEIndividual(2, (2, 15, 3))

    model = SEEArchitecture(
        args=None,
        classes=10,
        layers=2,
        code = ind.getDec(),
        channels=32,
        keepProb=1,
        dropPathKeepProb=1,
        useAuxHead=False,
        steps=1000
    )
    inputs = torch.Tensor(np.random.rand(1, 3, 32, 32))
    model, inputs = model.to(device), inputs.to(device)
    y,aux = model(inputs, 1)
    print(model.toDot())
    print(y.cpu().detach().numpy().shape)
