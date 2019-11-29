import sys
sys.path.append("./")
import logging
import numpy as np
from misc import utils
import Cell_Constructor
from copy import deepcopy
from actionInstruction import Action, Operations
from collections import OrderedDict
import torch.nn as nn
import torch

class Node_based_Network_cifar(nn.Module):
    def __init__(self, args, cell_type, code, classes, layers, channels, keep_prob, drop_path_keep_prob, use_aux_head, steps):
        super(Node_based_Network_cifar, self).__init__()
        self.args = args
        self.classes = classes
        self.layers = layers
        self.channels = channels
        self.keep_prob = keep_prob
        self.drop_path_keep_prob = drop_path_keep_prob
        self.use_aux_head = use_aux_head
        self.steps = steps
        self.normal_reduction_code = code

        # select the cell type
        if cell_type == 'node':
            self.Cell = Cell_Constructor.Node_Cell
        elif cell_type == 'block':
            self.Cell = Cell_Constructor.Block_Cell
        else:
            raise Exception("cell type {0} is undefined.".format(cell_type))

        self.reduction_layers = [self.layers, 2*self.layers +1]
        self.layers = self.layers*3
        self.multi_adds = 0

        if self.use_aux_head:
            self.aux_head_index = self.reduction_layers[-1]
        
        stem_multiplier = 3
        channels = stem_multiplier * self.channels
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        outs = [[32, 32, channels],[32, 32, channels]]
        self.multi_adds += 3 * 3 * 3 * channels * 32 * 32
        channels = self.channels
        self.cells = nn.ModuleList()

        for i in range(self.layers+2):
            if i not in self.reduction_layers:
                cell = self.Cell(code = self.normal_reduction_code[0], 
                                 prev_layers=outs, 
                                 channels=self.channels, 
                                 reduction=False, 
                                 layer_id=i,
                                 init_layers=self.layers,
                                 steps= self.steps,
                                 drop_path_keep_prob=self.drop_path_keep_prob)
            else:
                cell = self.Cell(code = self.normal_reduction_code[1], 
                                 prev_layers=outs, 
                                 channels=self.channels, 
                                 reduction=True, 
                                 layer_id=i,
                                 init_layers=self.layers,
                                 steps= self.steps,
                                 drop_path_keep_prob=self.drop_path_keep_prob)
            # CAUTION cell need multi_adds property
            self.multi_adds += cell.multi_adds
            self.cells.append(cell)
            outs = [outs[-1], cell.out_shape]

            if self.use_aux_head and i == self.aux_head_index:
                self.auxiliary_head = utils.AuxHeadCIFAR(outs[-1][-1], classes)
        
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(1 - self.keep_prob)
        self.classifier = nn.Linear(outs[-1][-1], classes)
        
        self.init_parameters()
    
    def forward(self, inputs, step=None):
        aux_logits = None
        s0 = s1 = self.stem(inputs)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, step)
            if self.use_aux_head and i == self.aux_head_index and self.training:
                aux_logits = self.auxiliary_head(s1)
        out = s1
        out = self.global_pooling(out)
        out = self.dropout(out)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, aux_logits
    
    def init_parameters(self):
        for w in self.parameters():
            if w.data.dim() >= 2:
                nn.init.kaiming_normal_(w.data)
    
    def toDot(self):
        final_graph = "digraph final{\n#cells\n#edges#inner_cells}\n"
        cell_temp = "cell_#id[shape=circle, color=pink, fontcolor=red, fontsize=10,label=#cell_type];\n"
        edge_temp = "cell_#id_pre -> cell_#id;\n"
        cells_list = "" + cell_temp.replace("#id", str(0)).replace("#cell_type", "CONV3x3")
        edges_list = ""
        cells_inner_list = [None,None]
        for i,cell in enumerate(self.cells,start=1):
            if cell.reduction:
                cells_list += cell_temp.replace("#id", str(i)).replace("#cell_type", "Reduction_Cell")
            else:
                cells_list += cell_temp.replace("#id", str(i)).replace("#cell_type", "Normall_Cell")
            # try:
            if cell.reduction and cells_inner_list[1] == None:
                cells_inner_list[1] = cell.toDot()
            elif not cell.reduction and cells_inner_list[0] == None:
                cells_inner_list[0] = cell.toDot()
            # except:
                # pass
            pre_pre_id, pre_id = i-2 if i > 1 else 0, i-1
            if pre_pre_id != 0:
                edges_list += edge_temp.replace("#id_pre", str(pre_pre_id)).replace("#id", str(i))
            edges_list += edge_temp.replace("#id_pre", str(pre_id)).replace("#id", str(i))
        final_graph = final_graph.replace("#cells", cells_list).replace("#edges", edges_list)
        cells_inner_list = set(cells_inner_list)
        return final_graph.replace("#inner_cells","\n".join(cells_inner_list))
    

if __name__ == "__main__":
    import sys
    sys.path.append('./Model')
    from individual import SEEIndividual

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ind = SEEIndividual(2, (2, 10, 3))
    model = Node_based_Network_cifar(
        args=None,
        cell_type='node',
        code= ind.getDec(),
        classes =10,
        layers=3,
        channels=32,
        keep_prob=0.8,
        drop_path_keep_prob=0.8,
        use_aux_head=True,
        steps=1
    )
    inputs = torch.Tensor(np.random.rand(32, 3, 32, 32))
    model, inputs = model.to(device), inputs.to(device)
    y,aux = model(inputs, 1)
    print(model.toDot())