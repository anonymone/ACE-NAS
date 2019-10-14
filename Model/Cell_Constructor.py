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
import torch.nn.functional as F

class Conv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(Conv, self).__init__()
        if isinstance(kernel_size, int):
            self.ops = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(C_out, affine=affine)
            )
        else:
            assert isinstance(kernel_size, tuple)
            k1, k2 = kernel_size[0], kernel_size[1]
            self.ops = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(C_out, C_out, (k1, k2), stride=(1, stride), padding=padding[0], bias=False),
                nn.BatchNorm2d(C_out, affine=affine),
                nn.ReLU(inplace=True),
                nn.Conv2d(C_out, C_out, (k2, k1), stride=(stride, 1), padding=padding[1], bias=False),
                nn.BatchNorm2d(C_out, affine=affine),
            )

    def forward(self, x, bn_train=False):
        x = self.ops(x)
        return x

class SepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=INPLACE),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=INPLACE),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )
    
    def forward(self, x):
        return self.op(x)

# use small search space
OPERATIONS = {
    0: SepConv, # 3x3
    1: SepConv, # 5x5
    2: nn.AvgPool2d, # 3x3
    3: nn.MaxPool2d, # 3x3
    # 4: Identity,
}

# use large search space
OPERATIONS_large = {
    # 0: Identity,
    4: Conv, # 1x1
    5: Conv, # 3x3
    6: Conv, # 1x3 + 3x1
    7: Conv, # 1x7 + 7x1
    8: nn.MaxPool2d, # 2x2
    9: nn.MaxPool2d, # 3x3
    10:nn.MaxPool2d, # 5x5
    11:nn.AvgPool2d, # 2x2
    12:nn.AvgPool2d, # 3x3
    13:nn.AvgPool2d, # 5x5
}
# action defined 
ACTION = {
    0: 'change_connection_A_B',
    1: 'add_node_C',
    2: 'clone_node_A',
    3: 'substitute_node_B_for_type', # A will be changed to type
    # 4: 'deldete_connection_A_B'
}


class code_parser:
    def __init__(self, node_search_space=OPERATIONS_large, action_search_space=ACTION):
        self.op_search_space = node_search_space
        self.act_search_space = action_search_space
        self.token_start_point = 4 if node_search_space == OPERATIONS_large else 0

    def get_op_token(self, code_type):
        return code_type% len(self.op_search_space) + self.token_start_point
    
    def get_action_token(self, code_type):
        return code_type% len(self.act_search_space)

    def get_op(self, code_type):
        code_type = code_type% len(self.op_search_space) + self.token_start_point
        return self.op_search_space[code_type]
    
    def get_action(self, code_type):
        code_type = code_type% len(self.act_search_space)
        return self.act_search_space[code_type]

class Node(nn.Module):
    def __init__(self, node_type, input_shape, channels, stride, drop_path_keep_prob=None, 
                layer_id=0, layers=0, steps=0):
        # for i, token in enumerate(OPERATIONS_large):
        #     stride = 2 if self.reduction and i in [0,1] else 1 
        super(Node, self).__init__()
        self.node_type = node_type
        self.channels = channels 
        self.stride = stride
        self.layer_id = layer_id
        self.layers = layers
        self.steps =  steps
        self.drop_path_keep_prob = drop_path_keep_prob
        self.multi_adds = 0

        if node_type == 0:
            pass
        elif node_type == 1:
            pass
        elif node_type ==2:
            pass
        elif node_type ==3:
            pass
        elif node_type == 4:
            # Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
            self.op = OPERATIONS_large[node_type](self.channels, self.channels, 1, stride, 0)
            input_shape = [input_shape[0] // stride, input_shape[1] // stride, channels]
            self.multi_adds += 1 * 1 * channels * channels * (input_shape[0] // stride) * (input_shape[1] // stride)
        elif node_type == 5:
            self.op = OPERATIONS_large[node_type](self.channels, self.channels, 3, stride, 1)
            input_shape = [input_shape[0] // stride, input_shape[1] // stride, channels]
            self.multi_adds += 3 * 3 * channels * channels * (input_shape[0] // stride) * (input_shape[1] // stride)
        elif node_type == 6:
            self.op = OPERATIONS_large[node_type](self.channels, self.channels, (1,3), stride, ((0,1),(1,0)))
            input_shape = [input_shape[0] // stride, input_shape[1] // stride, channels]
            self.multi_adds += 2 * 1 * 3 * channels * channels * (input_shape[0] // stride) * (input_shape[1] // stride)
        elif node_type == 7:
            self.op = OPERATIONS_large[node_type](self.channels, self.channels, (1,7), stride, ((0,3),(3,0)))
            input_shape = [input_shape[0] // stride, input_shape[1] // stride, channels]
            self.multi_adds += 2 * 1 * 7 * channels * channels * (input_shape[0] // stride) * (input_shape[1] // stride)
        elif node_type == 8:
            self.op = OPERATIONS_large[node_type](2, stride=stride, padding = 0)
            input_shape = [input_shape[0] // stride, input_shape[1] // stride, channels]
        elif node_type == 9:
            self.op = OPERATIONS_large[node_type](3, stride=stride, padding = 1)
            input_shape = [input_shape[0] // stride, input_shape[1] // stride, channels]
        elif node_type == 10:
            self.op = OPERATIONS_large[node_type](5, stride=stride, padding = 2)
            input_shape = [input_shape[0] // stride, input_shape[1] // stride, channels]
        elif node_type == 11:
            self.op = OPERATIONS_large[node_type](2, stride=stride, padding = 0)
            input_shape = [input_shape[0] // stride, input_shape[1] // stride, channels]
        elif node_type == 12:
            self.op = OPERATIONS_large[node_type](3, stride=stride, padding = 1)
            input_shape = [input_shape[0] // stride, input_shape[1] // stride, channels]
        elif node_type == 13:
            self.op = OPERATIONS_large[node_type](5, stride=stride, padding = 2)
            input_shape = [input_shape[0] // stride, input_shape[1] // stride, channels]
        self.out_shape = list(input_shape)

    def forward(self,x, step):
        if self.node_type in [8,11]:
            x = F.pad(x, [0, 1, 0, 1])
        x = self.op(x)
        if self.drop_path_keep_prob is not None and self.training:
            x = utils.apply_drop_path(x, self.drop_path_keep_prob, self.layer_id, self.layers, step, self.steps)
        return x


class Node_Cell(nn.Module):
    def __init__(self, code, prev_layers, channels, reduction, layer_id, init_layers, steps, drop_path_keep_prob=None):
        super(Node_Cell, self).__init__()
        self.layer_id = layer_id
        self.reduction = reduction
        self.code = code 
        self.init_layers = init_layers
        self.channels = channels
        self.steps =  steps
        self.drop_path_keep_prob = drop_path_keep_prob
        self.ops = nn.ModuleDict()
        self.parser = code_parser()
        self.multi_adds = 0
        # self.input_shape = list(input_shape)
        
        # calibrate size 
        prev_layers = [list(prev_layers[0]), list(prev_layers[1])]
        self.maybe_calibrate_size = utils.MaybeCalibrateSize(prev_layers, channels)
        # prev_layers = self.maybe_calibrate_size.out_shape[0]
        self.multi_adds += self.maybe_calibrate_size.multi_adds
        self.dependence_graph = self.decoder(code)

    def decoder(self, code):
        node_graph = {0:[]}
        init_nodes, action_code = code[0], code[1:]
        node_type_tokens = {0:'MaybeCalibrateSize'}

        # add init node 
        for i, node_type in zip(range(1, len(init_nodes)+1),init_nodes):
            node_graph[i] = [0]
            node_type_tokens[i] = self.parser.get_op_token(node_type)
        # build node graph
        for act_token, param1, param2 in action_code:
            action = self.parser.get_action(act_token)
            if action == 'change_connection_A_B':
                node_id1, node_id2 = param1%(len(node_graph)-1)+1, param2%(len(node_graph)-1)+1
                if node_id1 == node_id2:
                    node_id2 = node_id2 - 1
                is_loop,_  = utils.isLoop(node_graph, (node_id1, node_id2), (0, node_id2))
                if is_loop:
                    node_id1, node_id2 = node_id2, node_id1
                if node_id1 in node_graph[node_id2]:
                    # if edge exist, remove it.
                    node_graph[node_id2].remove(node_id1)
                    if len(node_graph[node_id2]) == 0:
                        node_graph[node_id2].append(0)
                else:
                    if 0 in node_graph[node_id2]:
                        node_graph[node_id2].remove(0)
                    node_graph[node_id2].append(node_id1)
            elif action == 'add_node_C':
                node_type, pre_node = param1, param2%len(node_graph)
                node_graph[len(node_graph)] = [pre_node]
                node_type_tokens[len(node_type_tokens)] = self.parser.get_op_token(node_type)
            elif action == 'clone_node_A':
                # 3 is used to control the ratio of remain_edges
                node_id, remain_edges = param1%(len(node_graph)-1)+1, True if param2 >= 3 else False
                if remain_edges:
                    new_node_id = len(node_graph)
                    node_graph[new_node_id] = node_graph[node_id].copy()
                    for i, v in node_graph.items():
                        if node_id in v:
                            node_graph[i].append(new_node_id)
                    node_type_tokens[len(node_type_tokens)] = node_type_tokens[node_id]
                else:
                    node_graph[len(node_graph)] = [0]
                    node_type_tokens[len(node_type_tokens)] = node_type_tokens[node_id]
            elif action == 'substitute_node_B_for_type':
                node_id1, node_type = param1%(len(node_graph)) + 1, param2
                node_type_tokens[node_id1] = self.parser.get_op_token(node_type)
            else:
                raise Exception('Encountered the unknown action token: {0}'.format(action))
        self.node_type_tokens = node_type_tokens
        # static the concat node
        nodes_all = list(node_graph.keys())
        nodes_used = []
        self.nodes_reduction = []
        def inline_remove(node_all, node_used):
            for i in node_used:
                node_all.remove(i)
            return node_all
        for node_id,nodes in node_graph.items():
            nodes_used += nodes
            if 0 in nodes:
                self.nodes_reduction.append(node_id)
        self.nodes_used = list(set(nodes_used))
        self.concat_nodes = inline_remove(nodes_all, self.nodes_used)            
        # build node instant
        _, node_order = utils.isLoop(node_graph)
        self.node_order = node_order
        prev_layers = dict()
        for node_id in self.node_order:
            if node_id == 0:
                input_shape = self.maybe_calibrate_size.out_shape[0]
                prev_layers[node_id] = input_shape
                continue
            stride = 2 if self.reduction and node_id in self.nodes_reduction else 1
            input_shape = min([prev_layers[i] for i in node_graph[node_id]])
            node = Node(self.node_type_tokens[node_id], input_shape,self.channels, stride = stride, 
                                                          drop_path_keep_prob=self.drop_path_keep_prob, 
                                                          layer_id=self.layer_id,
                                                          layers=self.init_layers,
                                                          steps=self.steps)
            self.ops[str(node_id)] = node
            self.multi_adds += node.multi_adds
            prev_layers[node_id] = node.out_shape

        out_hw = min([shape[0] for i, shape in prev_layers.items() if i in self.concat_nodes])
        self.final_combine = utils.FinalCombine(prev_layers, out_hw, self.channels, self.concat_nodes)
        self.out_shape = [out_hw, out_hw, self.channels * len(self.concat_nodes)]
        self.multi_adds += self.final_combine.multi_adds
        return node_graph
    
    def forward(self, s0, s1, step):
        s0 = self.maybe_calibrate_size(s0, s1)
        states = [s0] + [None for _ in self.node_order[1:]]
        for i in self.node_order[1:]:
            inputs = 0
            for pre_node in self.dependence_graph[i]:
                inputs += states[pre_node]
            x = self.ops[str(i)](inputs, step)
            states[i] = x
        return self.final_combine(states)

if __name__ == "__main__":
    import sys
    sys.path.append('./Model')
    from individual import SEEIndividual

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ind = SEEIndividual(2, (1, 10, 3))
    code = ind.getDec()[0]
    # code = np.array([[3,3,3],
    #         [8,0,4],
    #         [0,5,3],
    #         [0,5,8],
    #         [7,2,3],
    #         [4,6,1],
    #         [1,1,5],
    #         [5,2,8],
    #         [1,1,8],
    #         [5,7,2],
    #         [2,8,3],
    #         [6,5,4],
    #         [2,2,1],
    #         [4,2,7],
    #         [2,5,8]])
    model = Node_Cell(
        code = code,
        prev_layers= [[32,32,3], [32,32,3]],
        channels=32,
        reduction=True,
        layer_id=0,
        init_layers=2,
        steps=1000,
        drop_path_keep_prob=1
    )
    inputs = torch.Tensor(np.random.rand(1, 3, 32, 32))
    model, inputs = model.to(device), inputs.to(device)
    y = model(inputs, inputs, 1)
    print(y.shape)
    # print(model.toDot())
    # print(y.cpu().detach().numpy().shape)