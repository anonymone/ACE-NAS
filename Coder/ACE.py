import numpy as np
import pandas as pd
from pandas import DataFrame
import random
from copy import deepcopy
import uuid

import torch
import torch.nn as nn
import torch.nn.functional as F

from Coder.Code_Interface import code
import Coder.Network.utils as net_tools
from Coder.Network.nn import SepConv, Conv, FinalCombine, MaybeCalibrateSize, Network_CIFAR, Network_IMAGENET

# use small search space
OPERATIONS = {
    0: SepConv,  # 3x3
    1: SepConv,  # 5x5
    2: nn.AvgPool2d,  # 3x3
    3: nn.MaxPool2d,  # 3x3
    # 4: Identity,
}

# use large search space
OPERATIONS_large = {
    4: Conv,  # 1x1
    5: Conv,  # 3x3
    6: Conv,  # 1x3 + 3x1
    7: Conv,  # 1x7 + 7x1
    8: nn.MaxPool2d,  # 2x2
    9: nn.MaxPool2d,  # 3x3
    10: nn.MaxPool2d,  # 5x5
    11: nn.AvgPool2d,  # 2x2
    12: nn.AvgPool2d,  # 3x3
    13: nn.AvgPool2d,  # 5x5
    14: SepConv,  # 3x3
    15: SepConv,  # 5x5
    # 14: Identity
}

# action defined
ACTION = {
    # 0: 'change_connection_A_B',
    0: 'add_edge_C',
    1: 'add_node_C',
    2: 'clone_node_a_parallel',
    3: 'clone_node_a_squeue',
    4: 'clone_nodes_A_parallel',
    5: 'clone_nodes_A_squeue',
    6: 'dense_path'
    # 3: 'substitute_node_B_for_type',  # A will be changed to type
    # 4: 'deldete_connection_A_B'
}


class code_parser:
    def __init__(self, node_search_space=OPERATIONS_large, action_search_space=ACTION):
        self.op_search_space = node_search_space
        self.act_search_space = action_search_space
        self.token_start_point = 4 if node_search_space == OPERATIONS_large else 0

    def get_op_token(self, code_type):
        return code_type % len(self.op_search_space) + self.token_start_point

    def get_action_token(self, code_type):
        return code_type % len(self.act_search_space)

    def get_op(self, code_type):
        code_type = code_type % len(
            self.op_search_space) + self.token_start_point
        return self.op_search_space[code_type]

    def get_action(self, code_type):
        code_type = code_type % len(self.act_search_space)
        return self.act_search_space[code_type]


def build_ACE(fitness_size, ind_params):
    return ACE(fitness_size=fitness_size,
               unit_number_range=ind_params.unit_num,
               value_boundary=ind_params.value_boundary,
               classes=ind_params.classes,
               layers=ind_params.layers,
               channels=ind_params.channels,
               keep_prob=ind_params.keep_prob,
               drop_path_keep_prob=ind_params.drop_path_keep_prob,
               use_aux_head=ind_params.use_aux_head)


class ACE(code):
    def __init__(self,
                 fitness_size=2,
                 unit_number_range=(10, 20),
                 value_boundary=(0, 15),
                 classes=10,
                 layers=6,
                 channels=48,
                 keep_prob=0.8,
                 drop_path_keep_prob=0.8,
                 use_aux_head=True,
                 **kwargs):
        super(ACE, self).__init__()
        self.fitness = np.zeros(fitness_size)
        self.vb = value_boundary
        self.unr = unit_number_range
        # self.normal_dec = np.random.randint(low=self.vb[0], high=self.vb[1], size=(
        #     random.randint(self.unr[0], self.unr[1]), 3))
        # self.reduct_dec = np.random.randint(low=self.vb[0], high=self.vb[1], size=(
        #     random.randint(self.unr[0], self.unr[1]), 3))
        self.normal_dec = self.code_init()
        self.reduct_dec = self.code_init()
        self.shape = (self.normal_dec.shape, self.reduct_dec.shape)
        # surrogate fitness
        self.fitness_SG = np.zeros(1)
        # model parameters
        self.classes = classes
        self.layers = layers
        self.channels = channels
        self.keep_prob = keep_prob
        self.drop_path_keep_prob = drop_path_keep_prob
        self.use_aux_head = use_aux_head

        for name in kwargs.keys():
            exec("self.{0} = kwargs['{0}']".format(str(name)))

    def code_init(self):
        dec = np.array([[np.random.randint(0, 4) if random.random() < 0.8 else np.random.randint(0, 7), np.random.randint(
            low=self.vb[0], high=self.vb[1]), np.random.randint(low=self.vb[0], high=self.vb[1])] for i in range(random.randint(self.unr[0], self.unr[1]))])
        return dec

    def to_string(self, callback=None) -> str:
        normal_string = "-".join([".".join([str(t) for t in unit])
                                  for unit in self.normal_dec])
        reduct_string = "-".join([".".join([str(t) for t in unit])
                                  for unit in self.reduct_dec])
        if callback is None:
            return "<--->".join([normal_string, reduct_string])
        else:
            return callback("<--->".join([normal_string, reduct_string]))

    def get_dec(self) -> (np.ndarray, np.ndarray):
        return deepcopy((self.normal_dec, self.reduct_dec))

    def get_fitness(self):
        return self.fitness

    def set_dec(self, dec: (np.ndarray, np.ndarray)):
        self.normal_dec = dec[0].astype("int").reshape(-1, 3)
        self.reduct_dec = dec[1].astype("int").reshape(-1, 3)
        self.shape = (self.normal_dec.shape, self.reduct_dec.shape)

    def get_fitnessSG(self):
        return self.fitness_SG

    def set_fitnessSG(self, sg_fitness):
        self.fitness_SG = np.array(sg_fitness).reshape(-1)

    def get_model(self, steps, imagenet=False, **kwargs):
        if len(kwargs) != 0:
            classes = kwargs.pop('classes')
            layers = kwargs.pop('layers')
            channels = kwargs.pop('channels')
            keep_prob = kwargs.pop('keep_prob')
            drop_path_keep_prob = kwargs.pop('drop_path_keep_prob')
            use_aux_head = kwargs.pop('use_aux_head')
        else:
            classes = self.classes
            layers = self.layers
            channels = self.channels
            keep_prob = self.keep_prob
            drop_path_keep_prob = self.drop_path_keep_prob
            use_aux_head = self.use_aux_head
        if imagenet:
            return Network_IMAGENET(ACE_Cell, self.get_dec(), classes, layers, channels, keep_prob, drop_path_keep_prob, use_aux_head, steps)
        else:
            return Network_CIFAR(ACE_Cell, self.get_dec(), classes, layers, channels, keep_prob, drop_path_keep_prob, use_aux_head, steps)


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
        self.steps = steps
        self.drop_path_keep_prob = drop_path_keep_prob
        self.multi_adds = 0
        self.node_name = "Node"

        if node_type == 0:
            pass
        elif node_type == 1:
            pass
        elif node_type == 2:
            pass
        elif node_type == 3:
            pass
        elif node_type == 4:
            # Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
            self.op = OPERATIONS_large[node_type](
                self.channels, self.channels, 1, stride, 0)
            self.node_name = "CONV1x1"
            input_shape = [input_shape[0] // stride,
                           input_shape[1] // stride, channels]
            self.multi_adds += 1 * 1 * channels * channels * \
                (input_shape[0] // stride) * (input_shape[1] // stride)
        elif node_type == 5:
            self.op = OPERATIONS_large[node_type](
                self.channels, self.channels, 3, stride, 1)
            self.node_name = "CONV3x3"
            input_shape = [input_shape[0] // stride,
                           input_shape[1] // stride, channels]
            self.multi_adds += 3 * 3 * channels * channels * \
                (input_shape[0] // stride) * (input_shape[1] // stride)
        elif node_type == 6:
            self.op = OPERATIONS_large[node_type](
                self.channels, self.channels, (1, 3), stride, ((0, 1), (1, 0)))
            self.node_name = "CONV1x3"
            input_shape = [input_shape[0] // stride,
                           input_shape[1] // stride, channels]
            self.multi_adds += 2 * 1 * 3 * channels * channels * \
                (input_shape[0] // stride) * (input_shape[1] // stride)
        elif node_type == 7:
            self.op = OPERATIONS_large[node_type](
                self.channels, self.channels, (1, 7), stride, ((0, 3), (3, 0)))
            self.node_name = "CONV1x7"
            input_shape = [input_shape[0] // stride,
                           input_shape[1] // stride, channels]
            self.multi_adds += 2 * 1 * 7 * channels * channels * \
                (input_shape[0] // stride) * (input_shape[1] // stride)
        elif node_type == 8:
            self.op = OPERATIONS_large[node_type](2, stride=stride, padding=0)
            self.node_name = "MAXPOOL2x2"
            input_shape = [input_shape[0] // stride,
                           input_shape[1] // stride, channels]
        elif node_type == 9:
            self.op = OPERATIONS_large[node_type](3, stride=stride, padding=1)
            self.node_name = "MAXPOOL3x3"
            input_shape = [input_shape[0] // stride,
                           input_shape[1] // stride, channels]
        elif node_type == 10:
            self.op = OPERATIONS_large[node_type](5, stride=stride, padding=2)
            self.node_name = "MAXPOOL5x5"
            input_shape = [input_shape[0] // stride,
                           input_shape[1] // stride, channels]
        elif node_type == 11:
            self.op = OPERATIONS_large[node_type](2, stride=stride, padding=0)
            self.node_name = "AVGPOOL2x2"
            input_shape = [input_shape[0] // stride,
                           input_shape[1] // stride, channels]
        elif node_type == 12:
            self.op = OPERATIONS_large[node_type](3, stride=stride, padding=1)
            self.node_name = "AVGPOOL3x3"
            input_shape = [input_shape[0] // stride,
                           input_shape[1] // stride, channels]
        elif node_type == 13:
            self.op = OPERATIONS_large[node_type](5, stride=stride, padding=2)
            self.node_name = "AVGPOOL5x5"
            input_shape = [input_shape[0] // stride,
                           input_shape[1] // stride, channels]
        elif node_type == 14:
            self.op = OPERATIONS_large[node_type](
                self.channels, self.channels, 3, stride, 1)
            self.node_name = "SepCONV3x3"
            input_shape = [input_shape[0] // stride,
                           input_shape[1] // stride, channels]
            self.multi_adds += 3 * 3 * channels * channels * \
                (input_shape[0] // stride) * (input_shape[1] // stride)
        elif node_type == 15:
            self.op = OPERATIONS_large[node_type](
                self.channels, self.channels, 5, stride, 2)
            self.node_name = "SepCONV5x5"
            input_shape = [input_shape[0] // stride,
                           input_shape[1] // stride, channels]
            self.multi_adds += 5 * 5 * channels * channels * \
                (input_shape[0] // stride) * (input_shape[1] // stride)
        self.out_shape = list(input_shape)
        
    def forward(self, x, step):
        if self.node_type in [8, 11]:
            x = F.pad(x, [0, 1, 0, 1])
        x = self.op(x)
        if self.drop_path_keep_prob is not None and self.training:
            x = net_tools.apply_drop_path(
                x, self.drop_path_keep_prob, self.layer_id, self.layers, step, self.steps)
        return x


class ACE_Cell(nn.Module):
    def __init__(self,
                 code,
                 prev_layers,
                 channels,
                 reduction,
                 layer_id,
                 init_layers,
                 steps,
                 drop_path_keep_prob=None):
        super(ACE_Cell, self).__init__()
        self.layer_id = layer_id
        self.reduction = reduction
        self.code = code
        self.init_layers = init_layers
        self.channels = channels
        self.steps = steps
        self.drop_path_keep_prob = drop_path_keep_prob
        self.ops = nn.ModuleDict()
        self.__parser = code_parser()
        self.multi_adds = 0
        # calibrate size
        prev_layers = [list(prev_layers[0]), list(prev_layers[1])]
        self.maybe_calibrate_size = MaybeCalibrateSize(prev_layers, channels)
        self.multi_adds += self.maybe_calibrate_size.multi_adds
        self.dependence_graph = self.decoder(code)

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

    def decoder(self, code):
        node_graph = {0: []}
        init_action, action_unit = code[0], code[1:]
        node_type_tokens = {0: "Input_Node"}
        # initialize the child network
        for i, node_type in zip(range(1, len(init_action)+1), init_action):
            node_graph[i] = [0]
            node_type_tokens[i] = self.__parser.get_op_token(node_type)
        # build node graph
        for act_token, param1, param2 in action_unit:
            action = self.__parser.get_action(act_token)
            if action == 'add_edge_C':
                node_id1, node_id2 = param1 % (
                    len(node_graph)-1)+1, param2 % (len(node_graph)-1)+1
                if node_id1 == node_id2:
                    if node_id2 == 1:
                        node_id2 += 1
                    else:
                        node_id2 = node_id2 - 1
                is_loop, _ = net_tools.isLoop(
                    node_graph, (node_id1, node_id2), (0, node_id2))
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
                # node_type, pre_node = param1, param2 % len(node_graph)
                pre_node, node_type = param1%len(node_graph), param2
                node_graph[len(node_graph)] = [pre_node]
                node_type_tokens[len(node_type_tokens)
                                 ] = self.__parser.get_op_token(node_type)
            elif action == 'clone_node_a_parallel':
                # node_id, node_type = param1 % (
                #     len(node_graph)-1)+1, param2
                node_type, node_id = param1, param2%(
                    len(node_graph)-1)+1
                # if remain_edges:
                new_node_id = len(node_graph)
                node_graph[new_node_id] = node_graph[node_id].copy()
                for i, v in node_graph.items():
                    if node_id in v:
                        node_graph[i].append(new_node_id)
                node_type_tokens[len(node_type_tokens)
                                    ] = self.__parser.get_op_token(node_type)
                # else:
                #     node_graph[len(node_graph)] = [0]
                #     node_type_tokens[len(node_type_tokens)
                #                      ] = node_type_tokens[node_id]
            elif action == 'clone_node_a_squeue':
                # node_id, node_type = param1 % (len(node_graph)-1)+1, param2
                node_type, node_id = param1, param2% (len(node_graph)-1)+1
                new_node_id = len(node_graph)
                for i in node_graph:
                    if node_id in node_graph[i]:
                        node_graph[i].remove(node_id)
                        node_graph[i].append(new_node_id)
                node_graph[new_node_id] = [node_id]
                node_type_tokens[len(node_type_tokens)
                                 ] = self.__parser.get_op_token(node_type)
            elif action == 'clone_nodes_A_parallel':
                node_id, path_depth = param1 % (
                    len(node_graph)-1)+1, param2 % 5+1
                # node_set includes node_id
                node_set = self.get_node_path(node_graph, node_id, path_depth)
                if len(node_set) <= 1:
                    continue
                for n in range(len(node_set)):
                    node_graph[len(node_graph)] = node_graph[node_id].copy(
                    ) if n == 0 else [len(node_graph)-1]
                    node_type_tokens[len(node_type_tokens)
                                     ] = node_type_tokens[node_set[n]]
                for i in node_graph:
                    if node_set[-1] in node_graph[i]:
                        node_graph[i].append(len(node_graph)-1)
            elif action == 'clone_nodes_A_squeue':
                node_id, path_depth = param1 % (
                    len(node_graph)-1)+1, param2 % 5+1
                # node_set includes node_id
                node_set = self.get_node_path(node_graph, node_id, path_depth)
                if len(node_set) <= 1:
                    continue
                head = 0
                for n in range(len(node_set)):
                    node_graph[len(node_graph)] = [node_set[-1]
                                                   if n == 0 else len(node_graph)-1]
                    node_type_tokens[len(node_type_tokens)
                                     ] = node_type_tokens[node_set[n]]
                    if n == 0:
                        head = len(node_graph)-1
                for i in node_graph:
                    if node_set[-1] in node_graph[i] and i != head:
                        node_graph[i].remove(node_set[-1])
                        node_graph[i].append(len(node_graph)-1)
            elif action == 'dense_path':
                node_id, path_depth = param1 % (
                    len(node_graph)-1)+1, param2 % 5+1
                # node_set includes node_id
                node_set = self.get_node_path(node_graph, node_id, path_depth)
                if len(node_set) <= 1:
                    continue
                r_node_set = node_set.copy()
                r_node_set.reverse()
                for i, n in enumerate(r_node_set[:-1]):
                    for m in node_set[:-i-1]:
                        if m not in node_graph[n]:
                            node_graph[n].append(m)
            else:
                raise Exception(
                    'Encountered the unknown action token: {0}'.format(action))
        self.node_type_tokens = node_type_tokens
        # static the concat node
        nodes_all = list(node_graph.keys())
        nodes_used = []
        self.nodes_reduction = []

        def inline_remove(node_all, node_used):
            for i in node_used:
                node_all.remove(i)
            return node_all
        for node_id, nodes in node_graph.items():
            nodes_used += nodes
            if 0 in nodes:
                self.nodes_reduction.append(node_id)
        self.nodes_used = list(set(nodes_used))
        self.concat_nodes = inline_remove(nodes_all, self.nodes_used)
        # build node instant
        _, node_order = net_tools.isLoop(node_graph)
        self.node_order = node_order
        prev_layers = dict()
        for node_id in self.node_order:
            if node_id == 0:
                input_shape = self.maybe_calibrate_size.out_shape[0]
                prev_layers[node_id] = input_shape
                continue
            stride = 2 if self.reduction and node_id in self.nodes_reduction else 1
            input_shape = min([prev_layers[i] for i in node_graph[node_id]])
            node = Node(self.node_type_tokens[node_id], input_shape, self.channels, stride=stride,
                        drop_path_keep_prob=self.drop_path_keep_prob,
                        layer_id=self.layer_id,
                        layers=self.init_layers,
                        steps=self.steps)
            self.ops[str(node_id)] = node
            self.multi_adds += node.multi_adds
            prev_layers[node_id] = node.out_shape

        out_hw = min([shape[0] for i, shape in prev_layers.items()
                      if i in self.concat_nodes])
        self.final_combine = FinalCombine(
            prev_layers, out_hw, self.channels, self.concat_nodes)
        self.out_shape = [out_hw, out_hw,
                          self.channels * len(self.concat_nodes)]
        self.multi_adds += self.final_combine.multi_adds
        return node_graph

    def get_node_path(self, graph, start_node, depth):
        node_set = [start_node]
        found_node = -1
        branch = 0
        for i in range(depth):
            for i, i_set in graph.items():
                if node_set[-1] in i_set:
                    found_node = i
                    branch += 1
            if branch == 1:
                node_set.append(found_node)
                branch = 0
                found_node = -1
            else:
                break
        return node_set

    def to_dot(self):
        node_temp = "node_#id[shape=circle, color=pink, fontcolor=red, fontsize=10,label=#node_type];\n"
        edge_temp = "node_#pre_node_id -> node_#node_id;\n"
        graph_temp = "subgraph #name {\n#node_list\n#edge_list}"
        dependence_temp = deepcopy(self.dependence_graph)
        dependence_temp[len(dependence_temp)] = deepcopy(self.concat_nodes)
        nodes_list = ""
        edges_list = ""
        name = "Reduction" if self.reduction else "Normal"
        for node_id, pre_nodes in dependence_temp.items():
            if node_id == 0:
                nodes_list += node_temp.replace("#id", name +
                                                str(node_id)).replace("#node_type", "INPUT")
                continue
            elif node_id == len(self.dependence_graph):
                nodes_list += node_temp.replace("#id", name +
                                                str(node_id)).replace("#node_type", "CONCAT")
            else:
                nodes_list += node_temp.replace("#id", name + str(node_id)).replace(
                    "#node_type", self.ops[str(node_id)].node_name)
            for edge in pre_nodes:
                edges_list += edge_temp.replace("#pre_node_id", name+str(
                    edge)).replace("#node_id", name+str(node_id))
        return graph_temp.replace("#node_list", nodes_list).replace("#edge_list", edges_list).replace("#name", name)
