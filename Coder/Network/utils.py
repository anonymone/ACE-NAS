import torch
import torch.nn as nn 
import torchvision.transforms as transforms
# from torch.autograd import Variable
import torch.nn.functional as F
from copy import deepcopy

def apply_drop_path(x, drop_path_keep_prob, layer_id, layers, step, steps):
    layer_ratio = float(layer_id+1) / (layers)
    drop_path_keep_prob = 1.0 - layer_ratio * (1.0 - drop_path_keep_prob)
    step_ratio = float(step + 1) / steps
    drop_path_keep_prob = 1.0 - step_ratio * (1.0 - drop_path_keep_prob)
    if drop_path_keep_prob < 1.:
        mask = torch.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(drop_path_keep_prob).cuda()
        #x.div_(drop_path_keep_prob)
        #x.mul_(mask)
        x = x / drop_path_keep_prob * mask
    return x

def isLoop(graph, newEdge=None, deleteEdge=None):
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
    if deleteEdge is not None:
        a,b = deleteEdge
        if a in graph[b]:
            graph[b].remove(a)
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