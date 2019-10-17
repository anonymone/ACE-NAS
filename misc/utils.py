import os
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F


class AvgrageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10(args):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    train_transform.transforms.append(transforms.Normalize(CIFAR_MEAN, CIFAR_STD))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, valid_transform


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        x.div_(keep_prob)
        x.mul_(mask)
    return x


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
        print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        if not os.path.exists(os.path.join(path, 'scripts')):
            os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


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

class ReLUConvBN(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.relu = nn.ReLU(inplace=False)
        self.conv = nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)
    
    def forward(self, x):
        x = self.relu(x)
        x = self.conv(x)
        # if bn_train:
        #     self.bn.train()
        x = self.bn(x)
        return x

class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.path1 = nn.Sequential(nn.AvgPool2d(1, stride=2, padding=0, count_include_pad=False),
                                   nn.Conv2d(C_in, C_out // 2, 1, bias=False))
        self.path2 = nn.Sequential(nn.AvgPool2d(1, stride=2, padding=0, count_include_pad=False),
                                   nn.Conv2d(C_in, C_out // 2, 1, bias=False))
        self.bn = nn.BatchNorm2d(C_out, affine=affine)
    
    def forward(self, x, bn_train=False):
        if bn_train:
            self.bn.train()
        path1 = x
        path2 = F.pad(x, (0, 1, 0, 1), "constant", 0)[:, :, 1:, 1:]
        out = torch.cat([self.path1(path1), self.path2(path2)], dim=1)
        out = self.bn(out)
        return out

class FinalCombine(nn.Module):
    def __init__(self, layers, out_hw, channels, concat, affine=True):
        super(FinalCombine, self).__init__()
        self.out_hw = out_hw
        self.channels = channels
        self.concat = concat
        self.ops = nn.ModuleList()
        self.concat_fac_op_dict = {}
        self.multi_adds = 0
        for i in concat:
            hw = layers[i][0]
            if hw > out_hw:
                assert hw == 2 * out_hw, 'hw is {0}\n out_hw is {1}\n i is {2}'.format(hw, out_hw, i)
                self.concat_fac_op_dict[i] = len(self.ops)
                self.ops.append(FactorizedReduce(layers[i][-1], channels, affine))
                self.multi_adds += 1 * 1 * layers[i][-1] * channels * out_hw * out_hw
        
    def forward(self, states, bn_train=False):
        for i in self.concat:
            if i in self.concat_fac_op_dict:
                states[i] = self.ops[self.concat_fac_op_dict[i]](states[i], bn_train)
        out = torch.cat([states[i] for i in self.concat], dim=1)
        return out



# Used to adjust the size of multi-inputs.
class MaybeCalibrateSize(nn.Module):
    def __init__(self, layers, channels, affine=True):
        super(MaybeCalibrateSize, self).__init__()
        self.channels = channels
        self.multi_adds = 0
        hw = [layer[0] for layer in layers]
        c = [layer[-1] for layer in layers]
        
        x_out_shape = [hw[0], hw[0], c[0]]
        y_out_shape = [hw[1], hw[1], c[1]]
        # previous reduction cell
        if hw[0] != hw[1]:
            assert hw[0] == 2 * hw[1]
            self.relu = nn.ReLU(inplace=False)
            self.preprocess_x = FactorizedReduce(c[0], channels, affine)
            x_out_shape = [hw[1], hw[1], channels]
            self.multi_adds += 1 * 1 * c[0] * channels * hw[1] * hw[1]
        elif c[0] != channels:
            self.preprocess_x = ReLUConvBN(c[0], channels, 1, 1, 0, affine)
            x_out_shape = [hw[0], hw[0], channels]
            self.multi_adds += 1 * 1 * c[0] * channels * hw[1] * hw[1]
        if c[1] != channels:
            self.preprocess_y = ReLUConvBN(c[1], channels, 1, 1, 0, affine)
            y_out_shape = [hw[1], hw[1], channels]
            self.multi_adds += 1 * 1 * c[1] * channels * hw[1] * hw[1]
            
        self.out_shape = [x_out_shape, y_out_shape]
    
    def forward(self, s0, s1):
        if s0.size(2) != s1.size(2):
            s0 = self.relu(s0)
            s0 = self.preprocess_x(s0)
        elif s0.size(1) != self.channels:
            s0 = self.preprocess_x(s0)
        if s1.size(1) != self.channels:
            s1 = self.preprocess_y(s1)
        out = torch.add(s0,s1)
        return out

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

class AuxHeadCIFAR(nn.Module):
    def __init__(self, C_in, classes):
        """assuming input size 8x8"""
        super(AuxHeadCIFAR, self).__init__()
        self.relu1 = nn.ReLU(inplace=False)
        self.avg_pool = nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False)
        self.conv1 = nn.Conv2d(C_in, 128, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(128, 768, 2, bias=False)
        self.bn2 = nn.BatchNorm2d(768)
        self.relu3 = nn.ReLU(inplace=False)
        self.classifier = nn.Linear(768, classes)
        
    def forward(self, x, bn_train=False):
        if bn_train:
            self.bn1.train()
            self.bn2.train()
        x = self.relu1(x)
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu3(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x

def normalization(data):
    return (data - np.min(data,0))/(np.max(data,0)-np.min(data,0))

def hv_2d(pop_obj, ref_point=None):
    if ref_point is None:
        ref_point = np.ones(shape=(1,2))

    pop_obj = pop_obj[:,1:]
    # normalization data
    obj_min = pop_obj.min(0)
    obj_max = pop_obj.max(0) * 1.1
    pop_obj = (pop_obj - obj_min)/(obj_max - obj_min)

    # calculate HV
    pop_obj = pop_obj[np.argsort(pop_obj[:,0]),:]
    hv  = 0.0
    for i in range(1, pop_obj.shape[0]):
        obj_1 = np.abs(pop_obj[i,0] - pop_obj[i-1,0])
        obj_2 = pop_obj[i,1] + pop_obj[i-1,1]
        hv += obj_2*obj_1*0.5
    return hv

if __name__ == "__main__":
    pop = np.random.randint(0,9,(10,3))
    print(hv_2d(pop))  