import torch
import torch.nn as nn 
import torchvision.transforms as transforms
# from torch.autograd import Variable
import torch.nn.functional as F
from copy import deepcopy

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

class SepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )
       
    def forward(self, x):
        return self.op(x)

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

class AuxHeadImageNet(nn.Module):
    def __init__(self, C_in, classes):
        """input should be in [B, C, 7, 7]"""
        super(AuxHeadImageNet, self).__init__()
        self.relu1 = nn.ReLU(inplace=True)
        self.avg_pool = nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False)
        self.conv1 = nn.Conv2d(C_in, 128, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(128, 768, 2, bias=False)
        self.bn2 = nn.BatchNorm2d(768)
        self.relu3 = nn.ReLU(inplace=True)
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

class Network_IMAGENET(nn.Module):
    def __init__(self,
                cell_decoder:'a function used to decode the cell.',
                code,
                classes, 
                layers, 
                channels,
                keep_prob,
                drop_path_keep_prob,
                use_aux_head,
                steps:'the total training steps equs to (trainset_size / batchsize) * epochs'):
        super(Network_IMAGENET, self).__init__()
        self.classes = classes
        self.layers = layers
        self.channels = channels
        self.keep_prob = keep_prob
        self.drop_path_keep_prob = drop_path_keep_prob
        self.use_aux_head = use_aux_head
        self.steps = steps
        self.normal_code, self.reduct_code = code
        self.Cell = cell_decoder

        self.reduction_layers = [self.layers, 2*self.layers +1]
        self.layers = self.layers*3
        self.multi_adds = 0

        if self.use_aux_head:
            self.aux_head_index = self.reduction_layers[-1]
        channels = self.channels
        self.stem0 = nn.Sequential(
            nn.Conv2d(3, channels // 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.multi_adds += 3 * 3 * 3 * channels // 2 * 112 * 112 + 3 * 3 * channels // 2 * channels * 56 * 56
        self.stem1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.multi_adds += 3 * 3 * channels * channels * 56 * 56
        outs = [[56, 56, channels], [28, 28, channels]]
        channels = self.channels
        self.cells = nn.ModuleList()
        for i in range(self.layers+2):
            if i not in self.reduction_layers:
                cell = self.Cell(code = self.normal_code, 
                                 prev_layers=outs, 
                                 channels=self.channels, 
                                 reduction=False, 
                                 layer_id=i,
                                 init_layers=self.layers + 2,
                                 steps= self.steps,
                                 drop_path_keep_prob=self.drop_path_keep_prob)
            else:
                channels *= 2
                cell = self.Cell(code = self.reduct_code, 
                                 prev_layers=outs, 
                                 channels=self.channels, 
                                 reduction=True, 
                                 layer_id=i,
                                 init_layers=self.layers + 2,
                                 steps= self.steps,
                                 drop_path_keep_prob=self.drop_path_keep_prob)
            self.multi_adds += cell.multi_adds
            self.cells.append(cell)
            outs = [outs[-1], cell.out_shape]

            if self.use_aux_head and i == self.aux_head_index:
                self.auxiliary_head = AuxHeadImageNet(outs[-1][-1], classes)

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(1 - self.keep_prob)
        self.classifier = nn.Linear(outs[-1][-1], classes)
        
        self.init_parameters()
    
    def init_parameters(self):
        for w in self.parameters():
            if w.data.dim() >= 2:
                nn.init.kaiming_normal_(w.data)
    
    def forward(self, input, step=None):
        aux_logits = None
        s0 = self.stem0(input)
        s1 = self.stem1(s0)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, step)
            if self.use_aux_head and i == self.aux_head_index and self.training:
                aux_logits = self.auxiliary_head(s1)
        
        out = s1
        out = self.global_pooling(out)
        out = self.dropout(out)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, aux_logits

class Network_CIFAR(nn.Module):
    def __init__(self,
                cell_decoder:'a function used to decode the cell.',
                code,
                classes, 
                layers, 
                channels,
                keep_prob,
                drop_path_keep_prob,
                use_aux_head,
                steps:'the total training steps equs to (trainset_size / batchsize) * epochs'):
        super(Network_CIFAR, self).__init__()
        self.classes = classes
        self.layers = layers
        self.channels = channels
        self.keep_prob = keep_prob
        self.drop_path_keep_prob = drop_path_keep_prob
        self.use_aux_head = use_aux_head
        self.steps = steps
        self.normal_code, self.reduct_code = code
        self.Cell = cell_decoder

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
                cell = self.Cell(code = self.normal_code, 
                                 prev_layers=outs, 
                                 channels=self.channels, 
                                 reduction=False, 
                                 layer_id=i,
                                 init_layers=self.layers + 2,
                                 steps= self.steps,
                                 drop_path_keep_prob=self.drop_path_keep_prob)
            else:
                cell = self.Cell(code = self.reduct_code, 
                                 prev_layers=outs, 
                                 channels=self.channels, 
                                 reduction=True, 
                                 layer_id=i,
                                 init_layers=self.layers + 2,
                                 steps= self.steps,
                                 drop_path_keep_prob=self.drop_path_keep_prob)
            # CAUTION cell need multi_adds property
            self.multi_adds += cell.multi_adds
            self.cells.append(cell)
            outs = [outs[-1], cell.out_shape]

            if self.use_aux_head and i == self.aux_head_index:
                self.auxiliary_head = AuxHeadCIFAR(outs[-1][-1], classes)
        
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
    
    def to_dot(self):
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
                cells_inner_list[1] = cell.to_dot()
            elif not cell.reduction and cells_inner_list[0] == None:
                cells_inner_list[0] = cell.to_dot()
            # except:
                # pass
            pre_pre_id, pre_id = i-2 if i > 1 else 0, i-1
            if pre_pre_id != 0:
                edges_list += edge_temp.replace("#id_pre", str(pre_pre_id)).replace("#id", str(i))
            edges_list += edge_temp.replace("#id_pre", str(pre_id)).replace("#id", str(i))
        final_graph = final_graph.replace("#cells", cells_list).replace("#edges", edges_list)
        cells_inner_list = set(cells_inner_list)
        return final_graph.replace("#inner_cells","\n".join(cells_inner_list))