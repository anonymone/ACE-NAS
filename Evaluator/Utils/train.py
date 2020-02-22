import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import numpy as np
import logging

from Evaluator.Utils.recoder import AvgrageMeter, error_rate, accuracy

class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss

def build_train_utils(model, 
                    l2_reg = 3e-4,
                    momentum = 0.9,
                    lr_min=0.0,
                    lr_max = 0.025,
                    decay_period:'used ONLY ImageNet'=1,
                    gamma:'used ONLY ImageNet'=0.97,
                    label_smooth:'used ONLY ImageNet'=0.1,
                    dataset:str='CIFAR10',
                    epochs: 'the number of total train epochs' = 600,
                    epoch: 'the dataset is modified from crash down epoch.' = -1,
                    optimizer_state_dict: 'used to restore the optimizer state.' = None):
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr_max,
        momentum=momentum,
        weight_decay=l2_reg,
    )
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    
    if dataset in ['CIFAR10', 'CIFAR100']:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(epochs), lr_min, epoch)
        train_criterion = nn.CrossEntropyLoss()
        eval_criterion = nn.CrossEntropyLoss()
    elif dataset in ['IMAGENET', 'ImageNet']:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, decay_period, gamma, epoch)
        train_criterion = CrossEntropyLabelSmooth(1000, label_smooth)
        eval_criterion = nn.CrossEntropyLoss()
    return train_criterion, eval_criterion, optimizer, scheduler

def train(trainset, model, optimizer, global_step:'recent epoch', criterion, device='cpu', rate_static=error_rate) -> 'loss, top1, top5,':
    loss_rec = AvgrageMeter()
    top1_rec = AvgrageMeter()
    top5_rec = AvgrageMeter()
    # process bar
    bar_length = 30
    total = len(trainset)
    # stable parameters
    grad_bound = 5.0
    # load to device
    model = model.cuda()
    criterion = criterion.cuda()

    model.train()

    for step, (input, target) in enumerate(trainset):
        print('\r[Training {0:>2d}/{1:>2d}]'.format(step+1, total)+'['+'*'*np.floor(bar_length*((step+1)/total)).astype('int') +
                  '-'*(bar_length-np.floor(bar_length*((step+1)/total)).astype('int'))+']',end='')
        input = input.cuda().requires_grad_()
        target = target.cuda()
    
        optimizer.zero_grad()
        logits, aux_logits = model(input, global_step)
        global_step += 1
        loss = criterion(logits, target)
        if aux_logits is not None:
            aux_loss = criterion(aux_logits, target)
            loss += 0.4 * aux_loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_bound)
        optimizer.step()
    
        prec1, prec5 = rate_static(logits, target, topk=(1, 5))
        n = input.size(0)
        loss_rec.update(loss.data.item(), n)
        top1_rec.update(prec1.data.item(), n)
        top5_rec.update(prec5.data.item(), n)

    return loss_rec.avg, top1_rec.avg, top5_rec.avg, global_step

def valid(evalset, model, criterion, device='cpu', rate_static=error_rate) -> 'loss, top1, top5,':
    loss_rec = AvgrageMeter()
    top1_rec = AvgrageMeter()
    top5_rec = AvgrageMeter()
    # process bar
    bar_length = 30
    total = len(evalset)

    model = model.cuda()
    criterion = criterion.cuda()

    with torch.no_grad():
        model.eval()
        for step, (input, target) in enumerate(evalset):
            print('\r[Training {0:>2d}/{1:>2d}]'.format(step+1, total)+'['+'*'*np.floor(bar_length*((step+1)/total)).astype('int') +
                  '-'*(bar_length-np.floor(bar_length*((step+1)/total)).astype('int'))+']',end='')
            input = input.cuda().requires_grad_()
            target = target.cuda()
        
            logits, _ = model(input)
            loss = criterion(logits, target)
        
            prec1, prec5 = rate_static(logits, target, topk=(1, 5))
            n = input.size(0)
            loss_rec.update(loss.data.item(), n)
            top1_rec.update(prec1.data.item(), n)
            top5_rec.update(prec5.data.item(), n)

    return loss_rec.avg, top1_rec.avg, top5_rec.avg

def add_flops_counting_methods(net_main_module):
    # adding additional methods to the existing module object,
    # this is done this way so that each function has access to self object
    net_main_module.start_flops_count = start_flops_count.__get__(net_main_module)
    net_main_module.stop_flops_count = stop_flops_count.__get__(net_main_module)
    net_main_module.reset_flops_count = reset_flops_count.__get__(net_main_module)
    net_main_module.compute_average_flops_cost = compute_average_flops_cost.__get__(net_main_module)

    net_main_module.reset_flops_count()

    # Adding variables necessary for masked flops computation
    net_main_module.apply(add_flops_mask_variable_or_reset)

    return net_main_module


def compute_average_flops_cost(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Returns current mean flops consumption per image.

    """

    batches_count = self.__batch_counter__
    flops_sum = 0
    for module in self.modules():
        if is_supported_instance(module):
            flops_sum += module.__flops__

    return flops_sum / batches_count


def start_flops_count(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Activates the computation of mean flops consumption per image.
    Call it before you run the network.

    """
    add_batch_counter_hook_function(self)
    self.apply(add_flops_counter_hook_function)


def stop_flops_count(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Stops computing the mean flops consumption per image.
    Call whenever you want to pause the computation.

    """
    remove_batch_counter_hook_function(self)
    self.apply(remove_flops_counter_hook_function)


def reset_flops_count(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.

    Resets statistics computed so far.

    """
    add_batch_counter_variables_or_reset(self)
    self.apply(add_flops_counter_variable_or_reset)


def add_flops_mask(module, mask):
    def add_flops_mask_func(module):
        if isinstance(module, torch.nn.Conv2d):
            module.__mask__ = mask
    module.apply(add_flops_mask_func)


def remove_flops_mask(module):
    module.apply(add_flops_mask_variable_or_reset)


# ---- Internal functions
def is_supported_instance(module):
    if isinstance(module, (torch.nn.Conv2d, torch.nn.ReLU, torch.nn.PReLU, torch.nn.ELU, \
                           torch.nn.LeakyReLU, torch.nn.ReLU6, torch.nn.Linear, \
                           torch.nn.MaxPool2d, torch.nn.AvgPool2d, torch.nn.BatchNorm2d, \
                           torch.nn.Upsample, nn.AdaptiveMaxPool2d, nn.AdaptiveAvgPool2d)):
        return True

    return False


def empty_flops_counter_hook(module, input, output):
    module.__flops__ += 0


def upsample_flops_counter_hook(module, input, output):
    output_size = output[0]
    batch_size = output_size.shape[0]
    output_elements_count = batch_size
    for val in output_size.shape[1:]:
        output_elements_count *= val
    module.__flops__ += output_elements_count


def relu_flops_counter_hook(module, input, output):
    active_elements_count = output.numel()
    module.__flops__ += active_elements_count


def linear_flops_counter_hook(module, input, output):
    input = input[0]
    batch_size = input.shape[0]
    module.__flops__ += batch_size * input.shape[1] * output.shape[1]


def pool_flops_counter_hook(module, input, output):
    input = input[0]
    module.__flops__ += np.prod(input.shape)

def bn_flops_counter_hook(module, input, output):
    module.affine
    input = input[0]

    batch_flops = np.prod(input.shape)
    if module.affine:
        batch_flops *= 2
    module.__flops__ += batch_flops

def conv_flops_counter_hook(conv_module, input, output):
    # Can have multiple inputs, getting the first one
    input = input[0]

    batch_size = input.shape[0]
    output_height, output_width = output.shape[2:]

    kernel_height, kernel_width = conv_module.kernel_size
    in_channels = conv_module.in_channels
    out_channels = conv_module.out_channels
    groups = conv_module.groups

    filters_per_channel = out_channels // groups
    conv_per_position_flops = kernel_height * kernel_width * in_channels * filters_per_channel

    active_elements_count = batch_size * output_height * output_width

    if conv_module.__mask__ is not None:
        # (b, 1, h, w)
        flops_mask = conv_module.__mask__.expand(batch_size, 1, output_height, output_width)
        active_elements_count = flops_mask.sum()

    overall_conv_flops = conv_per_position_flops * active_elements_count

    bias_flops = 0

    if conv_module.bias is not None:

        bias_flops = out_channels * active_elements_count

    overall_flops = overall_conv_flops + bias_flops

    conv_module.__flops__ += overall_flops


def batch_counter_hook(module, input, output):
    # Can have multiple inputs, getting the first one
    input = input[0]
    batch_size = input.shape[0]
    module.__batch_counter__ += batch_size


def add_batch_counter_variables_or_reset(module):

    module.__batch_counter__ = 0


def add_batch_counter_hook_function(module):
    if hasattr(module, '__batch_counter_handle__'):
        return

    handle = module.register_forward_hook(batch_counter_hook)
    module.__batch_counter_handle__ = handle


def remove_batch_counter_hook_function(module):
    if hasattr(module, '__batch_counter_handle__'):
        module.__batch_counter_handle__.remove()
        del module.__batch_counter_handle__


def add_flops_counter_variable_or_reset(module):
    if is_supported_instance(module):
        module.__flops__ = 0


def add_flops_counter_hook_function(module):
    if is_supported_instance(module):
        if hasattr(module, '__flops_handle__'):
            return

        if isinstance(module, torch.nn.Conv2d):
            handle = module.register_forward_hook(conv_flops_counter_hook)
        elif isinstance(module, (torch.nn.ReLU, torch.nn.PReLU, torch.nn.ELU, \
                                 torch.nn.LeakyReLU, torch.nn.ReLU6)):
            handle = module.register_forward_hook(relu_flops_counter_hook)
        elif isinstance(module, torch.nn.Linear):
            handle = module.register_forward_hook(linear_flops_counter_hook)
        elif isinstance(module, (torch.nn.AvgPool2d, torch.nn.MaxPool2d, nn.AdaptiveMaxPool2d, \
                                 nn.AdaptiveAvgPool2d)):
            handle = module.register_forward_hook(pool_flops_counter_hook)
        elif isinstance(module, torch.nn.BatchNorm2d):
            handle = module.register_forward_hook(bn_flops_counter_hook)
        elif isinstance(module, torch.nn.Upsample):
            handle = module.register_forward_hook(upsample_flops_counter_hook)
        else:
            handle = module.register_forward_hook(empty_flops_counter_hook)
        module.__flops_handle__ = handle


def remove_flops_counter_hook_function(module):
    if is_supported_instance(module):
        if hasattr(module, '__flops_handle__'):
            module.__flops_handle__.remove()
            del module.__flops_handle__
# --- Masked flops counting


# Also being run in the initialization
def add_flops_mask_variable_or_reset(module):
    if is_supported_instance(module):
        module.__mask__ = None
