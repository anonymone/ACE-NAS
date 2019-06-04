import sys
sys.path.insert(0, './')

import os
import os
import numpy as np
import torch
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from Model import layers, individual

from Dataset import cifar10Search as Cifar10

import time
from misc import utils
from misc.flops_counter import add_flops_counting_methods

device = 'cuda'

# def main(code, epochs, save='SearchExp', exprRoot='./Experiments', seed=0, gpu=0, initChannel=24, modelLayers=11, auxiliary=False, cutout=False, dropPathProb=0.0):
def main(code, arg):
    # init parameters
    epochs = arg.trainSearch_epoch
    save=arg.trainSearch_save
    exprRoot=arg.trainSearch_exprRoot
    seed=0
    gpu=0
    initChannel= arg.trainSearch_initChannel
    auxiliary = arg.trainSearch_auxiliary
    cutout = arg.trainSearch_cutout
    dropPathProb = arg.trainSearch_dropPathProb
    # ---- train logger ----------------- #
    save_pth = os.path.join(exprRoot, '{}'.format(save))
    utils.create_exp_dir(save_pth)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')

    # ---- parameter values setting ----- #
    CIFAR_CLASSES = 10
    learning_rate = 0.025
    momentum = 0.9
    weight_decay = 3e-4
    data_root = arg.dataRoot
    batch_size = 128
    cutout_length = 16
    auxiliary_weight = 0.4
    grad_clip = 5
    report_freq = 50
    train_params = {
        'auxiliary': auxiliary,
        'auxiliary_weight': auxiliary_weight,
        'grad_clip': grad_clip,
        'report_freq': report_freq,
    }

    channels = [(3, initChannel),
                (initChannel, 2*initChannel),
                (2*initChannel, 4*initChannel)]

    model = layers.SEENetworkGenerator(
        code.getDec(), channels, CIFAR_CLASSES, (32, 32))

    # logging.info("Genome = %s", genome)
    logging.info("Architecture = %s", code.toString())

    torch.cuda.set_device(gpu)
    cudnn.benchmark = True
    torch.manual_seed(seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(seed)

    n_params = (np.sum(np.prod(v.size()) for v in filter(
        lambda p: p.requires_grad, model.parameters())) / 1e6)
    model = model.to(device)

    logging.info("param size = %fMB", n_params)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.SGD(
        parameters,
        learning_rate,
        momentum=momentum,
        weight_decay=weight_decay
    )

    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    if cutout:
        train_transform.transforms.append(utils.Cutout(cutout_length))

    train_transform.transforms.append(
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    train_data = Cifar10.CIFAR10(
        root=data_root, train=True, download=True, transform=train_transform)
    valid_data = Cifar10.CIFAR10(
        root=data_root, train=False, download=True, transform=valid_transform)

    # num_train = len(train_data)
    # indices = list(range(num_train))
    # split = int(np.floor(train_portion * num_train))

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size,
        # sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=4)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=batch_size,
        # sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, int(epochs))

    for epoch in range(epochs):
        scheduler.step()
        logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
        model.droprate = dropPathProb * epoch / epochs

        train_acc, train_obj = train(
            train_queue, model, criterion, optimizer, train_params)
        logging.info('train_acc %f', train_acc)

    valid_acc, valid_obj = infer(valid_queue, model, criterion)
    logging.info('valid_acc %f', valid_acc)

    # calculate for flops
    model = add_flops_counting_methods(model)
    model.eval()
    model.start_flops_count()
    random_data = torch.randn(1, 3, 32, 32)
    model(torch.autograd.Variable(random_data).to(device))
    n_flops = np.round(model.compute_average_flops_cost() / 1e6, 4)
    logging.info('flops = %f', n_flops)

    # save to file
    # os.remove(os.path.join(save_pth, 'log.txt'))
    with open(os.path.join(save_pth, 'log.txt'), "w") as file:
        # file.write("Genome = {}\n".format(genome))
        file.write("Architecture = {}\n".format(code.toString()))
        file.write("param size = {}MB\n".format(n_params))
        file.write("flops = {}MB\n".format(n_flops))
        file.write("valid_acc = {}\n".format(valid_acc))

    # logging.info("Architecture = %s", genotype))

    return {
        'valid_acc': valid_acc,
        'params': n_params,
        'flops': n_flops,
    }

# Training


def train(train_queue, net, criterion, optimizer, params):
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for step, (inputs, targets) in enumerate(train_queue):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs, outputs_aux = net(inputs)
        loss = criterion(outputs, targets)

        if params['auxiliary']:
            loss_aux = criterion(outputs_aux, targets)
            loss += params['auxiliary_weight'] * loss_aux

        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), params['grad_clip'])
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    #     if step % args.report_freq == 0:
    #         logging.info('train %03d %e %f', step, train_loss/total, 100.*correct/total)
    #
    # logging.info('train acc %f', 100. * correct / total)

    return 100.*correct/total, train_loss/total


def infer(valid_queue, net, criterion):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for step, (inputs, targets) in enumerate(valid_queue):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # if step % args.report_freq == 0:
            #     logging.info('valid %03d %e %f', step, test_loss/total, 100.*correct/total)

    acc = 100.*correct/total
    # logging.info('valid acc %f', 100. * correct / total)

    return acc, test_loss/total


if __name__ == "__main__":
    SEE_V3 = individual.SEEIndividual(31, 2)
    start = time.time()
    print(main(code=SEE_V3, epochs=1, save='SEE_V3', seed=1, initChannel=16,
               auxiliary=False, cutout=False, dropPathProb=0.0))
    print('Time elapsed = {} mins'.format((time.time() - start)/60))
