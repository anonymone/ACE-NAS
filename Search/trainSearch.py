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
from Dataset import cifar100Search as Cifar100

import time
from misc import utils
from misc.flops_counter import add_flops_counting_methods

device = 'cuda'

# def main(code, epochs, save='SearchExp', exprRoot='./Experiments', seed=0, gpu=0, initChannel=24, modelLayers=11, auxiliary=False, cutout=False, dropPathProb=0.0):


def main(code, args, complement=False):
    # init parameters
    epochs = args.trainSearch_epoch
    save = args.trainSearch_save
    exprRoot = args.trainSearch_exprRoot
    seed = 0
    gpu = 0
    initChannel = args.trainSearch_initChannel
    auxiliary = args.trainSearch_auxiliary
    cutout = args.trainSearch_cutout
    dropPathProb = args.trainSearch_dropPathProb
    # ---- train logger ----------------- #
    utils.create_exp_dir(exprRoot)
    save_pth = os.path.join(exprRoot, '{}'.format(
        save.replace('#id', str(code.ID))))
    utils.create_exp_dir(save_pth)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')

    # ---- parameter values setting ----- #
    CIFAR_CLASSES = args.trainSearchDatasetClassNumber
    learning_rate = 0.025
    momentum = 0.9
    weight_decay = 3e-4
    data_root = args.dataRoot
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

    channels = [(3, initChannel)] + [((2**(i-1))*initChannel, (2**i)
                                      * initChannel) for i in range(1, len(code.getDec()))]

    model = layers.SEENetworkGenerator(
        code.getDec(), channels, CIFAR_CLASSES, (32, 32))

    # logging.info("Genome = %s", genome)
    logging.info("Architecture ID = %s", code.ID)
    logging.info("Code = %s", code.toString())

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

    if args.trainSearchDataset == "cifar10":
        train_data = Cifar10.CIFAR10(
            root=data_root, train=True, download=True, transform=train_transform)
        valid_data = Cifar10.CIFAR10(
            root=data_root, train=False, download=True, transform=valid_transform)
    elif args.trainSearchDataset == "cifar100":
        train_data = Cifar100.CIFAR100(
            root=data_root, train=True, download=True, transform=train_transform)
        valid_data = Cifar100.CIFAR100(
            root=data_root, train=False, download=True, transform=valid_transform)
    else:
        raise Exception("Parameter trainSearchDataset:{0} is invalid.".format(args.trainSearchDataset))

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

        train_err, train_obj = train(
            train_queue, model, criterion, optimizer, train_params)
        logging.info('train_err %f', train_err)

    valid_err, valid_obj = infer(valid_queue, model, criterion)
    logging.info('valid_error %f', valid_err)

    # calculate for flopss1
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
        file.write("ID = {}\n".format(code.ID))
        file.write("Architecture = {}\n".format(code.toString()))
        file.write("param size = {}MB\n".format(n_params))
        file.write("flops = {}MB\n".format(n_flops))
        file.write("valid_error = {}\n".format(valid_err))
    with open(os.path.join(save_pth, 'network.dot'), "w") as file:
        file.write(model.toDot())

    # logging.info("Architecture = %s", genotype))
    if complement:
        return {
            'valid_err': valid_err,
            'params': n_params,
            'flops': n_flops,
        }
    else:
        return {
            'valid_acc': 100.0-valid_err,
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

    return 100.0-(100.*correct/total), train_loss/total


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

    return 100.0-acc, test_loss/total


if __name__ == "__main__":
    parser = argparse.ArgumentParser("TEST")
    # train search method setting.
    parser.add_argument('--save', type=str, default='SEE_Exp',
                    help='experiment name')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--generation', type=int, default=30, help='random seed')

    # population setting
    parser.add_argument('--popSize', type=int, default=30,
                        help='The size of population.')
    parser.add_argument('--objSize', type=int, default=2,
                        help='The number of objectives.')
    parser.add_argument('--blockLength', type=tuple, default=(3, 15, 3),
                        help='A tuple containing (phase, unit number, length of unit)')
    parser.add_argument('--valueBoundary', type=tuple,
                        default=(0, 9), help='Decision value bound.')
    parser.add_argument('--crossoverRate', type=float, default=0.3,
                        help='The propability rate of crossover.')
    parser.add_argument('--mutationRate', type=float, default=1,
                        help='The propability rate of crossover.')
    # train search method setting.
    parser.add_argument('--trainSearch_epoch', type=int, default=30,
                        help='# of epochs to train during architecture search')
    parser.add_argument('--trainSearch_save', type=str,
                        default='SEE_#id', help='the filename including each model.')
    parser.add_argument('--trainSearch_exprRoot', type=str,
                        default='./Experiments/model', help='the root path of experiments.')
    parser.add_argument('--trainSearch_initChannel', type=int,
                        default=32, help='# of filters for first cell')
    parser.add_argument('--trainSearch_auxiliary',
                        type=bool, default=False, help='')
    parser.add_argument('--trainSearch_cutout', type=bool, default=False, help='')
    parser.add_argument('--trainSearch_dropPathProb',
                        type=float, default=0.0, help='')
    parser.add_argument('--dataRoot', type=str,
                        default='./Dataset', help='The root path of dataset.')
    parser.add_argument('--trainSearchDataset', type=str,
                        default='cifar10', help='The name of dataset.')
    parser.add_argument('--trainSearchDatasetClassNumber', type=int,
                        default=10, help='The classes number of dataset.')
    # testing setting
    parser.add_argument('--evalMode', type=str, default='EXP',
                        help='Evaluating mode for testing usage.')

    args = parser.parse_args()
    SEE_V3 = individual.SEEIndividual(2, (3, 10, 3))
    start = time.time()
    print(main(code=SEE_V3, args=args))
    print('Time elapsed = {} mins'.format((time.time() - start)/60))
