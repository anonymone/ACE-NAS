import sys
sys.path.insert(0, './')

import torch 
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision

import os 
import sys
import time
import glob
import logging
import argparse
import numpy as np
import random   

from misc import utils
from Model import individual, NAOlayer, Network_Constructor

parser = argparse.ArgumentParser(description='Final Validation of Searched Architecture')
parser.add_argument('--save', type=str, default='ValidationCifar10', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--data_worker', type=int, default=4, help='the number of the data worker.')
parser.add_argument('--data', type=str, default='./Dataset', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', help='the dataset: cifar10, cifar100 ...')
parser.add_argument('--eport', type=str, help='the path to save the output file.')

parser.add_argument('--search_space', default='Node_Cell', type=str)
parser.add_argument('--code_str', type=str, default='Phase:565-942-627-465-742-441-262-663-208-711-065-861-284-788-325-Phase:645-703-557-802-760-512-294-012-232-585-161-526-622-286-341')
parser.add_argument('--layers', default=6, type=int, help='total number of layers (equivalent w/ N=6)')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')

parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--eval_batch_size', type=int, default=250, help='eval batch size')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--min_learning_rate', type=float, default=0.0, help='minimum learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--keep_prob', type=float, default=0.6)
parser.add_argument('--drop_path_keep_prob', type=float, default=0.8)
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')

parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--cutout', action='store_true', default=True, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--auxiliary', action='store_true', default=True, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
# parser.add_argument('--droprate', default=0, type=float, help='dropout probability (default: 0.0)')

parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
args = parser.parse_args()

args.save = './Experiments/{0}-{1}'.format(args.save, time.strftime("%Y-%m-%d-%m"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('./Validation/*Cifar*.py'))

device = 'cuda' if torch.cuda.is_available() else 'cpu'

log_format = '%(asctime)s %(levelname)s %(message)s'
logging.basicConfig(stream=sys.stdout,
                    level=logging.INFO, 
                    format=log_format, 
                    datefmt='%m/%d %I:%M:%S')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

def main():
    if not torch.cuda.is_available():
        logging.warn('no gpu device available!')

    logging.info("args = %s", args)

    cudnn.enabled = True
    cudnn.benchmark = False
    cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    best_err = 100  # initiate a artificial best error so far

    # Data
    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    if args.dataset == 'cifar10':
        train_data = torchvision.datasets.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
        valid_data = torchvision.datasets.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)
        class_num = 10
    elif args.dataset == 'cifar100':
        train_data = torchvision.datasets.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
        valid_data = torchvision.datasets.CIFAR100(root=args.data, train=False, download=True, transform=valid_transform)
        class_num = 100

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.data_worker)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.eval_batch_size, shuffle=False, pin_memory=True, num_workers=args.data_worker)

    indDec = args.code_str

    # Model
    ind = individual.SEEIndividual(objSize=2, blockLength=(2,15,3))
    logging.info("Code dec: {0}".format(indDec))
    indDec = indDec.replace('Phase:',"").split('-')
    code = []
    for unit in indDec:
        for bit in unit:
            code.append(int(bit))
    ind.setDec(code)

    initChannel = args.init_channels
    # channels = [(3, initChannel)] + [((2**(i-1))*initChannel, (2**i)
    #                                 * initChannel) for i in range(1, len(ind.getDec()))]

    steps = int(np.ceil(50000 / args.batch_size)) * args.epochs

    if args.search_space == 'NAO_Cell':
        net = NAOlayer.SEEArchitecture(args=args,
                                        code= ind.getDec(), 
                                        classes=class_num,
                                        layers=args.layers,
                                        channels=initChannel,
                                        keepProb=args.keep_prob, 
                                        dropPathKeepProb=args.drop_path_keep_prob,
                                        useAuxHead=args.auxiliary, 
                                        steps=steps)
    elif args.search_space == 'Node_Cell':
        net = Network_Constructor.Node_based_Network_cifar(args=args,
                                                            code= ind.getDec(), 
                                                            cell_type='node',
                                                            classes=class_num,
                                                            layers=args.layers,
                                                            channels=initChannel,
                                                            keep_prob=args.keep_prob, 
                                                            drop_path_keep_prob=args.drop_path_keep_prob,
                                                            use_aux_head=args.auxiliary, 
                                                            steps=steps)
    else:
        raise Exception('Search space {0} is vailed.'.format(args.search_space))


    # logging.info("{}".format(net))
    logging.info("param size = %fMB", utils.count_parameters_in_MB(net))

    net = net.to(device)

    n_epochs = args.epochs

    train_criterion = nn.CrossEntropyLoss().to(device)
    eval_criterion = nn.CrossEntropyLoss().to(device)

    parameters = filter(lambda p: p.requires_grad, net.parameters())
    optimizer = optim.SGD(parameters,
                          lr=args.learning_rate,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs, eta_min=args.min_learning_rate)
    step = 0
    for epoch in range(n_epochs):
        scheduler.step()
        logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
        # net.droprate = args.droprate * epoch / args.epochs

        train_loss, train_err, step = train(train_queue, net, train_criterion, optimizer, step)
        _, valid_err = infer(valid_queue, net, eval_criterion)

        if valid_err < best_err:
            utils.save(net, os.path.join(args.save, 'weights_err_{0}.pt'.format(train_err)))
            best_err = valid_err
    logging.info("The best Test Error: {0}".format(best_err))


# Training
def train(train_queue, net, criterion, optimizer, global_step):
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for step, (inputs, targets) in enumerate(train_queue):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs, outputs_aux = net(inputs, global_step)
        global_step += 1
        loss = criterion(outputs, targets)

        if args.auxiliary:
            loss_aux = criterion(outputs_aux, targets)
            loss += args.auxiliary_weight * loss_aux
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if step % args.report_freq == 0:
            logging.info('train %03d %e %f', step, train_loss/total, 100.-(100.*correct/total))

    logging.info('train err %f', 100.-(100.*correct/total))

    return train_loss/total, 100.-(100.*correct/total), global_step


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
            #     logging.info('valid %03d %e %f', step, test_loss/total, 100.-(100.*correct/total))

    acc = 100.-(100.*correct/total)
    logging.info('valid err %f', 100.-(100.*correct/total))

    return test_loss/total, acc


if __name__ == '__main__':
    main()