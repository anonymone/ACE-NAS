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
import logging
import argparse
import numpy as np

from misc import utils
from Model import individual, layers

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--data', type=str, default='./Dataset', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--min_learning_rate', type=float, default=0.0, help='minimum learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--save', type=str, default='ValidationCifar100', help='experiment name')
parser.add_argument('--cutout', action='store_true', default=True, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--layers', default=20, type=int, help='total number of layers (equivalent w/ N=6)')
parser.add_argument('--droprate', default=0, type=float, help='dropout probability (default: 0.0)')
parser.add_argument('--init_channels', type=int, default=32, help='num of init channels')
parser.add_argument('--arch', type=str, default='NSGANet', help='which architecture to use')
parser.add_argument('--filter_increment', default=4, type=int, help='# of filter increment')
parser.add_argument('--SE', action='store_true', default=False, help='use Squeeze-and-Excitation')
parser.add_argument('--net_type', type=str, default='micro', help='(options)micro, macro')
args = parser.parse_args()
args.save = './Experiments/train-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save)

device = 'cuda'

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    # if args.auxiliary and args.net_type == 'macro':
    #     logging.info('auxiliary head classifier not supported for macro search space models')
    #     sys.exit(1)

    logging.info("args = %s", args)

    cudnn.enabled = True
    cudnn.benchmark = True
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    best_err = 0  # initiate a artificial best accuracy so far

    # Data
    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    train_data = torchvision.datasets.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
    valid_data = torchvision.datasets.CIFAR100(root=args.data, train=False, download=True, transform=valid_transform)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=0)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=128, shuffle=False, pin_memory=True, num_workers=0)

    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    ind = individual.SEEIndividual(objSize=2, blockLength=(3,10,3))
    indDec = 'Phase:140-784-356-328-051-835-517-755-735-364-Phase:841-537-449-083-632-703-545-124-021-474-Phase:452-723-657-147-288-765-652-712-258-422'
    logging.info("Code dec: {0}".format(indDec))
    indDec = indDec.replace('Phase:',"").split('-')
    code = []
    for unit in indDec:
        for bit in unit:
            code.append(int(bit))
    ind.setDec(code)
    net = layers.SEENetworkGenerator(ind.getDec(), [[3,128],[128,128],[128,128]],100,(32,32))

    # logging.info("{}".format(net))
    logging.info("param size = %fMB", utils.count_parameters_in_MB(net))

    net = net.to(device)

    n_epochs = args.epochs

    parameters = filter(lambda p: p.requires_grad, net.parameters())

    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    optimizer = optim.SGD(parameters,
                          lr=args.learning_rate,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs, eta_min=args.min_learning_rate)

    for epoch in range(n_epochs):
        scheduler.step()
        logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
        # net.droprate = args.droprate * epoch / args.epochs

        train(train_queue, net, criterion, optimizer)
        _, valid_err = infer(valid_queue, net, criterion)

        if valid_err < best_err:
            utils.save(net, os.path.join(args.save, 'weights.pt'))
            best_err = valid_err
    logging.info("The best Test Error: {0}".format())

# Training
def train(train_queue, net, criterion, optimizer):
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for step, (inputs, targets) in enumerate(train_queue):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs, outputs_aux = net(inputs)
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

    logging.info('train err %f', 100. * correct / total)

    return train_loss/total, 100.*correct/total


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

            if step % args.report_freq == 0:
                logging.info('valid %03d %e %f', step, test_loss/total, 100.*correct/total)

    acc = 100.-(100.*correct/total)
    logging.info('valid err %f', 100. * correct / total)

    return test_loss/total, acc


if __name__ == '__main__':
    main()
