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
import time
#from quotes import Quotes

from Coder.ACE import ACE
from Coder.Network.utils import ACE_parser_tool
from Evaluator.Utils.train import train, valid, build_train_utils
from Evaluator.Utils.dataset import build_imagenet
from Evaluator.Utils.recoder import create_exp_dir, model_save, count_parameters, count_parameters

parser = argparse.ArgumentParser(description='Final Validation of Searched Architecture on ImageNet')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--save_root', type=str,default='./Experiments/', help='experiment name')
parser.add_argument('--encoding_str', type=str,default="3.0.7-7.2.1-7.0.7-0.4.9-9.0.3-8.1.4<--->4.3.7-7.0.6-9.1.1-9.9.2-1.1.9-8.5.4-8.1.7-5.0.9-0.9.2")

parser.add_argument('--data_path', type=str, default='./Res/Dataset/', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='ImageNet',help='the dataset: ImageNet ...')
parser.add_argument('--feed_num_work', type=int, default=0,help='the number of the data worker.')
parser.add_argument('--load_num_work', type=int, default=32, help='the number of the data worker.')
parser.add_argument('--zip_file', type=bool, default=False)
parser.add_argument('--lazy_load', type=bool, default=False)
parser.add_argument('--train_batch_size', type=int,default=96, help='batch size')
parser.add_argument('--eval_batch_size', type=int,default=36, help='eval batch size')

parser.add_argument('--layers', default=1, type=int,help='total number of layers (equivalent w/ N=6)')
parser.add_argument('--channels', type=int, default=52,help='num of init channels')

parser.add_argument('--epochs', type=int, default=1,help='num of training epochs')
parser.add_argument('--device', type=str, default="cuda")
parser.add_argument('--grad_clip', type=float, default=5.0)
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--gamma', type=float, default=0.97, help='learning rate decay')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--lr_max', type=float, default=0.025,help='init learning rate')
parser.add_argument('--keep_prob', type=float, default=0.6)
parser.add_argument('--drop_path_keep_prob', type=float, default=0.8)
parser.add_argument('--l2_reg', type=float, default=3e-5)
parser.add_argument('--decay_period', type=int, default=1, help='epochs between two learning rate decays')
parser.add_argument('--use_aux_head', action='store_true',default=True, help='use auxiliary tower')

# You can train imagenet on N GPUs using the train_NAONet_V2_imagenet.sh script with --batch_size=128*$N and --lr=0.1*$N


args = parser.parse_args()

create_exp_dir(args.save_root)
args.save_root = os.path.join(
    args.save_root, 'FINAL_{0}_{1}'.format(args.dataset, time.strftime("%Y%m%d-%H")))
create_exp_dir(args.save_root, scripts_to_save=glob.glob('*_ImageNet.*'))

# logging setting
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s %(message)s")
fh = logging.FileHandler(os.path.join(args.save_root, 'experiments.log'))
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.addHandler(fh)

# record settings
logging.info("[Experiments Setting]\n"+"".join(
    ["[{0}]: {1}\n".format(name, value) for name, value in args.__dict__.items()]))

# fix seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

cudnn.enabled = True
cudnn.benchmark = True

trainset, validset = build_imagenet(data_path=args.data_path,
                                    load_num_work=args.load_num_work,
                                    feed_num_work=args.feed_num_work,
                                    zip_file=args.zip_file, lazy_load=args.lazy_load,
                                    train_batch_size=args.train_batch_size,
                                    valid_batch_size=args.eval_batch_size)

steps = int(len(trainset)) * args.epochs
sample = ACE(fitness_size=2,
             classes=1000,
             layers=args.layers,
             channels=args.channels,
             keep_prob=args.keep_prob,
             drop_path_keep_prob=args.drop_path_keep_prob,
             use_aux_head=args.use_aux_head)
sample.set_dec(ACE_parser_tool.string_to_numpy(args.encoding_str))
model = sample.get_model(steps=steps, imagenet=True)

n_params = count_parameters(model)

if torch.cuda.device_count()>1:
    logging.info("Use %d %s", torch.cuda.device_count(), "GPUs !")
    model = nn.DataParallel(model)

train_criterion, eval_criterion, optimizer, scheduler = build_train_utils(model=model,
                                                                          l2_reg=args.l2_reg,
                                                                          momentum=args.momentum,
                                                                          lr_max=args.lr_max,
                                                                          decay_period=args.decay_period,
                                                                          gamma=args.gamma,
                                                                          label_smooth=args.label_smooth,
                                                                          epochs=args.epochs)

# Expelliarmus
#q = Quotes()
logging.info("[Initialize Model] Model size{1:.2f}M Model Encoding string {0}\n".format(sample.to_string()))

# train procedure
step = 0
s_time = 0
total_time = 0
error_best = 100.0
device = args.device
for epoch in range(args.epochs):
    # time cost record
    s_time = time.time()

    train_loss, train_top1, train_top5, step = train(
        trainset, model, optimizer, step, train_criterion, device)
    logging.debug("[Epoch {0:>5d}] [Train] loss {1:.3f} lr {2:.5f} error Top1 {3:.2f} error Top5 {4:.2f}".format(
        epoch, train_loss, scheduler.get_lr()[0], train_top1, train_top5))
    scheduler.step()
    # valid procdure
    valid_loss, valid_top1, valid_top5 = valid(
        validset, model, eval_criterion, device)
    logging.info("[Valid Error] loss {0:.3f} error Top1 {1:.2f} error Top5 {2:.2f} Params {3:.2f}M".format(
        valid_loss, valid_top1, valid_top5, n_params))
    # save model
    if error_best > valid_top1:
        error_best = valid_top1
        model_save(model, os.path.join(args.save_root, "checkpoint"),
                   "model_{0:.3f}".format(valid_top1))

    s_time = (time.time() - s_time)/60.0
    total_time += s_time
    logging.info("[Train Epoch] [{0:>3d}/{1:>3d}] time cost {2:.2f}mins total time cost {3:.2f}h time left {4:.2f}h".format(
        epoch+1, args.epochs, s_time, total_time/60.0, ((total_time/(epoch+1))*(args.epochs-epoch-1))/60.0))
