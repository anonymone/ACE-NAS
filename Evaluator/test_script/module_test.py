import sys
sys.path.append('./')
import argparse
import time
import logging
import numpy as np
import torch
import random
import sys
import os
import glob
from copy import deepcopy
from quotes import Quotes

from Coder.ACE import build_ACE
from SearchEngine.EA_Engine import NSGA2, EA_population
from SearchEngine.Utils import EA_tools
from Evaluator.EA_evaluator import EA_eval
from Evaluator.Utils import recoder
from Coder.Network.utils import ACE_parser_tool

parser = argparse.ArgumentParser(
    "EA based Neural Architecture Search Experiments")
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--save_root', type=str, default='./Experiments/')
parser.add_argument('--generations', type=int, default=30)
# encoding setting
parser.add_argument('--unit_num', default=(15, 25))
parser.add_argument('--value_boundary', default=(0, 15))
# model setting
parser.add_argument('--layers', type=int, default=1)
parser.add_argument('--channels', type=int, default=16)
parser.add_argument('--keep_prob', type=float, default=0.6)
parser.add_argument('--drop_path_keep_prob', type=float, default=0.8)
parser.add_argument('--use_aux_head', type=bool, default=False)
parser.add_argument('--classes', type=int, default=10)
# population setting
parser.add_argument('--pop_size', type=int, default=30)
parser.add_argument('--obj_num', type=int, default=2)
parser.add_argument('--mutate_rate', type=float, default=0.5)
parser.add_argument('--crossover_rate', type=float, default=1)
# eval setting
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--mode', type=str, default='EXPERIMENT')
parser.add_argument('--data_path', type=str, default='./Res/Dataset/')
parser.add_argument('--cutout_size', type=int, default=None)
parser.add_argument('--num_work', type=int, default=0)
parser.add_argument('--train_batch_size', type=int, default=196)
parser.add_argument('--eval_batch_size', type=int, default=196)
parser.add_argument('--split_train_for_valid', type=float, default=0.9)
parser.add_argument('--small_set', type=float, default=0.2)
parser.add_argument('--l2_reg', type=float, default=3e-4)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--lr_min', type=float, default=0.001)
parser.add_argument('--lr_max', type=float, default=0.1)
parser.add_argument('--epochs', type=int, default=25)

# surrogate
parser.add_argument('--surrogate_allowed', type=recoder.args_bool, default='True')
parser.add_argument('--surrogate_path', type=str,
                    default='./Res/PretrainModel/')
parser.add_argument('--surrogate_premodel', type=str,
                    default='2020_03_10_09_29_34')
parser.add_argument('--surrogate_step', type=int, default=5)
parser.add_argument('--surrogate_search_times', type=int, default=10)
parser.add_argument('--surrogate_preserve_topk', type=int, default=5)

args = parser.parse_args()

recoder.create_exp_dir(args.save_root)
args.save_root = os.path.join(
    args.save_root, 'EA_SEARCH_{0}'.format(time.strftime("%Y%m%d-%H-%S")))
recoder.create_exp_dir(args.save_root, scripts_to_save=glob.glob('*_EA.*'))

torch.cuda.set_device(args.device)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

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

model = build_ACE(2, args)

evaluator = EA_eval(save_root=args.save_root,
                    mode=args.mode,
                    data_path=args.data_path,
                    cutout_size=args.cutout_size,
                    num_work=args.num_work,
                    train_batch_size=args.train_batch_size,
                    eval_batch_size=args.eval_batch_size,
                    split_train_for_valid=args.split_train_for_valid,
                    small_set=args.small_set,
                    l2_reg=args.l2_reg,
                    momentum=args.momentum,
                    lr_min=args.lr_min,
                    lr_max=args.lr_max,
                    epochs=args.epochs)

normal_graph = {
    0 :[],
    1:[0],
    2:[0],
    3:[0],
    4:[0],
    5:[0],
    6:[0],
    7:[0],
    8:[0],
    9:[0],
    10:[0]
}

reduction_graph = {
    0:[],
    1:[0],
    2:[0],
    3:[0],
    4:[0],
    5:[0],
    6:[0],
    7:[3,4],
    8:[5,6],
    9:[5,6],
    10:[0]
}

strings = '11.11.5-3.1.1-13.2.6-13.7.12-7.2.9-1.6.3-7.7.6-3.12.9-5.8.14-9.5.13-0.10.5-6.9.11-1.7.10-1.5.2-10.6.4<--->7.6.6-3.6.8-5.7.3-0.9.7-2.9.8-0.0.14-8.1.8-2.12.14-3.3.13-2.9.13-12.11.6-4.0.11-6.6.3-4.8.6-0.10.5-1.5.13-7.10.0-2.9.14-11.11.6-1.1.9-3.12.0-3.4.14-2.14.2'
model.set_dec(ACE_parser_tool.string_to_numpy(strings))
evaluator.evaluate([model])