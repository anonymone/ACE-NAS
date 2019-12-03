import argparse
import time
import logging
import numpy as np
import torch
import random
import sys
import os
from quotes import Quotes

from Coder.ACE import build_ACE
from SearchEngine.EA_Engine import NSGA2, EA_population
from SearchEngine.Utils import EA_tools
from Evaluator.EA_evaluator import EA_eval
from Evaluator.Utils import recoder

# Experiments parameter settings
parser = argparse.ArgumentParser(
    "EA based Neural Architecture Search Experiments")
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--save_root', type=str, default='./Experiments/')
parser.add_argument('--generations', type=int, default=30)
# encoding setting
parser.add_argument('--unit_num', default=(10, 20))
parser.add_argument('--value_boundary', default=(0, 15))
# model setting
parser.add_argument('--layers', type=int, default=3)
parser.add_argument('--channels', type=int, default=16)
parser.add_argument('--keep_prob', type=float, default=0.8)
parser.add_argument('--drop_path_keep_prob', type=float, default=0.8)
parser.add_argument('--use_aux_head', type=bool, default=True)
parser.add_argument('--classes', type=int, default=10)
# population setting
parser.add_argument('--pop_size', type=int, default=30)
parser.add_argument('--obj_num', type=int, default=2)
parser.add_argument('--mutate_rate', type=float, default=1)
parser.add_argument('--crossover_rate', type=float, default=0.8)
# eval setting
parser.add_argument('--mode', type=str, default='DEBUG')
parser.add_argument('--data_path', type=str, default='./Dataset/')
parser.add_argument('--cutout_size', type=int, default=16)
parser.add_argument('--num_work', type=int, default=10)
parser.add_argument('--train_batch_size', type=int, default=96)
parser.add_argument('--eval_batch_size', type=int, default=96)
parser.add_argument('--split_train_for_valid', type=float, default=None)
parser.add_argument('--l2_reg', type=float, default=3e-4)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--lr_min', type=float, default=0)
parser.add_argument('--lr_max', type=float, default=0.025)
parser.add_argument('--epochs', type=int, default=30)

args = parser.parse_args()

recoder.create_exp_dir(args.save_root)
args.save_root = os.path.join(args.save_root, 'EA_SEARCH_{0}'.format(time.strftime("%Y%m%d-%H")))
recoder.create_exp_dir(args.save_root)

# logging setting
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s %(message)s")
fh = logging.FileHandler(os.path.join(args.save_root, 'experiments.log'))
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.addHandler(fh)

# record settings
logging.info("[Experiments Setting]\n"+"".join(["[{0}]: {1}\n".format(name, value) for name, value in args.__dict__.items()]))

# fix seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

population = EA_population(obj_number=args.obj_num,
                           pop_size=args.pop_size,
                           mutation_rate=args.mutate_rate,
                           crossover_rate=args.crossover_rate,
                           mutation=EA_tools.ACE_Mutation_V2,
                           crossover=EA_tools.ACE_CrossoverV1,
                           ind_generator=build_ACE,
                           ind_params=args)

evaluator = EA_eval(save_root=args.save_root,
                    mode=args.mode,
                    data_path=args.data_path,
                    cutout_size=args.cutout_size,
                    num_work=args.num_work,
                    train_batch_size=args.train_batch_size,
                    eval_batch_size=args.eval_batch_size,
                    split_train_for_valid=args.split_train_for_valid,
                    l2_reg=args.l2_reg,
                    momentum=args.momentum,
                    lr_min=args.lr_min,
                    lr_max=args.lr_max,
                    epochs=args.epochs)
engine = NSGA2

# Expelliarmus
q = Quotes()

for gen in range(args.generations):
    logging.info("[Generation{0:>2d}] {2} -- {1}".format(gen, *q.random()) )
    population.new_pop()
    evaluator.evaluate(population.get_ind())
    _, rm_inds = engine.enviromentalSeleection(population.to_matrix(), args.pop_size)
    logging.info('[Removed Individuals]\n'+"".join([str(i)+'\n' for i in rm_inds]))
    population.remove_ind(rm_inds)
    population.save(save_path=os.path.join(args.save_root,'populations'), file_name='population_{0:_>2d}'.format(gen))
