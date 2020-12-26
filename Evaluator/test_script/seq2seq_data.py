import sys
sys.path.append('./')
from SearchEngine.EA_Engine import EA_population
from SearchEngine.Utils.EA_tools import ACE_Mutation_V3,ACE_CrossoverV2
from Coder.ACE import build_ACE

import random 
import numpy as np
import argparse

# Experiments parameter settings
parser = argparse.ArgumentParser(
    "EA based Neural Architecture Search Experiments")
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--save_root', type=str, default='./Experiments/')
parser.add_argument('--generations', type=int, default=30)
# encoding setting
parser.add_argument('--unit_num', default=(8, 16))
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
parser.add_argument('--mode', type=str, default='DEBUG')
parser.add_argument('--data_path', type=str, default='./Res/Dataset/')
parser.add_argument('--cutout_size', type=int, default=None)
parser.add_argument('--num_work', type=int, default=0)
parser.add_argument('--train_batch_size', type=int, default=196)
parser.add_argument('--eval_batch_size', type=int, default=196)
parser.add_argument('--split_train_for_valid', type=float, default=None)
parser.add_argument('--l2_reg', type=float, default=3e-4)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--lr_min', type=float, default=0.001)
parser.add_argument('--lr_max', type=float, default=0.025)
parser.add_argument('--epochs', type=int, default=25)

args = parser.parse_args()

def code_init():
    dec = np.array([[np.random.randint(0, 4) if random.random() < 0.8 else np.random.randint(0, 7), np.random.randint(
        low=args.value_boundary[0], high=args.value_boundary[1]), np.random.randint(low=args.value_boundary[0], high=args.value_boundary[1])] for i in range(random.randint(args.unit_num[0], args.unit_num[1]))])
    return dec

def to_string(normal_dec, reduct_dec,callback=None) -> str:
    normal_string = "-".join([".".join([str(t) for t in unit])
                                for unit in normal_dec])
    reduct_string = "-".join([".".join([str(t) for t in unit])
                                for unit in reduct_dec])
    if callback is None:
        return "<--->".join([normal_string, reduct_string])
    else:
        return callback("<--->".join([normal_string, reduct_string]))

file_train = open('./Res/PretrainModel/train','w')
for i, d in enumerate([to_string(code_init(), code_init(), callback=lambda x: x.replace('-',' ').replace('<   >', ' <---> ')) for i in range(4000000)]):
    file_train.writelines('{0}\t{0}\n'.format(d))
    if i%1000==0:
        print(i)
    if i %99999==0:
        file_train.flush()
file_train.flush

file_train = open('./Res/PretrainModel/eval','w')
for i, d in enumerate([to_string(code_init(), code_init(), callback=lambda x: x.replace('-',' ').replace('<   >', ' <---> ')) for i in range(1000000)]):
    file_train.writelines('{0}\t{0}\n'.format(d))
    if i%1000==0:
        print(i)
    if i %99999==0:
        file_train.flush()
file_train.flush