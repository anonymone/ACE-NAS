import sys
sys.path.append('./')
from SearchEngine.EA_Engine import EA_population
from SearchEngine.Utils.EA_tools import ACE_Mutation_V3,ACE_CrossoverV2
from Coder.ACE import build_ACE

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


population = EA_population(obj_number=2,
                           pop_size=3000000,
                           mutation_rate=1,
                           crossover_rate=1,
                           mutation=ACE_Mutation_V3,
                           crossover=ACE_CrossoverV2,
                           ind_generator=build_ACE,
                           ind_params=args)

file_train = open('./Res/PretrainModel/train','w')
for i, d in enumerate(population.to_string(callback=lambda x: x.replace('-',' ').replace('<   >', ' <---> '))):
    file_train.writelines('{0}\t{0}\n'.format(d))
    if i %999==0:
        file_train.flush()
file_train.flush

population = EA_population(obj_number=2,
                           pop_size=2000000,
                           mutation_rate=1,
                           crossover_rate=1,
                           mutation=ACE_Mutation_V3,
                           crossover=ACE_CrossoverV2,
                           ind_generator=build_ACE,
                           ind_params=args)

file_train = open('./Res/PretrainModel/eval','w')
for i, d in enumerate(population.to_string(callback=lambda x: x.replace('-',' ').replace('<   >', ' <---> '))):
    file_train.writelines('{0}\t{0}'.format(d))
    if i %999==0:
        file_train.flush()
file_train.flush