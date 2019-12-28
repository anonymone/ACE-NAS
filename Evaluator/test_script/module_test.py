import sys
sys.path.append('./')
import argparse
import numpy as np
import logging
import os

from Coder.ACE import build_ACE
from SearchEngine.Utils import EA_tools
from Evaluator.EA_evaluator import EA_eval
from Evaluator.Utils import recoder

from Evaluator.Utils.dataset import build_cifar10, auto_data
from Evaluator.Utils.train import train, valid, build_train_utils
from Coder.ACE import ACE
from SearchEngine.EA_Engine import EA_population

from Evaluator.RL_evaluator import RL_eval
from Evaluator.Utils.surrogate import auto_seq2seq
from Evaluator.Utils.recoder import create_exp_dir

from Evaluator.Utils.surrogate import EmbeddingModel as em
from Evaluator.Utils.surrogate import RankNetDataset, Seq2Rank

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
parser.add_argument('--layers', type=int, default=1)
parser.add_argument('--channels', type=int, default=24)
parser.add_argument('--keep_prob', type=float, default=0.6)
parser.add_argument('--drop_path_keep_prob', type=float, default=0.8)
parser.add_argument('--use_aux_head', type=bool, default=False)
parser.add_argument('--classes', type=int, default=10)
# population setting
parser.add_argument('--pop_size', type=int, default=30)
parser.add_argument('--obj_num', type=int, default=2)
parser.add_argument('--mutate_rate', type=float, default=1)
parser.add_argument('--crossover_rate', type=float, default=0.8)
# eval setting
parser.add_argument('--mode', type=str, default='DEBUG')
parser.add_argument('--data_path', type=str, default='./Dataset/')
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

# create_exp_dir('./Experiments/')
# create_exp_dir('./Experiments/module_test/')

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s %(message)s")
fh = logging.FileHandler(os.path.join('./Experiments/module_test/', 'experiments.log'))
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.addHandler(fh)

population = EA_population(obj_number=args.obj_num,
                           pop_size=args.pop_size,
                           mutation_rate=args.mutate_rate,
                           crossover_rate=args.crossover_rate,
                           mutation=EA_tools.ACE_Mutation_V2,
                           crossover=EA_tools.ACE_CrossoverV1,
                           ind_generator=build_ACE,
                           ind_params=args)
eval_tool = EA_eval(num_work=0, epochs=1, train_batch_size=32, save_root='./Experiments/test_module/')
eval_tool.set_mode('DEBUG')
eval_tool.evaluate(population.get_ind())
# population.save(save_path='./Experiments/test_module/')

# ind = ACE(1,channels=16)
# eval_tool = RL_eval(num_work=0, split_train_for_valid=0.4,train_batch_size=36,epochs=1)
# eval_tool.set_mode("DEBUG")
# result = eval_tool.evaluate(ind)
# print(result)

# model = auto_seq2seq('./Res/PretrainModel/', './Experiments/module_test/','./Experiments/module_test/')
# model.train()

# auto_data.save_data(auto_data.generate_data(num_samples=100), file_name='train')
# auto_data.save_data(auto_data.generate_data(num_samples=100), file_name='eval')


encoder = em(model_path='./Res/PretrainModel/', model_file='2019_12_28_06_03_12')
model = Seq2Rank(encoder, 
                model_save_path='./Experiments/test_module/',
                input_preprocess=lambda x: x.replace('-', ' ').replace('<   >', ' <---> '))
data = [(model.input_preprocess(ind.to_string()), ind.get_fitness()[0]) for ind in population.get_ind()]
model.update_dataset(data)
model.trian()
model.evaluation(population.get_ind())
print(population.pop_size)