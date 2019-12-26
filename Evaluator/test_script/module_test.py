import sys
sys.path.append('./')
import argparse
import numpy as np
import logging
import os

from Evaluator.Utils.dataset import build_cifar10, auto_data
from Evaluator.Utils.train import train, valid, build_train_utils
from Evaluator.EA_evaluator import EA_eval
from Coder.ACE import ACE
from SearchEngine.EA_Engine import EA_population

from Evaluator.RL_evaluator import RL_eval
from Evaluator.Utils.surrogate import auto_seq2seq
from Evaluator.Utils.recoder import create_exp_dir

# create_exp_dir('./Experiments/')
# create_exp_dir('./Experiments/module_test/')

# logger = logging.getLogger()
# logger.setLevel(logging.DEBUG)
# formatter = logging.Formatter("%(asctime)s %(message)s")
# fh = logging.FileHandler(os.path.join('./Experiments/module_test/', 'experiments.log'))
# fh.setLevel(logging.INFO)
# fh.setFormatter(formatter)
# ch = logging.StreamHandler()
# ch.setLevel(logging.DEBUG)
# ch.setFormatter(formatter)
# logger.addHandler(ch)
# logger.addHandler(fh)

# population = EA_population(2, 30, ind_generator=ACE)
# eval_tool = EA_eval(num_work=0, epochs=1, train_batch_size=32, save_root='./Experiments/test_module/')
# eval_tool.set_mode('DEBUG')
# eval_tool.evaluate(population.get_ind())
# population.save(save_path='./Experiments/test_module/')

# ind = ACE(1,channels=16)
# eval_tool = RL_eval(num_work=0, split_train_for_valid=0.4,train_batch_size=36,epochs=1)
# eval_tool.set_mode("DEBUG")
# result = eval_tool.evaluate(ind)
# print(result)

# model = auto_seq2seq('./Res/PretrainModel/', './Experiments/module_test/','./Experiments/module_test/')
# model.train()

auto_data.save_data(auto_data.generate_data(num_samples=100))