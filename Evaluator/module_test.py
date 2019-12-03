import sys
sys.path.append('./')
import argparse
import numpy as np

from Evaluator.Utils.dataset import build_cifar10
from Evaluator.Utils.train import train, valid, build_train_utils
from Evaluator.EA_evaluator import EA_eval
from Coder.ACE import ACE
from SearchEngine.EA_Engine import EA_population

population = EA_population(2, 30, ind_generator=ACE)
eval_tool = EA_eval(num_work=0, epochs=1, train_batch_size=32, save_root='./Experiments/test_module/')
eval_tool.set_mode('DEBUG')
eval_tool.evaluate(population.get_ind())
population.save(save_path='./Experiments/test_module/')