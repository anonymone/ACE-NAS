import sys
sys.path.append('./')
import argparse
import numpy as np

from Evaluator.Utils.dataset import build_cifar10
from Evaluator.Utils.train import train, valid, build_train_utils
from Evaluator.EA_evaluator import EA_eval
from Coder.ACE import ACE

epochs = 1

ind = ACE()
eval_tool = EA_eval(num_work=0, epochs=1, train_batch_size=96)
eval_tool.evaluate([ind])