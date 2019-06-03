import sys
# update your projecty root path before running
sys.path.insert(0, './')

import os
import time
import logging
import argparse
from misc import utils

import numpy as np
from Search import trainSearch
from EvolutionAlgorithm.NSGA2 import NSGA2
from Model.individual import SEEPopulation
from

parser = argparse.ArgumentParser("Multi-objetive Genetic Algorithm for SEENAS")
parser.add_argument('--save', type=str, default='SEE Experiments',
                    help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
# train search method setting.
parser.add_argument('--trainSearch_epoch', type=int, default=25,help='# of epochs to train during architecture search')
parser.add_argument('--trainSearch_save', type=str, default='SEE_#Generation_#id', help='the filename including each model.')
parser.add_argument('--trainSearch_exprRoot', type=str, default='./Experiments', help='the root path of experiments.')
parser.add_argument('--trainSearch_initChannel', type=int, default=24, help='# of filters for first cell')
parser.add_argument('--trainSearch_auxiliary', type=bool, default=False, help='')
parser.add_argument('--trainSearch_cutout', type=bool, default=False, help='')
parser.add_argument('--trainSearch_dropPathProb', type=bool, default=False, help='')
parser.add_argument('--dataRoot', type=str, default='./Dataset', help='The root path of dataset.')

args = parser.parse_args()
args.save = './Experiments/SEESearch/search-{}-{}-{}'.format(
    args.save, args.search_space, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

pop_hist = []  # keep track of every evaluated architecture

# init population
Engine = NSGA2()
population = SEEPopulation(popSize=arg.popSize, decSize=arg.decSize,
                           objSize=arg.objSize, blockLength=arg.blockLength,
                           valueBoundary=arg.valueBoundary, mutation=None,
                           evaluation=trainSearch.main, arg=arg)
population.eval()
population.save(./Experiments)
population.newPop()
popValue = population.toMatrix()
index = Engine.enviromentalSeleection(popValue,arg.popSize)

