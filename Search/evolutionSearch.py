import sys
# update your projecty root path before running
sys.path.insert(0, './')

import os
import time
import logging
import argparse

from misc import utils
from misc import evo_operator
import numpy as np
from Search import trainSearch
from EvolutionAlgorithm.NSGA2 import NSGA2
from Model.individual import SEEPopulation

parser = argparse.ArgumentParser("Multi-objetive Genetic Algorithm for SEENAS")
parser.add_argument('--save', type=str, default='SEE_Exp',
                    help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--generation', type=int, default=30, help='random seed')

# population setting
parser.add_argument('--popSize', type=int, default=30,help='The size of population.')
parser.add_argument('--objSize', type=int, default=2,help='The number of objectives.')
parser.add_argument('--blockLength', type=tuple, default=(3, 12, 3),help='A tuple containing (phase, unit number, length of unit)')
parser.add_argument('--valueBoundary', type=tuple, default=(0,9),help='Decision value bound.')
parser.add_argument('--crossoverRate', type=float, default=0.2,help='The propability rate of crossover.')
# train search method setting.
parser.add_argument('--trainSearch_epoch', type=int, default=25,help='# of epochs to train during architecture search')
parser.add_argument('--trainSearch_save', type=str, default='SEE_#id', help='the filename including each model.')
parser.add_argument('--trainSearch_exprRoot', type=str, default='./Experiments/model', help='the root path of experiments.')
parser.add_argument('--trainSearch_initChannel', type=int, default=24, help='# of filters for first cell')
parser.add_argument('--trainSearch_auxiliary', type=bool, default=False, help='')
parser.add_argument('--trainSearch_cutout', type=bool, default=True, help='')
parser.add_argument('--trainSearch_dropPathProb', type=bool, default=False, help='')
parser.add_argument('--dataRoot', type=str, default='./Dataset', help='The root path of dataset.')
# testing setting
parser.add_argument('--evalMode', type=str, default='EXP', help='Evaluating mode for testing usage.')

args = parser.parse_args()
args.save = './Experiments/search-{}-{}'.format(
    args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

pop_hist = []  # keep track of every evaluated architecture

# init population
Engine = NSGA2.NSGA2()
population = SEEPopulation(popSize=args.popSize, crossover=evo_operator.SEECrossoverV1,
                           objSize=args.objSize, blockLength=args.blockLength,
                           valueBoundary=args.valueBoundary, mutation=evo_operator.SEEMutationV1,
                           evaluation=trainSearch.main, args=args)

# evaluation
population.evaluation()
population.save(os.path.join(args.save,'Generation-{0}'.format('init')))

for generation in range(args.generation):
    logging.info("=======================Generatiion {0}=======================".format(generation))
    population.newPop()
    population.evaluation()
    popValue = population.toMatrix()
    index = Engine.enviromentalSeleection(popValue,args.popSize)
    index2 = [x for x in range(population.popSize) if x not in index]
    population.remove(index2)
    population.save(os.path.join(args.save,'Generation-{0}'.format(generation)))
    