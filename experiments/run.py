import configparser
import sys
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np

sys.path.append('./')
sys.path.append('./Population/')
sys.path.append('./Engine/')
sys.path.append('./EvolutionAlgorithm/')

config = configparser.ConfigParser()
config.read('./Experiments/config.txt')

from individual import individual
from population import population
from StateControl import evaluator
from NSGA2 import NSGA2

EA = NSGA2.NSGA2(config)

pop = population(config=config)

population = pop.get_matrix()

newpop = EA.mutate(population)

a = newpop[0][6, :]

b = population[0][6, :]
EA.crossOver(ind1=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]),
             ind2=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]))

a = np.random.randint(9, size=(3, 2))
a = np.array([[3, 8],
       [7, 1],
       [4, 3],
       [1, 5],
       [8, 4],
       [1, 6],
       [4, 0],
       [7, 0],
       [1, 3],
       [8, 8]])
F = EA.fastNondomiatedSort((a, (3, 2)))

EA.isDominated(a[1], a[2])
