import configparser
import sys
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
sys.path.append('../')
sys.path.append('../Population/')
sys.path.append('../Engine/')

config = configparser.ConfigParser()
config.read('./config.txt')

from individual import individual
from population import population
from StateControl import evaluator

pop = population(config=config)

config.keys()

Eval = evaluator(config)
Eval.initEngine()

ind = pop.get_population()
len(ind)

for i in range(30):
    Eval.insert(ind[i])

for i in range(30):
    Eval.insert('None')