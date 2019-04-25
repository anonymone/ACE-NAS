import configparser
import sys
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import json
import logging

sys.path.append('../')
sys.path.append('../Population/')
sys.path.append('../Engine/')
sys.path.append('../EvolutionAlgorithm/')

config = configparser.ConfigParser()
config.read('./config.txt')

logging.basicConfig(filename='./logs/train.log',level=logging.DEBUG)

from individual import individual
from population import population
from StateControl import evaluator
from NSGA2 import NSGA2

EA = NSGA2.NSGA2(config)
Engine = evaluator(config)
pop = population(config=config)

Engine.initEngine()

# evaluate
for ind in pop.get_population():
    fitness = Engine.train(ind.get_dec())
    ind.set_fitness(fitness)

for i in range(int(config['EA setting']['runTimes'])):
    print("GENERATION {0}".format(i))
    logging.info("GENERATION {0}".format(i))
    population = pop.get_matrix()
    newpopMatrix = EA.mutate(population)
    newPop = np.vstack([population[0],newpopMatrix[0]])
    newGeneration = EA.enviromentalSeleection(pop= (newPop,(newPop.shape[0],population[1][1])),popNum=30)
    newPop = pop.matrix_to_Pop(newGeneration)
    pop.add_population(newPop)
    # evaluate
    for ind in pop.get_population():
        fitness = Engine.train(ind.get_dec())
        ind.set_fitness(fitness)
