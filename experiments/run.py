import configparser
import sys
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import json
import logging

sys.path.append('./')
sys.path.append('./Population/')
sys.path.append('./Engine/')
sys.path.append('./EvolutionAlgorithm/')

config = configparser.ConfigParser()
config.read('./experiments/config.txt')

logging.basicConfig(filename='./experiments/logs/train.log',level=logging.DEBUG)

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
    fitness = Engine.train(ind.get_dec(),Mode='None')
    ind.set_fitness(fitness)

for i in range(int(config['EA setting']['runTimes'])):
    print("GENERATION {0}".format(i))
    logging.info("GENERATION {0}".format(i))
    population = pop.get_matrix()
    newpopMatrix = EA.mutate(population)
    newPop = pop.matrix_to_Pop(newpopMatrix)
    newPop.extend(pop.get_population())
    # evaluate
    for ind in newPop:
        if ind.isTrained():
            logging.info("Ind is trained. {0}".format(ind.get_dec()))
            continue
        try:
            fitness = Engine.train(ind.get_dec(),Mode='None')
            ind.set_fitness(fitness)
        except:
            logging.info("Ind is invalid {0}".format(ind.get_dec()))
            ind.set_fitness([[np.inf,np.inf]])
    newPop = pop.get_matrix(popInput=newPop)
    newGeneration = EA.enviromentalSeleection(pop= newPop,popNum=int(config['population setting']['popSize']))
    newGeneration = pop.matrix_to_Pop(newGeneration)
    pop.add_population(newGeneration)
    
