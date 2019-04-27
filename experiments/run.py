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

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(filename='./logs/train.log',
                    level=logging.DEBUG, format=FORMAT)

from individual import individual
from population import population
from StateControl import evaluator
from NSGA2 import NSGA2

EA = NSGA2.NSGA2(config)
Engine = evaluator(config)
pop = population(config=config)

Engine.initEngine()

# evaluate
logging.info("GENERATION init")
for ind in pop.get_population():
    fitness = Engine.train(ind.get_dec(), Mode='DEBUG')
    ind.set_fitness(fitness)
    logging.info("Fitness >>> acc:{0}, comp:{1} ".format(
                fitness[0,0], fitness[0,1]))
if not pop.save(fileName='Initation.dat'):
    logging.error('Population save Failed.')

for i in range(int(config['EA setting']['runTimes'])):
    logging.info("GENERATION {0}".format(i))
    population = pop.get_matrix()
    newpopMatrix = EA.mutate(population)
    newPop = pop.matrix_to_Pop(newpopMatrix)
    newPop.extend(pop.get_population())
    # evaluate
    # [train, pass, invalid]
    trainModelCount = [0, 0, 0]
    for ind in newPop:
        if ind.isTrained():
            trainModelCount[1] = trainModelCount[1]+1
            continue
        try:
            fitness = Engine.train(ind.get_dec(), Mode='DEBUG')
            ind.set_fitness(fitness)
            logging.info("Fitness >>> acc:{0}, comp:{1} ".format(
                fitness[0,0], fitness[0,1]))
            trainModelCount[0] = trainModelCount[0]+1
        except:
            logging.info("Ind is invalid {0}".format(ind.get_dec()))
            ind.set_fitness([[np.inf, np.inf]])
            trainModelCount[2] = trainModelCount[2]+1
    logging.info("Trainning count : trained:{0}, pass:{1}, invalid:{2}".format(
        trainModelCount[0], trainModelCount[1], trainModelCount[2]))

    newPop = pop.get_matrix(popInput=newPop)
    newGeneration = EA.enviromentalSeleection(
        pop=newPop, popNum=int(config['population setting']['popSize']))
    newGeneration = pop.matrix_to_Pop(newGeneration)
    pop.add_population(newGeneration)
    if not pop.save(fileName='Generation{0}.dat'.format(i)):
        logging.error('Population save Failed.')