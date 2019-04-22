import EAbase as EAbase
from Engine import StateControl

import random
import numpy as np


class NSGA2(EABase):
    def __init__(self, config):
        super(NSGA2, self).__init__()
        self.mutateRate = int(config['EA setting']['mutateRate'])
        self.crossOver = bool(config['EA setting']['crossOver'])
        self.mutateType = config['EA setting']['codeType']
        self.mutateGenerate = np.randint(1,9)

    def mutate(self, pop, mutateFunc=None):
        newPop = pop.copy()
        (popLength, fitnessNum) = newPop[1]
        if mutateFunc is None:
            for ind in newPop[0]:
                index = np.unique([random.randint(0, popLength-1)
                                   for x in range(random.randint(1, int(popLength/2)))])
                ind[index] = [self.mutateGenerate for x in range(len(index))]
        return newPop

    def crossOver(self, ind1, ind2):
        cutPoint = np.randint(2, len(ind1)-2)
        (sub1,sub2) = (ind1[0][0:cutPoint], ind1[0][cutPoint:])
        (sub3,sub4) = (ind2[0][0:cutPoint], ind2[0][cutPoint:])
        return (np.hstack(sub1,sub4),np.hstack(sub3,sub2))


    def crowdingDistance(self, pop):
        pass

    def fastNondomiatedSort(self, pop):
        pass

    def newPop(self, pop):
        pass
