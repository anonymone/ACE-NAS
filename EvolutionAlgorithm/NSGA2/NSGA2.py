import EAbase
from Engine import StateControl

import random
import copy
import numpy as np


class NSGA2(EAbase.EAbase):
    def __init__(self, config):
        super(NSGA2, self).__init__()
        self.mutateRate = float(config['EA setting']['mutateRate'])
        self.isCrossOver = bool(config['EA setting']['crossOver'])
        self.mutateType = config['EA setting']['codeType']
        self.mutateGenerate = np.random.randint(1, 9)

    def mutate(self, pop, mutateFunc=None):
        newPop = copy.deepcopy(pop)
        (popLength, fitnessNum) = newPop[1]
        if mutateFunc is None:
            for ind in newPop[0]:
                index = np.unique([random.randint(0, popLength-1)
                                   for x in range(random.randint(1, int(popLength/2)))])
                ind[index] = [self.mutateGenerate for x in range(len(index))]
        return newPop

    def crossOver(self, ind1, ind2):
        cutPoint = np.random.randint(2, ind1.shape[0]-2)
        (sub1, sub2) = (ind1[0:cutPoint], ind1[cutPoint:])
        (sub3, sub4) = (ind2[0:cutPoint], ind2[cutPoint:])
        return np.vstack([np.hstack([sub1, sub4]), np.hstack([sub3, sub2])])

    def crowdingDistance(self, pop):
        pass

    def fastNondomiatedSort(self, pop):
        pass

    def newPop(self, pop):
        pass
