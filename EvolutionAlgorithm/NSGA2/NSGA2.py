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
                index = np.unique([random.randint(0, len(ind)-3)
                                   for x in range(random.randint(1, int(popLength/2)))])
                ind[index] = [self.mutateGenerate for x in range(len(index))]
                ind[-fitnessNum:] = 0
        return newPop

    def crossOver(self, ind1, ind2):
        cutPoint = np.random.randint(2, ind1.shape[0]-2)
        (sub1, sub2) = (ind1[0:cutPoint], ind1[cutPoint:])
        (sub3, sub4) = (ind2[0:cutPoint], ind2[cutPoint:])
        return np.vstack([np.hstack([sub1, sub4]), np.hstack([sub3, sub2])])

    def isDominated(self, ind1, ind2, proType='min'):
        if proType == 'min':
            return np.any(ind1 < ind2) and not np.any(ind1 > ind2)
        else:
            return np.any(ind1 > ind2) and not np.any(ind1 < ind2)

    def fastNondomiatedSort(self, pop):
        popLength, fitnessNum = pop[1]
        popSize = pop[0].shape[0]
        fitnessValue = pop[0][:, -fitnessNum:]
        S = [list() for _ in range(popSize)]
        n = np.zeros(shape=popSize, dtype=int)
        F = list()
        F.append([])
        for p in range(popLength):
            for q in range(popLength):
                # if p == q:
                #     continue
                if self.isDominated(fitnessValue[p], fitnessValue[q]):
                    S[p].append(q)
                if self.isDominated(fitnessValue[q], fitnessValue[p]):
                    n[p] = n[p] + 1
            if n[p] == 0:
                F[0].append(p)
        i = 0
        while len(F[i]) > 0:
            F.append([])
            for p in F[i]:
                for q in S[p]:
                    n[q] = n[q] - 1
                    if n[q] == 0:
                        F[i+1].append(q)
            i = i + 1
        if not F:
            a =1
        return F

    def crowdingDistance(self, pop):
        popLength, fitnessNum = pop[1]
        popSize = pop[0].shape[0]
        fitnessValue = pop[0][:, -fitnessNum:]
        Idistance = np.zeros(popSize)
        for m in range(fitnessNum):
            objVector = fitnessValue[:, m]
            orderIndex = np.argsort(objVector)
            subIndex = orderIndex[1:-1]
            Idistance[orderIndex[0]] = Idistance[orderIndex[-1]] = np.inf
            for i in range(len(subIndex)):
                Idistance[subIndex[i]] = Idistance[subIndex[i]] + (
                    objVector[orderIndex[i+1]] - objVector[orderIndex[i-1]])/(np.max(objVector)+0.1-np.min(objVector))
        return  Idistance

    def enviromentalSeleection(self, pop, popNum):
        F = self.fastNondomiatedSort(pop)
        count = 0
        selectingIndex = []
        for FNum, Fi in zip(range(len(F)),F):
            count += len(Fi)
            if count >= popNum:
                break
            selectingIndex.extend(Fi)
            
        if count != popNum:
            subpop = pop[0][F[FNum]]
            subpop = (subpop,pop[1])
            crowdValue = self.crowdingDistance(subpop)
            index = np.argsort(-crowdValue)
            index = index[0:(len(F[FNum])-(count-popNum))]
            selectingIndex.extend(index)
        newPop = pop[0][selectingIndex,:]
        return (newPop,(len(newPop),pop[1][1]))

    def newPop(self, pop):
        pass
