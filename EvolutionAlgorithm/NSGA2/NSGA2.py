import sys
sys.path.insert(0,'./EvolutionAlgorithm')
import EAbase

import random
import copy
import numpy as np


class NSGA2(EAbase.EAbase):
    def __init__(self):
        super(NSGA2, self).__init__()

    def isDominated(self, ind1, ind2, proType='min'):
        if proType == 'min':
            return np.any(ind1 < ind2) and not np.any(ind1 > ind2)
        else:
            return np.any(ind1 > ind2) and not np.any(ind1 < ind2)

    def fastNondomiatedSort(self, pop):
        popSize, fitnessNum = pop[:,1:].shape
        fitnessValue = pop[:, 1:]
        S = [list() for _ in range(popSize)]
        n = np.zeros(shape=popSize, dtype=int)
        F = list()
        F.append([])
        for p in range(popSize):
            for q in range(popSize):
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
        popSize, fitnessNum = pop[:,1:].shape
        fitnessValue = pop[:, 1:]
        Idistance = np.zeros(popSize)
        for m in range(fitnessNum):
            objVector = fitnessValue[:, m]
            orderIndex = np.argsort(objVector)
            # orderIndex = orderIndex[::-1]
            subIndex = orderIndex[1:-1]
            Idistance[orderIndex[0]] = Idistance[orderIndex[-1]] = np.inf
            for i in range(len(subIndex)):
                Idistance[subIndex[i]] = Idistance[subIndex[i]] + (
                    objVector[orderIndex[i+2]] - objVector[orderIndex[i]])/(np.max(objVector)+0.1-np.min(objVector))
        return  Idistance

    def enviromentalSeleection(self, pop, popNum):
        # if pop size is less than popNum, enviromentalSelection will select all individuals
        popNum = popNum if popNum < len(pop) else len(pop)
        F = self.fastNondomiatedSort(pop)
        count = 0
        selectingIndex = []
        for FNum, Fi in zip(range(len(F)),F):
            count += len(Fi)
            if count >= popNum:
                count -= len(Fi)
                break
            selectingIndex.extend(Fi)
            
        if count != popNum:
            subpop = pop[Fi]
            crowdValue = self.crowdingDistance(subpop)
            index = np.argsort(-crowdValue).tolist()
            index = index[0:(popNum-count)]
            selectingIndex.extend([Fi[x] for x in index])
            selectingIndex.sort()
        return selectingIndex


if __name__ == "__main__":
    engin = NSGA2()
    popSize = 100
    pop = np.hstack(((np.array([x for x in range(popSize)]).reshape(-1,1), np.random.randint(0,9,size=(popSize,2)))))
    # pop = np.array([[0, 2, 7],
    #                 [1, 7, 0],
    #                 [2, 5, 8],
    #                 [3, 1, 4],
    #                 [4, 4, 4],
    #                 [5, 8, 2],
    #                 [6, 0, 8],
    #                 [7, 6, 6],
    #                 [8, 6, 5],
    #                 [9, 2, 4]])
    print(pop)
    elistist = engin.enviromentalSeleection(pop,50)
    print(elistist)