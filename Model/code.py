import numpy as np
import pandas
from pandas import DataFrame


class code():
    '''
    This is a basic individual class of all solution.
    '''

    def __init__(self, arg=None):
        '''
        Param: self.dec, numpy type, size (-1).
        Param: self.fitness, numpy type, size (-1).
        Param: shape, list, length of [dec, fitness].
        '''
        self.dec = None
        self.fitness = None
        self.shape = [0, 0]
        if arg is not None:
            assert type(arg) == dict, 'arg is not dict type.'
            for key in arg:
                self.__dict__[key] = arg[key]

    def getDec(self):
        return self.dec.copy()

    def getFitness(self):
        return self.fitness.copy()

    def setDec(self, dec):
        assert type(dec) == np.ndarray, 'dec is not a ndarray.'
        self.shape[0] = dec.size
        self.dec = dec.copy().reshape(-1)

    def setFitness(self, fitness):
        assert type(dec) == np.ndarray, 'fitness is not a ndarray.'
        self.shape[1] = fitness.size
        self.fitness = fitness.copy().reshape(-1)

    def toString(self):
        return 'Code: {0} \n Fitness: {1}'.format(self.dec, self.fitness)

    def toVextor(self):
        return np.hstack((self.dec, self.fitness))


class population:
    def __init__(self, objSize, decSize):
        self.objSize = objSize
        self.decSize = decSize
        self.popSize = 0
        self.individuals = list()

    def addIndividuals(self, IndList):
        assert type(
            IndList) == list, 'IndList in addIndividuals must be a list.'
        self.individuals.extend(IndList)
        self.popSize = self.popSize + len(IndList)

    def pop(self):
        return self.individuals.pop()

    def push(self, ind):
        assert self.individuals.__len__ > 0, 'Population has no individual.'
        return self.individuals.insert(0, ind)

    def save(self, savePath='./data'):
        tabel = {
            'Dec': list(),
            'Fitness': list()
        }
        for ind in self.individuals:
            tabel['Dec'].append(ind.getDec())
            tabel['Fitness'].append(ind.getFitness())
        tabel = DataFrame(tabel)
        tabel.to_json(savePath)


# SEE class
class SEEIndividual(code):
    def __init__(self,  fitnessSize, blockLength=3, valueBoundary=(1, 9), arg=None):
        super(SEEIndividual, self).__init__(arg=arg)
        self.blockLength = blockLength
        self.boundary = valueBoundary
        self.dec = 

    def isTraind(self):
        return np.any(np.sum(self.getFitness()) != 0)


class SEEPopulation(population):
    def __init__(self, objSize, decSize):
        super(SEEPopulation, self).__init__(objSize=objSize, decSize=decSize)

    def init(self, popSize=30, blockLength=3, valueBoundary=(1, 9)):
        inds = [SEEIndividual(fitnessSize, blockLength, valueBoundary)
                for _ in range(popSize)]
        self.addIndividuals(inds)

    def toMatrix(self):
        matrix = np.vstack(
            [np.hstack((ind.get_dec(), ind.get_fitness())) for ind in self.individuals])
        return matrix


if __name__ == "__main__":
    pop = SEEPopulation(objSize=2, decSize=10)
