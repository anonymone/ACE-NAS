import numpy as np
import pandas
from pandas import DataFrame
import random


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
        if type(fitness) == list:
            fitness = np.array(fitness)
        assert type(fitness) == np.ndarray, 'fitness is not a ndarray.'
        self.shape[1] = fitness.size
        self.fitness = fitness.copy().reshape(-1)

    def toString(self):
        return 'Code: {0} \nFitness: {1}'.format(self.dec, self.fitness)

    def toVector(self):
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
        assert self.popSize > 0, 'Population has no individual.'
        self.popSize = self.popSize - 1
        return self.individuals.pop()

    def push(self, ind):
        if type(ind) == list:
            self.individuals.extend(IndList)
            self.popSize = self.popSize + len(IndList)
        else:
            try:
                self.individuals.insert(0, ind)
                self.popSize = self.popSize + 1
            except:
                raise Exception('Individual insert is failed!')


    def save(self, savePath='./data', fileFormat='csv'):
        tabel = {
            'Dec': list(),
            'Fitness': list()
        }
        for ind in self.individuals:
            tabel['Dec'].append(ind.getDec())
            tabel['Fitness'].append(ind.getFitness())
        tabel = DataFrame(tabel)
        if fileFormat == 'csv':
            tabel.to_csv(savePath)
        elif fileFormat == 'json':
            tabel.to_json(savePath)
        else:
            raise Exception('Error file format is specified!')


# SEE class
class SEEIndividual(code):
    def __init__(self, decSize, objSize, blockLength=3, valueBoundary=(1, 9), arg=None):
        super(SEEIndividual, self).__init__(arg=arg)
        self.blockLength = blockLength
        self.boundary = valueBoundary
        self.dec = np.array([random.randint(*valueBoundary)
                             for _ in range(decSize*blockLength)])
        self.fitness = np.zeros(objSize)
        self.shape = [decSize, objSize]

    def toString(self):
        dec = self.dec.reshape((-1,self.blockLength))
        str_dec = ''
        for i in dec:
            str_dec = str_dec + str(i).replace('[','').replace(']','').replace(' ','') + '-'
        return 'Code: {0} \nFitness: {1}'.format(str_dec[0:-1], self.fitness)

    def isTraind(self):
        return np.any(np.sum(self.getFitness()) != 0)


class SEEPopulation(population):
    def __init__(self, popSize, decSize, objSize, blockLength=3, valueBoundary=(1, 9)):
        super(SEEPopulation, self).__init__(objSize=objSize, decSize=decSize,)
        self.individuals = [SEEIndividual(decSize=self.decSize, objSize=self.objSize, blockLength=blockLength, valueBoundary=valueBoundary)
                            for _ in range(popSize)]
        self.popSize = popSize

    def toMatrix(self):
        matrix = np.vstack(
            [np.hstack((ind.getDec(), ind.getFitness())) for ind in self.individuals])
        return matrix


if __name__ == "__main__":
    pop = SEEPopulation(popSize=30, objSize=2, decSize=10)
    ind = pop.pop()
    ind.setFitness([1,2])
    pop.push(ind)
    ind = pop.pop()
    ind.setFitness([3,4])
    pop.push(ind)
    # print(pop.toMatrix())
    print(ind.toString())
    # print(ind.toVector())
    pop.save('./hi.csv')
