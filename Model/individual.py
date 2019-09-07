import numpy as np
import pandas as pd
from pandas import DataFrame
import random
from copy import deepcopy
import argparse
import uuid
import logging


class code():
    '''
    This is a basic individual class of all solution.
    '''

    def __init__(self, args=None):
        '''
        Param: self.dec, numpy type, size (-1).
        Param: self.fitness, numpy type, size (-1).
        Param: shape, list, length of [dec, fitness].
        '''
        self.dec = np.array([])
        self.fitness = np.array([])
        self.blockLength = 1
        self.shape = [0, 0]
        if args is not None:
            assert type(args) == dict, 'args is not dict type.'
            for key in args:
                self.__dict__[key] = args[key]
        self.evaluated = False

    def getDec(self):
        return self.dec.copy()

    def getFitness(self):
        return self.fitness.copy()

    def setDec(self, dec):
        '''
        used to change the value of Dec.
        Param: dec, numpy.ndarray, sise(-1)
        '''
        try:
            dec = np.array(dec)
        except:
            raise Exception('dec is not a ndarray.')
        self.shape[0] = dec.size
        if self.blockLength == 1:
            self.dec = dec.copy().reshape(-1)
        else:
            self.dec = dec.copy().reshape(self.blockLength)

    def setFitness(self, fitness):
        '''
        used to change the value of fitness.
        Param: fitness, numpy.ndarray, sise(-1)
        '''
        try:
            fitness = np.array(fitness)
        except:
            raise Exception('fitness is not a ndarray.')
        self.shape[1] = fitness.size
        self.fitness = fitness.copy().reshape(-1)

    def toString(self):
        '''
        return the code and fitness as string type.
        '''
        return 'Code: {0} \nFitness: {1}'.format(self.dec, self.fitness)

    def toVector(self):
        '''
        return the code and fitness as a numpy.ndarray with size (-1).
        '''
        return np.hstack((self.dec, self.fitness))
    
    def isEvaluated(self):
        return self.evaluated


class population:
    def __init__(self, objSize, mutation, evaluation, callback=None, crossover=None, crossoverRate=0.2, mutationRate=0.9,args=None):
        '''
        Param: objSize, int, the number of objective.
        Param: popSize, int, the number of individuals.
        Param: individuals, list, storing all individuals. 
        Param: mutate, callback func, mutating function.
        Param: crossoverRate, float, range(0,1), the cross over rate of population.
        Param: eval, callback func, evaluating function.
        Param: callback, callback func, to do something specific thing.
        Param: crossover, callback func, the cross over method.
        '''
        self.objSize = objSize
        self.popSize = 0
        self.individuals = list()
        self.mutate = mutation
        self.crossoverRate = crossoverRate
        self.mutationRate = mutationRate
        self.eval = evaluation
        self.callback = callback
        self.crossover = crossover
        self.args = args

    def evaluation(self, individuals=None, **kwargs):
        if self.args.evalMode == 'DEBUG' and individuals is None:
            for indId, ind in zip(range(self.popSize), self.individuals):
                if self.individuals[indId].isTraind():
                    continue
                self.individuals[indId].setFitness(np.random.random((1,2)))
                self.individuals[indId].evaluated = True
            return None
        elif self.args.evalMode == 'DEBUG' and individuals is not None:
            for indId, _ in enumerate(individuals):
                if individuals[indId].isTraind():
                    continue
                individuals[indId].setFitness(np.random.random((1,2)))
                individuals[indId].evaluated = True
            return deepcopy(individuals)
        else:
            assert self.eval != None, 'evaluating method is not defined.'
            if individuals is None:
                for indId, ind in zip(range(self.popSize), self.individuals):
                    if self.individuals[indId].isTraind():
                        continue
                    fitness = self.eval(ind, self.args,complement=True, **kwargs)
                    self.individuals[indId].setFitness([fitness['valid_err'],fitness['flops']])
                    self.individuals[indId].evaluated = True
                return None
            else :
                for indId, ind in enumerate(individuals):
                    if individuals[indId].isTraind():
                        continue
                    fitness = self.eval(ind, self.args,complement=True, **kwargs)
                    individuals[indId].setFitness([fitness['valid_err'],fitness['flops']])
                    individuals[indId].evaluated = True
                return deepcopy(individuals)

    def newPop(self, individuals=None, index=None, inplace=True):
        assert self.mutate != None, 'mutating method is not defined.'
        if index is not None and individuals is None:
            subPop = deepcopy(self.individuals[index])
        elif index is None and individuals is None:
            subPop = deepcopy(self.individuals)
        elif index is not None and individuals is not None:
            subPop = deepcopy(individuals[index])
        else:
            subPop = deepcopy(individuals)
        newPop = []
        for indID, ind in zip(range(len(subPop)), subPop):
            if random.random() < self.mutationRate:
                code = self.mutate(ind.getDec(), self.args)
                subPop[indID].setDec(code)
                subPop[indID].setFitness(
                    [0 for _ in range(subPop[indID].shape[1])])
                # Update ID
                subPop[indID].ID = uuid.uuid1()
                subPop[indID].evaluated = False
                newPop.append(subPop[indID])
            if self.crossover is not None and random.random() < self.crossoverRate:
                ind1, ind2 = self.individuals[random.randint(
                    0, self.popSize-1)], self.individuals[random.randint(0, self.popSize-1)]
                fitness1, fitness2 = ind1.getFitness(), ind2.getFitness()
                winTimes1 = np.sum(fitness1 < fitness2)
                winTimes2 = len(fitness1) - winTimes1
                if winTimes1 > winTimes1:
                    betterCode = ind1.getDec()
                else:
                    betterCode = ind2.getDec()
                newCode1,newCode2 = self.crossover(ind.getDec(), betterCode, self.args)
                newInd = [subPop[indID].copy(newCode1), subPop[indID].copy(newCode1)]
                # Update ID
                for i in newInd:
                    i.ID = uuid.uuid1()
                    i.evaluated = False
                # self.add(newInd)
                newPop.extend(newInd)
        if inplace:
            self.add(newPop)
        else:
            return newPop

    def getInd(self, index=None):
        if index is None:
            return self.individuals
        else:
            return [self.individuals[i] for i in index]

    def remove(self, index):
        '''
        remove individuals in index.
        '''
        assert self.popSize > 0, 'Population has no individual.'
        # make sure that del is start from tail of list.
        index.sort(reverse=True)
        for i in index:
            try:
                del self.individuals[i]
            except:
                raise Exception(
                    'delete the {0}(st/rd/th) individual in population with size {1} failed.'.format(i, self.popSize))
            self.popSize = self.popSize - 1

    def add(self, ind):
        '''
        insert the ind befor the existing individuals in self.individuals.
        Param: individual class or a list of individual class.
        '''
        if type(ind) != list:
            ind = [ind]
        try:
            for i in ind:
                self.individuals.insert(0, i)
                self.popSize = self.popSize + 1
        except:
            raise Exception('Individual insert is failed!')

    def save(self, savePath='./data', fileFormat='csv'):
        '''
        Save the population into file.
        Param: savePath, dir string, the saving path.
        Param: fileFormat, one of ['csv','json'], Specifying the saving file format.
        '''
        tabel = {
            'Dec': list(),
            'Fitness': list()
        }
        for ind in self.individuals:
            tabel['Dec'].append(ind.getDec().reshape(-1))
            tabel['Fitness'].append(ind.getFitness())
        tabel = DataFrame(tabel)
        if fileFormat == 'csv':
            tabel.to_csv(savePath+'.csv')
        elif fileFormat == 'json':
            tabel.to_json(savePath+'.json')
        else:
            raise Exception('Error file format is specified!')


# SEE class
class SEEIndividual(code):
    def __init__(self, objSize, blockLength=(3, 13, 3), valueBoundary=(0, 9), args=None):
        '''
        :Param blockLength, tuple, (phase, block length, unit length) unit is the smallest part of the operating code.
        :Param boundary, tuple, the Max and Min value of code.
        :Param dec, numpy.array, the decision vector.
        :Param fitness , numpy.array, fitness vector.
        :shape the size of decsion vector and objectice vector.
        '''
        super(SEEIndividual, self).__init__(args=args)
        self.blockLength = blockLength
        self.boundary = valueBoundary
        self.dec = np.random.randint(*valueBoundary, blockLength)
        self.dec[0, 0, 1] = np.random.randint(3, 9)
        for i in range(blockLength[0]):
            # backbone need 4~6 nodes.
            self.dec[i, 0, 1] = np.random.randint(4,7)
        self.fitness = np.zeros(objSize)
        # the surrogate predicting value
        self.fitness_SG = np.zeros(1)
        decSize = blockLength[0]*blockLength[1]*blockLength[2]
        self.shape = [decSize, objSize]
        self.ID = uuid.uuid1()
    
    def copy(self, dec=None):
        if dec is not None:
            return SEEIndividual(self.shape[1], self.blockLength, self.boundary,None)
        else:
            return SEEIndividual(self.objSize, self.blockLength, self.valueBoundary,self.args)

    def toString(self, showFitness=False, displayUsed= True):
        dec = self.dec.reshape(self.blockLength)
        str_dec = ''
        if displayUsed:
            for phase in dec:
                str_dec = str_dec + 'Phase:'
                for i in phase:
                    str_dec = str_dec + \
                        str(i).replace('[', '').replace(
                            ']', '').replace(' ', '') + '-'
            if showFitness:
                fitnessString = '--> {0}'.format(self.fitness)
            else:
                fitnessString = ''
            return '{0}'.format(str_dec[0:-1]) + fitnessString
        else:
            return " ".join(["-".join([str(s) for s in unit]) for unit in self.getDec().reshape(-1,3)])

    def setFitnessSG(self, fitnessSG):
        if type(fitnessSG) != np.array:
            fitnessSG = np.array(fitnessSG).reshape(1)
        self.fitness_SG = fitnessSG

    def getFitnessSG(self):
        return self.fitness_SG

    def isTraind(self):
        '''
        used to check whether individual is assigned with a fitness.
        '''
        return self.isEvaluated()


class SEEPopulation(population):
    def __init__(self, popSize, objSize, 
                    blockLength=(3, 12, 3), 
                    valueBoundary=(0, 9), 
                    mutation=None, 
                    crossover=None, 
                    evaluation=None, 
                    args=None):
        super(SEEPopulation, self).__init__(objSize=objSize, crossover=crossover,
                                            mutation=mutation, evaluation=evaluation, 
                                            crossoverRate=args.crossoverRate, 
                                            mutationRate= args.mutationRate, args=args)
        self.individuals = [SEEIndividual(objSize=self.objSize, blockLength=blockLength, valueBoundary=valueBoundary)
                            for _ in range(popSize)]
        self.popSize = popSize

    def newPop(self, index=None, inplace=True):
        Pop = super().newPop(index=index, inplace=False)
        for ind in Pop:
            ind.setFitnessSG(0.0)
        self.add(Pop)

    def load(self, path, objSize, blockLength, valueBoundary, addMode = False):
        try:
            historyData = pd.read_csv(path)
        except:
            logging.error("Open {0} failed!".format(path))
        decLength = blockLength[0]*blockLength[1]*blockLength[2]
        individuals = []
        for Id, indData in enumerate(historyData.values):
            ID, dec, fitness, fitnessSG = indData[0],indData[2:decLength+2], indData[decLength+2:decLength+objSize+2], indData[2+decLength+objSize:]
            ind = SEEIndividual(objSize=self.objSize, blockLength=blockLength, valueBoundary=valueBoundary)
            ind.setDec(dec.astype("int"))
            ind.setFitness(fitness)
            ind.setFitnessSG(fitnessSG)
            individuals.append(ind)
            ind.ID = ID
        if addMode:
            self.popSize = self.popSize + len(individuals)
            self.individuals.extend(individuals)
        else:
            self.popSize = len(individuals)
            self.individuals = individuals


    def toMatrix(self, needDec=False):
        '''
        return a table containing dec and fitness, numpy.ndarray.
        '''
        if needDec:
            matrix = np.vstack(
                [np.hstack(([ind_id], ind.getDec().flatten(), ind.getFitness(), ind.getFitnessSG())) for ind_id, ind in zip(range(self.popSize), self.individuals)])
        else:
            matrix = np.vstack(
                [np.hstack(([ind_id], ind.getFitness(), ind.getFitnessSG())) for ind_id, ind in zip(range(self.popSize), self.individuals)])

        return matrix

    def toString(self, needFitness=True):
        '''
        return a list with tuple contraining (dec string like 'unit-unit-...', fitness )
        '''
        def num2string(dec):
            dec = dec.reshape(-1,3)
            return " ".join(["-".join([str(s) for s in unit]) for unit in dec])

        if needFitness:
            stringList = [(num2string(ind.getDec()), ind.getFitness()) for ind in self.individuals]
        else:
            stringList = [num2string(ind.getDec()) for ind in self.individuals]
        return stringList
    
    def save(self, savePath='./data', fileFormat='csv'):
        tabel = DataFrame(self.toMatrix(needDec=True))
        tabel.index = [ind.ID for ind in self.individuals]
        if fileFormat=='csv':
            tabel.to_csv(savePath+'.csv')
        elif fileFormat == 'json':
            tabel.to_json(savePath+'.json')
        else:
            raise Exception('Error file format is specified!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Multi-objetive Genetic Algorithm for SEENAS")
    parser.add_argument('--crossoverRate', type=float,
                        default=0.2, help='The propability rate of crossover.')
    parser.add_argument('--mutationRate', type=float,
                        default=0.2, help='The propability rate of crossover.')
    args = parser.parse_args()
    pop = SEEPopulation(popSize=30, objSize=2, args=args)
    pop.load(path='./Dataset/initialization/population.csv',objSize=2,blockLength=(3,15,3),valueBoundary=(0,9))
    # ind = pop.individuals[0]
    # ind.setFitness([1.4314, 2.342])
    # pop.add(ind)
    # ind.setFitness([3.34, 4.4231])
    # pop.add([ind, ind])
    # print(pop.toMatrix())
    # print(ind.toString())
    # # print(ind.toVector())
    # print(pop.toMatrix())
    # pop.save('./Experiments/hi')
    print(pop.toString())
