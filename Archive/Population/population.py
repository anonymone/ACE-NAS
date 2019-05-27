import individual
import numpy as np
import json


class population():
    def __init__(self, config, arg=None):
        self.popSize = int(config['population setting']['popSize'])
        self.fitnessSize = int(config['individual setting']['objectiveNum'])
        self.generation = dict()
        # initiate generation
        self.generation['0'] = [individual.SEA_individual(
            config['individual setting']) for x in range(self.popSize)]
        self.config = config

    def get_population(self, index=-1):
        if index == -1:
            index = str(len(self.generation)-1)
        return self.generation[index]

    def add_population(self, newPop):
        if len(newPop) != self.popSize:
            print('The new population size {0} is invalid'.format(len(newPop)))
            if len(newPop) == 0:
                return 
        self.generation[str(len(self.generation))] = newPop

    def update_population(self, newPop, index=-1):
        if index == -1:
            index = str(len(self.generation)-1)
        self.generation[index] = newPop

    def matrix_to_Pop(self, matrix):
        popLength, fitnessNum = matrix[1]
        popSize = matrix[0].shape[0]
        newPop = list()
        for ind in matrix[0]:
            newInd = individual.SEA_individual(self.config['individual setting'], decIn = np.array(ind[0:len(ind)-fitnessNum]))
            if not np.any(np.sum(ind[-fitnessNum:]) == 0):
                newInd.set_fitness(ind[-fitnessNum:])
            newPop.append(newInd)
        return newPop
        

    def get_matrix(self, index=-1, popInput = None):
        '''
        structure define:
        (
            Matrix([popDec|fitness]),
            (popLength, fitnessNum)
        )
        '''
        if index == -1:
            index = str(len(self.generation)-1)
        if popInput is None:
            pop = self.generation[index]
        else:
            pop = popInput
        matrix = np.vstack(
            [np.hstack((ind.get_dec(), ind.get_fitness())) for ind in pop])
        return (matrix, (len(pop), self.fitnessSize))

    def save(self, index=-1, popInput = None, fileName='None'):
        Generation = dict()
        if index == -1:
            index = str(len(self.generation)-1)
        if popInput is None:
            pop = self.generation[index]
        else:
            pop = popInput
        try:
            Writer = open(self.config['population setting']['savePath']+fileName,mode='w')
            for i in range(len(pop)):        
                Generation['candidates{0}'.format(i)] = str(np.hstack((pop[i].get_dec(), pop[i].get_fitness())))
            Generation['popSize'] = self.popSize
            Generation['fitnessSize'] = self.fitnessSize
            strings = json.dumps(Generation)
            Writer.write(strings)
            Writer.close()
            return True
        except:
            return False

