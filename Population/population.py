import individual
import numpy as np


class population():
    def __init__(self, config, arg=None):
        self.popSize = int(config['population setting']['popSize'])
        self.fitnessSize = int(config['individual setting']['objectiveNum'])
        self.generation = dict()
        # initiate generation
        self.generation['0'] = [individual.SEA_individual(
            config['individual setting']) for x in range(self.popSize)]

    def get_population(self, index=-1):
        if index == -1:
            index = str(len(self.generation)-1)
        return self.generation[index]

    def add_population(self, newPop):
        if len(newPop) != self.popSize:
            print('The new population size is not suit rule')
        self.generation[str(len(self.generation))] = newPop

    def update_population(self, newPop, index=-1):
        if index == -1:
            index = str(len(self.generation)-1)
        self.generation[index] = newPop

    def get_matrix(self, index=-1):
        '''
        structure define:
        (
            Matrix([popDec|fitness]),
            (popLength, fitnessNum)
        )
        '''
        if index == -1:
            index = str(len(self.generation)-1)
        pop = self.generation[index]
        matrix = np.vstack(
            [np.hstack((ind.get_dec(), ind.get_fitness())) for ind in pop])
        return (matrix, (self.popSize, self.fitnessSize))
