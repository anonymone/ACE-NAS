import numpy as np


class individual():
    '''
    This is a basic individual class of all solution.
    '''

    def __init__(self, config, arg=None):
        '''
        self.__dec is numpy type
        self.__fitness is numpy type
        '''
        self.dec = None
        self.fitness = None
        if arg is not None:
            assert type(arg) == dict, 'arg is not dict type.'
            for key in arg:
                self.__dict__[key] = arg[key]

    def get_dec(self):
        return self.dec.copy()

    def get_fitness(self):
        return self.fitness.copy()

    def set_dec(self, dec):
        self.dec = dec.copy()

    def set_fitness(self, fitness):
        self.fitness = fitness.copy()


class SEA_individual(individual):
    def __init__(self, config, decIn = None):
        # load setting
        super(SEA_individual, self).__init__(config=config)
        self.codeLength = int(config['codeLength'])
        self.blockLength = int(config['blockLength'])
        self.codeType = config['codeType']
        self.codeMaxValue = int(config['codeMaxValue'])
        self.fitnessSize = int(config['objectiveNum'])
        # initiate decision variables
        if decIn is None:
            self.dec = np.random.random_integers(
                1, self.codeMaxValue, size=(1, self.codeLength*self.blockLength))
            self.dec[0,0] = 1
        else:
            self.dec = decIn.copy().reshape((1,decIn.shape[0]))
        self.fitness = np.zeros((1, self.fitnessSize))
    
    def set_fitness(self, fitness):
        self.fitness = fitness.copy()

    def isTrained(self):
        if np.sum(self.fitness) == 0:
            return False
        else:
            return True