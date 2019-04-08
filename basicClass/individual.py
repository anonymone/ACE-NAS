import numpy as np


class individual():
    '''
    This is a basic individual class of all solution.
    '''
    def __init__(self, decSize, arg = None):
        '''
        self.__dec is numpy type
        self.__fitness is numpy type
        '''
        self.__dec = None
        self.__fitness = None
        if arg is not None:
            assert type(arg) == dict, 'arg is not dict type.'
            for key in arg:
                self.__dict__[key] = arg[key]
    
    def dec(self):
        return self.__dec.copy()

    def fitness(self):
        return self.__fitness.copy()