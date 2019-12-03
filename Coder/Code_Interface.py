import numpy as np
import pandas as pd
from pandas import DataFrame
import random
from copy import deepcopy
import uuid

class code():
    '''
    This is a basic code class which define 
    the standard interface of encoding strategy class.
    '''
    def __init__(self):
        self.dec = np.array([0])
        self.fitness = np.array([0])
        self.shape = self.dec.shape
        self.evaluated = False
        self.ID = uuid.uuid1().int

    def get_dec(self):
        return deepcopy(self.dec)

    def get_fitness(self):
        return self.fitness

    def get_Id(self):
        return self.ID

    def set_dec(self, dec):
        self.dec = dec
    
    def set_fitness(self, fitness):
        self.fitness = np.array(fitness).reshape(-1)
        self.evaluated = True
    
    def reset_fitness(self):
        self.fitness = np.zeros(self.fitness.shape)
        self.evaluated = False
    
    def get_model(self) -> 'network model':
        print("This interface need to be modified defore use it.")

    def toString(self):
        return "".join(self.dec)
    
    def is_evaluated(self):
        return self.evaluated
