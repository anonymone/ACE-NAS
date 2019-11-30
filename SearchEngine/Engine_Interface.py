import numpy as np
import pandas as pd
from pandas import DataFrame
import random
from copy import deepcopy

class population:
    def __init__(self,
                obj_number,
                pop_size,
                ind_generator:'the class of individual'=None):
        self.obj_number = obj_number
        self.pop_size = pop_size
        self.ind_generator = ind_generator
        if self.ind_generator is None:
            self.individuals = dict()
        else:
            self.individuals = dict()
            for i in range(self.pop_size):
                ind = self.ind_generator(self.obj_number)
                self.individuals[ind.get_Id()] = ind

    
    def get_ind(self, IDs=None):
        if IDs is None:
            return [ind for ind in self.individuals.values()]
        else:
            inds = list()
            for i in IDs:
                try:
                    inds.append(self.individuals[i])
                except:
                    print("Individual ID:{0} is not in population.".format(i))
            return inds
    
    def add_ind(self, ind):
        if type(ind) != list:
            ind = [ind]
        try:
            for i in ind:
                self.individuals[i.ID] = i
                self.pop_size = self.pop_size + 1
        except:
            raise Exception('Individual add failed!')
    
    def remove_ind(self, IDs=None):
        assert self.pop_size > 0, 'Population has no individual.'
        if not isinstance(IDs, list):
            IDs = [IDs]
        # clear all
        if IDs is None:
            self.individuals = dict()
            self.pop_size = 0
        else:
            for i in IDs:
                try:
                    del self.individuals[i]
                except:
                    raise Exception(
                        'delete the {0}(st/rd/th) individual in population with size {1} failed.'.format(i, self.pop_size))
                self.pop_size = self.pop_size - 1
    
    def save(self, save_path='./data/', file_format='csv'):
        table = {
            'ID': list(),
            'Encoding string' : list()
        }
        for i in range(self.obj_number):
            table['Fitness{0}'.format(i)] = list()
        for ID,ind in self.individuals.items():
            table['ID'].append(ID)
            table['Encoding string'].append(ind.to_string())
            fitness = ind.get_fitness()
            for i in range(self.obj_number):
                table['Fitness{0}'.format(i)].append(fitness[i])
        table = DataFrame(table)
        if file_format == 'csv':
            table.to_csv(save_path+'.csv')
        elif file_format == 'json':
            table.to_json(save_path+'.json')
        else:
            raise Exception('Error file format is specified!') 