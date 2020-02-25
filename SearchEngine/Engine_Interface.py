import numpy as np
import pandas as pd
from pandas import DataFrame
import random
from copy import deepcopy
import os

from Evaluator.Utils.recoder import create_exp_dir

class population:
    def __init__(self,
                obj_number,
                pop_size,
                ind_params,
                ind_generator:'the class of individual'=None):
        self.obj_number = obj_number
        self.pop_size = pop_size
        self.ind_generator = ind_generator
        self.ind_params = ind_params
        if self.ind_generator is None:
            self.individuals = dict()
        else:
            self.individuals = dict()
            for i in range(self.pop_size):
                ind = self.ind_generator(fitness_size = self.obj_number, ind_params=ind_params)
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
        # update the pop_size. there is a very tiny possibility that pop_size mismatch with self.individuals and I have not find the bugs yet. Orz.
        # The bug ocuurs when the topk individuals still alive after surrogate search.
        # self.pop_size= len(self.individuals)
        if type(ind) != list:
            ind = [ind]
        try:
            for i in ind:
                if i.ID not in self.individuals.keys():
                    self.pop_size = self.pop_size + 1
                self.individuals[i.ID] = i
        except:
            raise Exception('Individual add failed!')
    
    def remove_ind(self, IDs=None):
        assert self.pop_size > 0, 'Population has no individual.'
        # clear all
        if IDs is None:
            self.individuals = dict()
            self.pop_size = 0
            return None
        if not isinstance(IDs, list):
            IDs = [IDs]
        else:
            for i in IDs:
                try:
                    del self.individuals[i]
                except:
                    raise Exception(
                        'delete the {0}(st/rd/th) individual in population with size {1} failed.'.format(i, self.pop_size))
                self.pop_size = self.pop_size - 1
    
    def save(self, save_path='./data/', file_name='population', file_format='csv'):
        create_exp_dir(save_path)
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
            table.to_csv(os.path.join(save_path,'{0}.csv'.format(file_name)), index=False)
        elif file_format == 'json':
            table.to_json(os.path.join(save_path,'{0}.json'.format(file_name)), index=False)
        else:
            raise Exception('Error file format is specified!') 
    
    def to_matrix(self):
        print("This method needs to be modified before using it.")
        return None