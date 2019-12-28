import random
from copy import deepcopy
import numpy as np
import os
from pandas import DataFrame

from Evaluator.Utils.recoder import create_exp_dir
from SearchEngine.Engine_Interface import population
from SearchEngine.Utils.EA_tools import ACE_CrossoverV1, ACE_Mutation_V2


class EA_population(population):
    def __init__(self,
                 obj_number,
                 pop_size,
                 ind_params,
                 mutation_rate=1,
                 crossover_rate=0.2,
                 mutation: 'muatation callback function' = ACE_Mutation_V2,
                 crossover: 'crossover callback function' = ACE_CrossoverV1,
                 ind_generator: 'the class of individual' = None):
        super(EA_population, self).__init__(
            obj_number=obj_number,
            pop_size=pop_size,
            ind_generator=ind_generator,
            ind_params=ind_params)
        # population update strategy
        self.mutate = mutation
        self.crossover = crossover
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

    def select(self, select_number=None):
        if select_number is None:
            select_number = self.pop_size
        mutate_pool = [random.randint(0, self.pop_size-1)
                       for _ in range(select_number)]
        IDs = list(self.individuals.keys())
        return [deepcopy(self.individuals[IDs[i]]) for i in mutate_pool]

    def new_pop(self, inplace=True):
        assert self.mutate != None, 'Muatating method is not defined.'
        mutate_pool = self.select()
        random.shuffle(mutate_pool)
        pop1, pop2 = mutate_pool[0:int(
            len(mutate_pool)/2)], mutate_pool[int(len(mutate_pool)/2):]
        new_pop = list()
        # crossover
        for ind1, ind2 in zip(pop1, pop2):
            new_ind1, new_ind2 = self.ind_generator(
                self.obj_number, self.ind_params), self.ind_generator(self.obj_number, self.ind_params)
            new_ind1.set_dec(ind1.get_dec())
            new_ind2.set_dec(ind2.get_dec())
            if random.random() < self.crossover_rate:
                new_code1, new_code2 = self.crossover(new_ind1.get_dec(), new_ind2.get_dec())
                new_ind1.set_dec(new_code1)
                new_ind2.set_dec(new_code2)
            if random.random() < self.mutation_rate:
                ind1_code, ind2_code = new_ind1.get_dec(), new_ind2.get_dec()
                new_ind1.set_dec(self.mutate(ind1_code))
                new_ind2.set_dec(self.mutate(ind2_code))
            new_pop.extend([new_ind1, new_ind2])
        if inplace:
            self.add_ind(new_pop)
        else:
            return new_pop

    def to_matrix(self):
        matrix = np.vstack(
            [np.hstack(([ind.get_Id()], ind.get_fitness(), ind.get_fitnessSG())) for ind in self.individuals.values()])
        return matrix

    def to_string(self, callback=None):
        if callback is None:
            return [ind.to_string() for ind in self.get_ind()]
        else:
            return [callback(ind.to_string()) for ind in self.get_ind()]
    
    def save(self, save_path='./data/', file_name='population', file_format='csv', mode='EXPERIMENT'):
        create_exp_dir(save_path)
        table = {
            'ID': list(),
            'Encoding string' : list()
        }
        if mode in ['EXPERIMENT', 'DEBUG']:
            for i in range(self.obj_number):
                table['Fitness{0}'.format(i)] = list()
            for ID,ind in self.individuals.items():
                table['ID'].append(ID)
                table['Encoding string'].append(ind.to_string())
                fitness = ind.get_fitness()
                for i in range(self.obj_number):
                    table['Fitness{0}'.format(i)].append(fitness[i])
        elif mode == 'SURROGATE':
            table['SG_value'] = list()
            for ID,ind in self.individuals.items():
                table['ID'].append(ID)
                table['Encoding string'].append(ind.to_string())
                table['SG_value'].append(ind.get_fitnessSG()[0])
        table = DataFrame(table)
        if file_format == 'csv':
            table.to_csv(os.path.join(save_path,'{0}.csv'.format(file_name)), index=False)
        elif file_format == 'json':
            table.to_json(os.path.join(save_path,'{0}.json'.format(file_name)), index=False)
        else:
            raise Exception('Error file format is specified!') 

    def get_topk(self, k, obj_select=0, order='INC'):
        pop_value = self.to_matrix()
        max_index = pop_value[np.argsort(pop_value[:,obj_select+1]),:]
        if order=='INC':
            return self.get_ind(IDs=max_index[0:k,0])
        elif order=='DEC':
            return self.get_ind(IDs=max_index[-k:,0])

class NSGA2():
    def __init__(self):
        '''
        Deb, Kalyanmoy, et al. "A fast and elitist multiobjective 
        genetic algorithm: NSGA-II." IEEE Transactions on Evolutionary 
        Computation 6.2 (2002): 182-197.
        '''

    @staticmethod
    def isDominated(ind1, ind2, proType='min'):
        if proType == 'min':
            return np.any(ind1 < ind2) and not np.any(ind1 > ind2)
        else:
            return np.any(ind1 > ind2) and not np.any(ind1 < ind2)

    @staticmethod
    def fastNondomiatedSort(pop):
        pop_size, fitnessNum = pop[:, 1:].shape
        fitnessValue = pop[:, 1:]
        S = [list() for _ in range(pop_size)]
        n = np.zeros(shape=pop_size, dtype=int)
        F = list()
        F.append([])
        for p in range(pop_size):
            for q in range(pop_size):
                # if p == q:
                #     continue
                if NSGA2.isDominated(fitnessValue[p], fitnessValue[q]):
                    S[p].append(q)
                if NSGA2.isDominated(fitnessValue[q], fitnessValue[p]):
                    n[p] = n[p] + 1
            if n[p] == 0:
                F[0].append(p)
        i = 0
        while len(F[i]) > 0:
            F.append([])
            for p in F[i]:
                for q in S[p]:
                    n[q] = n[q] - 1
                    if n[q] == 0:
                        F[i+1].append(q)
            i = i + 1
        if not F:
            a = 1
        return F

    @staticmethod
    def crowdingDistance(pop):
        pop_size, fitnessNum = pop[:, 1:].shape
        fitnessValue = pop[:, 1:]
        Idistance = np.zeros(pop_size)
        for m in range(fitnessNum):
            objVector = fitnessValue[:, m]
            orderIndex = np.argsort(objVector)
            # orderIndex = orderIndex[::-1]
            subIndex = orderIndex[1:-1]
            Idistance[orderIndex[0]] = Idistance[orderIndex[-1]] = np.inf
            for i in range(len(subIndex)):
                Idistance[subIndex[i]] = Idistance[subIndex[i]] + (
                    objVector[orderIndex[i+2]] - objVector[orderIndex[i]])/(np.max(objVector)+0.1-np.min(objVector))
        return Idistance

    @staticmethod
    def enviromentalSeleection(pop:'np.ndarray, IDs, Fitness1, Fitness2, ...', popNum : int) -> 'list() Best IDs':
        # if pop size is less than popNum, enviromentalSelection will select all individuals
        popNum = popNum if popNum < len(pop) else len(pop)
        F = NSGA2.fastNondomiatedSort(pop)
        count = 0
        selectingIndex = []
        for FNum, Fi in zip(range(len(F)), F):
            count += len(Fi)
            if count >= popNum:
                count -= len(Fi)
                break
            selectingIndex.extend(Fi)

        if count != popNum:
            subpop = pop[Fi]
            crowdValue = NSGA2.crowdingDistance(subpop)
            index = np.argsort(-crowdValue).tolist()
            index = index[0:(popNum-count)]
            selectingIndex.extend([Fi[x] for x in index])
            selectingIndex.sort()
        IDs = pop[selectingIndex,0]
        rm_IDs = pop[[i for i in range(len(pop)) if i not in selectingIndex], 0]
        return IDs.tolist(), rm_IDs.tolist()
