import sys
sys.path.append('./')
import numpy as np

from Coder.ACE import ACE
from EA_Engine import EA_population, NSGA2

pop = EA_population(2, 30, ind_generator=ACE, crossover_rate=1, mutation_rate=1)
matrix = pop.to_matrix()
pop_pool = pop.select(10)
pop_pool[0].set_fitness([2,3])
pop.new_pop()
pop.save('./Experiments/test_module/pop')
rm_ind = pop.get_ind()[0:10]
pop.remove_ind([ind.get_Id() for ind in rm_ind])
for ind in pop.get_ind():
    ind.set_fitness(np.random.rand(1,2))
best_IDs, rm_IDs = NSGA2.enviromentalSeleection(pop.to_matrix(), 15)
pop.remove_ind(rm_IDs)
print(pop.pop_size)