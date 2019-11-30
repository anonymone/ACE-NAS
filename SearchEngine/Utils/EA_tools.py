import numpy as np
import random
from copy import deepcopy

def ACE_Mutation_V2(code):
    normal_code, reduction_code = code[0], code[1]
    unit_size, unit_length = normal_code.shape
    normal_mutaion_converage = random.randint(1, 5)*0.1
    normal_mutation_index = [(random.randint(0, unit_size-1), random.randint(0, unit_length-1))
                              for _ in range(int(unit_length*unit_size*normal_mutaion_converage))]
    
    unit_size, unit_length = reduction_code.shape
    reduction_mutation_converage = random.randint(1, 5)*0.1
    reduction_mutation_index = [(random.randint(0, unit_size-1), random.randint(0, unit_length-1)) 
                              for _ in range(int(unit_length*unit_size*normal_mutaion_converage))]
    for i, j in normal_mutation_index:
        if j == 0:
            token = random.randint(0, 3)
        else:
            token = random.randint(0, 15)
        if j == 0 and token == normal_code[i, j]%4:
            normal_code[i, j]= (normal_code[i, j] + random.randint(0,3))%4
        elif j != 0 and token == normal_code[i, j]%10:
            normal_code[i, j]= (normal_code[i, j] + random.randint(0,9))%10
        else:
            normal_code[i, j]= token
    for i, j in reduction_mutation_index:
        if j == 0:
            token = random.randint(0, 3)
        else:
            token = random.randint(0, 15)
        if j == 0 and token == reduction_code[i, j]%4:
            reduction_code[i, j]= (reduction_code[i, j] + random.randint(0,3))%4
        elif j != 0 and token == reduction_code[i, j]%10:
            reduction_code[i, j]= (reduction_code[i, j] + random.randint(0,9))%10
        else:
            reduction_code[i, j]= token
    return (normal_code, reduction_code)

def ACE_CrossoverV1(code1, code2):
    '''Fixed-length cross'''
    ind1_code1, ind1_code2 = code1
    ind2_code1, ind2_code2 = code2
    return ((ind2_code1, ind1_code2), (ind1_code1, ind2_code2))
