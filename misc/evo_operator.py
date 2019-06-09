import numpy as np
import random
from copy import deepcopy


def SEEMutationV1(code):
        mutateCoverage = 0.2
        phase, unitNumber, unitLength = code.shape
        newCode = deepcopy(code)
        mutateIndex = [(random.randint(0, phase-1), random.randint(0, unitNumber-1))
                       for _ in range(int(unitNumber*phase*mutateCoverage))]
        for i,j in mutateIndex:
            newCode[i,j,:] = np.random.randint(0,9,size=(1,unitLength)) 
            if j == 0:
                newCode[i,j,1] = np.random.randint(2,4) 
        return newCode

def SEECrossoverV1(code1,code2):
    '''Fixed-length cross'''
    phase, unitNumber, unitLength = code1.shape
    # Cross over in SEEcrossover is operated in phase lavel if it ok.
    if phase>1:
        croPhase = random.randint(1,phase-2)
        sub1_1, sub1_2 = code1[0:croPhase,:,:],code1[croPhase:,:,:]
        sub2_1, sub2_2 = code2[0:croPhase,:,:],code2[croPhase:,:,:]
    return np.vstack((sub1_1,sub2_2)),np.vstack((sub2_1,sub1_2))
