import numpy as np
import random
from copy import deepcopy


def SEEMutationV1(code, args):
    mutateCoverage = 0.2
    phase, unitNumber, unitLength = code.shape
    newCode = deepcopy(code)
    mutateIndex = [(random.randint(0, phase-1), random.randint(0, unitNumber-1))
                   for _ in range(int(unitNumber*phase*mutateCoverage))]
    for i, j in mutateIndex:
        perturb = np.random.randint(-1, 1, size=(1, unitLength))
        perturb[perturb == 0] = 1
        newCode[i, j, :] = newCode[i, j, :] + perturb
        newCode[i, j, :] = newCode[i, j, :] % 10
        newCode[newCode < 0] = np.random.randint(
            0, 9, size=np.sum(newCode < 0))
        if j == 0:
            newCode[i, j, 1] = np.random.randint(3, 5)
    return newCode


def SEECrossoverV1(code1, code2, args):
    '''Fixed-length cross'''
    phase, unitNumber, unitLength = code1.shape
    # Cross over in SEEcrossover is operated in phase lavel if it ok.
    if phase > 1:
        croPhase = random.randint(1, phase-2)
        sub1_1, sub1_2 = code1[0:croPhase, :, :], code1[croPhase:, :, :]
        sub2_1, sub2_2 = code2[0:croPhase, :, :], code2[croPhase:, :, :]
    return np.vstack((sub1_1, sub2_2)), np.vstack((sub2_1, sub1_2))


def SEE_Cell_CrossoverV1(code1, code2, args,exchange_prob=0.5):
    n_code1, r_code1 = code1[0], code1[1]
    n_code2, r_code2 = code2[0], code2[1]
    if random.random() > exchange_prob:
        return np.vstack((r_code2, r_code1)), np.vstack((n_code1, n_code2))
    else:
        return np.vstack((n_code1, r_code2)), np.vstack((n_code2, r_code1))


def WF_BEEMutation(code, args):
    pass


def WF_BEECrossover(code1, code2, args):
    pass
