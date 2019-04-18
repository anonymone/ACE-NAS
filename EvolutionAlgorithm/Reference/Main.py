import time
from P_settings import *
# from P_objective import *
import P_objective
from F_NDSort import *
from F_distance import *
from F_mating import *
from P_generator import *
from F_EnvironmentSelect import *
import matplotlib.pyplot as plt
import profile
import scipy.fftpack as sci


# Copyright 2018 Yang Shang shang

def Initialization_Pop(Dct_num, dim, population_size):
    Population = np.hstack((np.random.uniform(-1.0, 1.0, [
                           population_size, Dct_num]),  np.random.randint(0, 2, [population_size, dim-1])))
    Boundary = np.hstack(
        (np.tile([[1], [-1]], [1, Dct_num]), np.tile([[1], [0]], [1, dim-1])))
    Coding = ['Real', 'Binary']
    return Population, Boundary, Coding


def Idct_transform(Individual, dim, Dct_num):  # 需要改进
    DCT_Matrix = np.zeros((dim*dim))
    DCT_coefficient = Individual[:Dct_num]
    select_index = np.hstack((1, Individual[Dct_num:])) == 1
    i = 0
    total_index = []
    while(i*(i+1) < 2*Dct_num):  # DCT 系数过多 就不适用(仅适用于 上半部分，下半部分需要判断)，需要改进，目前足以
        i_index = [i]
        for j in range(1, i+1):
            i_index = np.hstack((i_index, i_index[j-1]+dim-1))
        total_index = np.hstack((total_index, i_index))
        i += 1
    DCT_coefficient = np.hstack(
        (DCT_coefficient, np.zeros((np.int64(i*(i+1)/2-Dct_num),))))
    DCT_Matrix[np.int64(total_index)] = DCT_coefficient
    DCT_Matrix = np.reshape(DCT_Matrix, (dim, dim))
    # 默认 type =2 取值{1，2，3},详情见
    Weights_Matrix = sci.idctn(DCT_Matrix, type=2, norm='ortho')
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.fftpack.dct.html   ,scipy.fftpack.dct ，norm='ortho'仅在type=2 可以使用
    # sci.idctn([[1, 3], [2, 2]], type=2, norm='ortho')
    Used_Weights = Weights_Matrix[:, select_index]
    return Used_Weights, Weights_Matrix


def Evaluate(Population, Dct_num, dim):
    pop_size = Population.shape[0]
    for i in range(pop_size):
        pop_size = 1

    return pop_size


def EA_Run(Generations, PopSize, M, Run, Problem, Algorithm):
    Generations, PopSize = P_settings(Algorithm, Problem, M)
    Population, Boundary, Coding = P_objective.P_objective(
        "init", Problem, M, PopSize)
    FunctionValue = P_objective.P_objective("value", Problem, M, Population)

    FrontValue = F_NDSort(FunctionValue, "half")[0]
    CrowdDistance = F_distance(FunctionValue, FrontValue)

    since = time.time()

    plt.ion()

    for Gene in range(Generations):

        MatingPool = F_mating(Population, FrontValue, CrowdDistance)

        Offspring = P_generator(MatingPool, Boundary, Coding, PopSize)
        FunctionValue_Offspring = P_objective.P_objective(
            "value", Problem, M, Offspring)

        Population = np.vstack((Population, Offspring))
        FunctionValue = np.vstack((FunctionValue, FunctionValue_Offspring))
        Population, FunctionValue, FrontValue, CrowdDistance, MaxFront = F_EnvironmentSelect(
            Population, FunctionValue, PopSize)

        plt.clf()
        # plt.plot(FunctionValue[:, 0], FunctionValue[:, 1], "*")
        plt.scatter(FunctionValue[:, 0], FunctionValue[:, 1])
        plt.pause(0.001)

        print(Algorithm, "Run :", Gene, "代，Complete：", 100*Gene/Generations,
              "%, time consuming:", np.round(time.time()-since, 2), "s")

    FunctionValueNon = FunctionValue[(FrontValue == 1)[0], :]
    plt.plot(FunctionValueNon[:, 0], FunctionValueNon[:, 1], "*")
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    # p = Idct_transform([1,2,3,4,5,6,7,8,1,0,1], 4, 8)

    Generations = 100
    PopSize = 100
    M = 2
    Run = 1
    Algorithm = "NSGA-II"
    Problem = "DTLZ1"
    # profile.run("EA_Run(Generations, PopSize, M, Run, Problem, Algorithm)")
    EA_Run(Generations, PopSize, M, Run, Problem, Algorithm)
