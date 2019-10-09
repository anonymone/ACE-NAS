import sys
# update your projecty root path before running
sys.path.insert(0, './')

import os
import time
import logging
import argparse

import torch
from torch.optim.lr_scheduler import StepLR
import torchtext
import pandas as pd
from copy import deepcopy

import seq2seq
from seq2seq.trainer import SupervisedTrainer
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq
from seq2seq.loss import Perplexity
from seq2seq.optim import Optimizer
from seq2seq.dataset import SourceField, TargetField
from seq2seq.evaluator import Predictor
from seq2seq.util.checkpoint import Checkpoint

from misc import utils
from misc import evo_operator
import numpy as np
from Search import trainSearch
from EvolutionAlgorithm.NSGA2 import NSGA2
from Model.individual import SEEPopulation
from Model.embeddingModel import EmbeddingModel as em
from Model.surroogate import Predictor, RankNetDataset

parser = argparse.ArgumentParser("Multi-objetive Genetic Algorithm for WF-BEE with SG")
parser.add_argument('--save', type=str, default='SEE_Exp', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--generation', type=int, default=30, help='random seed')

# Embedding model setting
parser.add_argument('--Embedding_TrainPath', action='store', dest='train_path', default='./Dataset/encodeData/data.txt',help='Path to train data')
parser.add_argument('--Embedding_DevPath', action='store', dest='dev_path', default='./Dataset/encodeData/data_val.txt',help='Path to dev data')
parser.add_argument('--Embedding_ExptDir', action='store', dest='expt_dir', default='./Dataset/encodeData',help='Path to experiment directory. If load_checkpoint is True, then path to checkpoint directory has to be provided')
parser.add_argument('--Embedding_LoadCheckpoint', action='store', dest='load_checkpoint', default='2019_08_26_07_35_34',help='The name of the checkpoint to load, usually an encoded time string')
parser.add_argument('--Embedding_Resume', action='store_true', dest='resume',default=False,help='Indicates if training has to be resumed from the latest checkpoint')

# Predictor model setting 
parser.add_argument('--PredictorModelDataset', dest='predictDataset', default='./Dataset/encodeData/surrogate.txt')
parser.add_argument('--PredictorModelPath', dest='predictPath', default='./Dataset/encodeData/RankModel/')
parser.add_argument('--PredictorModelEpoch', dest='predictEpoch', default= 20)
parser.add_argument('--PredictorSelectNumberofIndividuals', dest='predictSelectNum', default= 2)
parser.add_argument('--PredictorSearchEpoch', dest='predictSearchEpoch', default= 100)

# population setting
parser.add_argument('--popSize', type=int, default=30, help='The size of population.')
parser.add_argument('--objSize', type=int, default=2,help='The number of objectives.')
parser.add_argument('--blockLength', type=tuple, default=(2, 15, 3),help='A tuple containing (phase, unit number, length of unit)')
parser.add_argument('--valueBoundary', type=tuple,default=(0, 9), help='Decision value bound.')
parser.add_argument('--crossoverRate', type=float, default=0.1,help='The propability rate of crossover.')
parser.add_argument('--mutationRate', type=float, default=1,help='The propability rate of crossover.')

# train search method setting.
parser.add_argument('--dataRoot', type=str,default='./Dataset', help='The root path of dataset.')
parser.add_argument('--trainSearch_exprRoot', type=str,default='./Experiments/model', help='the root path of experiments.')
parser.add_argument('--trainSearch_initChannel', type=int,default=16, help='# of filters for first cell')
parser.add_argument('--trainSearch_layers', type=int, default=3)
parser.add_argument('--trainSearch_epoch', type=int, default=30,help='# of epochs to train during architecture search')
parser.add_argument('--trainSearch_drop_path_keep_prob', type=float, default=8.0)
parser.add_argument('--trainSearch_keep_prob', type=float, default=0.8)
parser.add_argument('--trainSearchDataset', type=str,default='cifar10', help='The name of dataset.')
parser.add_argument('--trainSearchDatasetClassNumber', type=int,default=10, help='The classes number of dataset.')
parser.add_argument('--trainSearch_save', type=str,default='SEE_#id', help='the filename including each model.')
parser.add_argument('--trainSearch_preLoad', type=bool, default=True, help='load the fixed population.')
parser.add_argument('--trainSearch_dropPathProb',type=float, default=0.0, help='')
parser.add_argument('--trainSearch_cutout', type=bool, default=False, help='')
parser.add_argument('--trainSearchSurrogate', type=int, dest='trainSGF',default=5, help='the frequence of evaluation by surrogate.')

parser.add_argument('--trainSearch_auxiliary',type=bool, default=False, help='')
# testing setting
# DEBUG is replace all evaluation 
# FAST is load prepared Data
parser.add_argument('--evalMode', type=str, default='FAST', help='Evaluating mode for testing usage.')

args = parser.parse_args()
args.save = './Experiments/search-{}-{}'.format(
    args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save)
args.trainSearch_exprRoot = os.path.join(args.save, "model")
utils.create_exp_dir(args.trainSearch_exprRoot)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
# set seed
np.random.seed(args.seed)
pop_hist = []  # keep track of every evaluated architecture

# recording setting
logging.info("args = %s", args)

# init population
Engine = NSGA2.NSGA2()
population = SEEPopulation(popSize=args.popSize, crossover=evo_operator.SEE_Cell_CrossoverV1,
                           objSize=args.objSize, blockLength=args.blockLength,
                           valueBoundary=args.valueBoundary, mutation=evo_operator.SEEMutationV1,
                           evaluation=trainSearch.main,args=args)

# init the encoder mdoel
embedModel = em(opt=args)

if args.trainSearch_preLoad:
    # we fix the initialization in general experiments
    population.load(path='./Dataset/initialization/Generation-init.csv',
                    objSize=args.objSize,
                    blockLength=args.blockLength,
                    valueBoundary=args.valueBoundary)
else:
    # evaluation
    population.evaluation()
    
population.save(os.path.join(args.save, 'Generation-{0}'.format('init')))
enCodeNumpy  =  embedModel.encode2numpy(population.toString())

# load the  predictor model
# PredicDataset = RankNetDataset(pd.read_csv(args.predictDataset))
predicDataset = RankNetDataset()
predicDataset.addData(enCodeNumpy[:,:-1])

# need to check wether the fitness is select correctly.
predictor = Predictor(encoder=embedModel, args= args,modelSavePath=args.predictPath)
# predictor.trian(dataset=predicDataset, trainEpoch=args.predictEpoch)

# used for recording hv
hv = []

for generation in range(args.generation):
    # record the generation where is applying the real evaluation method.
    realTrainPoint = [ x for x in range(0, args.generation + 1, args.trainSGF)]
    # create the new model file
    logging.info("===========================Generatiion {0}===========================".format(generation))
    if generation in realTrainPoint:
        # the real evaluation 
        if args.evalMode == "FAST" and os.path.exists('./Dataset/initialization/Generation-{0}.csv'.format(generation)):
            population.load(path='./Dataset/initialization/Generation-0.csv',
                            objSize=args.objSize,
                            blockLength=args.blockLength,
                            valueBoundary=args.valueBoundary,
                            addMode=True)
        else:
            population.newPop(inplace=True)
            population.evaluation()
        popValue = population.toMatrix()
        # only use the acc, param
        popValue = popValue[:,:-1]
        enCodeNumpy  =  embedModel.encode2numpy(population.toString())
        predicDataset.updateData(enCodeNumpy[:,:-1])
        predictor.trian(dataset=predicDataset, trainEpoch=int(args.predictEpoch), newModel=True)
    else:
        surrogatePop = deepcopy(population)
        # individuals = surrogatePop.individuals
        for surrogateRunTimes in range(args.predictSearchEpoch):
            if (surrogateRunTimes+1)%10 == 0:
                logging.info("=======================Generatiion {0}.{1} with Surrogate=======================".format(generation,surrogateRunTimes+1))
            surrogatePop.newPop(inplace=True)
            popValue = predictor.evaluation(args=args,
                                            individuals=surrogatePop.individuals)
            # test Only use acc.
            index = Engine.enviromentalSeleection(popValue[:,:-1], args.popSize)
            index2 = [x for x in range(surrogatePop.popSize) if x not in index]
            surrogatePop.remove(index2)
        popValue = surrogatePop.toMatrix()
        popValue = popValue[:,[0,-1]]
        index = Engine.enviromentalSeleection(popValue[:,:-1], popNum = args.predictSelectNum)
        theBestInd = surrogatePop.getInd(index)
        theBestInd = surrogatePop.evaluation(individuals=theBestInd)
        population.add(theBestInd)
    popValue = population.toMatrix()
    ##########################
    # -1 : use acc and param.#
    # -2 : only use acc.     #
    ##########################
    popValue = popValue[:,:-1]
    index = Engine.enviromentalSeleection(popValue, args.popSize)
    index2 = [x for x in range(population.popSize) if x not in index]
    population.remove(index2)
    # static the best middle and worrse.
    popValue = population.toMatrix(needDec=False)
    try:
        hv.append(utils.hv_2d(popValue[:,:-1]))
        logging.info("POPULATION Hyper-Volume : {0}".format(hv))
    except:
        logging.warn("Hyper-Volume calculating failed.")
    # best, middle, worrse = np.min(popValue[:,1]),np.min(popValue[:,2])
    population.save(os.path.join(
        args.save, 'Generation-{0}'.format(generation)))
