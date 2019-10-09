#!/bin/sh
nvidia-smi

Embedding_Checke_Point_Path=2019_08_26_07_35_34
evalMode=FAST

python3 Search/evolutionSearch.py --save=Run_$evalMode \
                                  --seed=0 \
                                  --generation=30 \
                                  --Embedding_LoadCheckpoint=$Embedding_Checke_Point_Path \
                                  --PredictorModelEpoch=20 \
                                  --PredictorModelPath=./Dataset/encodeData/RankModel/ \
                                  --popSize=30 \
                                  --mutationRate=1 \
                                  --trainSearch_epoch=30 \
                                  --trainSearch_initChannel=16 \
                                  --trainSearchDataset=cifar10 \
                                  --trainSearchDatasetClassNumber=10 \
                                  --trainSearchSurrogate=5 \
                                  --trainSearch_layers=1 \
                                  --evalMode=$evalMode