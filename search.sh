#!/bin/sh

python3 Search/evolutionSearch.py --save SEE_seed0 \
                                  --seed 0 \
                                  --generation 30 \
                                  --EmbeddingLoadCheckpoint 2019_08_26_07_35_34 \
                                  --PredictorModelEpoch 20 \
                                  --PredictorModelPath ./Dataset/encodeData/RankModel/ \
                                  --popSize 30 \
                                  --mutationRate 0.8 \
                                  --trainSearch_epoch 30 \
                                  --trainSearch_initChannel 32 \
                                  --trainSearchDataset cifar10 \
                                  --trainSearchDatasetClassNumber 10 \
                                  --trainSearchSurrogate 5 \
                                  --evalMode DEBUG