#!/bin/sh
nvidia-smi

Embedding_Checke_Point_Path=2019_08_26_07_35_34
search_space=Node_Cell
dataset=cifar10

python3 Validation/trainCifar.py  --code_str=Phase:565-942-627-465-742-441-262-663-208-711-065-861-284-788-325-Phase:645-703-557-802-760-512-294-012-232-585-161-526-622-286-341 \
                                  --save=ValidationCifar_$search_space\_$dataset \
                                  --seed=0 \
                                  --data_worker=12 \
                                  --dataset=cifar10 \
                                  --search_space=$search_space \
                                  --batch_size=128 \
                                  --eval_batch_size=250 \
                                  --epoch=600 \
                                  --learning_rate=0.025 \
                                  --keep_prob=0.6 \
                                  --drop_path_keep_prob=0.8 \
                                  --weight_decay=3e-4 \
                                  --auxiliary_weight=0.4 \
                                  --layers=6 \
                                  --init_channels=36