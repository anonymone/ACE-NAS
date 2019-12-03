#!/bin/sh
nvidia-smi

MODE = EXPERIMENT

python3 search_EA.py --seed=0 \
                    --save_root=./Experiments/ \
                    --generations=24 \
                    --layers=1 \
                    --channels=24 \
                    --keep_prob=0.6 \
                    --drop_path_keep_prob=0.8 \
                    --classes=10 \
                    --pop_size=30 \
                    --obj_num=2 \
                    --mutate_rate=1 \
                    --crossover_rate=0.8 \
                    --mode= $MODE \
                    --data_path=./Dataset/ \
                    --num_work=10 \
                    --train_batch_size=196 \
                    --eval_batch_size=196 \
                    --l2_reg=3e-4 \
                    --momentum=0.9 \
                    --lr_min=0.001 \
                    --lr_max=0.025 \
                    --epochs=25 \
                    --split_train_for_valid=0.8