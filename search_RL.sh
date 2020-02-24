#!/bin/sh
:<<Experiment_NOTE
[Experiment NOTE]
-This area will not execute which only used to record the experiment description.
Experiment_NOTE

MODE=EXPERIMENT
DEVICE=cuda:0

python3 search_RL.py --seed=0 \
                    --save_root=./Experiments/ \
                    --layers=1 \
                    --channels=16 \
                    --keep_prob=0.6 \
                    --drop_path_keep_prob=0.8 \
                    --classes=10 \
                    --mode=$MODE \
                    --data_path=./Res/Dataset/ \
                    --num_work=10 \
                    --train_batch_size=196 \
                    --eval_batch_size=196 \
                    --l2_reg=3e-4 \
                    --momentum=0.9 \
                    --lr_min=0.001 \
                    --lr_max=0.1 \
                    --epochs=25 \
                    --split_train_for_valid=0.8 \
                    --q_lr=0.1 \
                    --q_discount_factor=1 \
                    --q_random_sample=100 \
                    --device=$DEVICE