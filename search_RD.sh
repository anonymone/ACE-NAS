#!/bin/sh
:<<Experiment_NOTE
[Experiment NOTE]
-This area will not execute which only used to record the experiment description.
Experiment_NOTE

MODE=EXPERIMENT
DEVICE=cuda:3

python3 search_RD.py --seed=5 \
                    --save_root=./Experiments/ \
                    --pop_size=1200 \
                    --layers=1 \
                    --channels=16 \
                    --keep_prob=0.6 \
                    --drop_path_keep_prob=0.8 \
                    --classes=10 \
                    --obj_num=2 \
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
                    --device=$DEVICE
