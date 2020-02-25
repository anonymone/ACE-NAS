#!/bin/sh
:<<Experiment_NOTE
[Experiment NOTE]
-This area will not execute which only used to record the experiment description.
Experiment_NOTE

MODE=EXPERIMENT
DEVICE=cuda:0

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
                    --surrogate_path=./Res/PretrainModel/ \
                    --surrogate_premodel=2019_12_28_06_03_12 \
                    --surrogate_step=5 \
                    --surrogate_search_times=1000 \
                    --surrogate_preserve_topk=2 \
                    --device=$DEVICE