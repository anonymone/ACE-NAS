#!/bin/sh
:<<Experiment_NOTE
[Experiment NOTE]
-This area will not execute which only used to record the experiment description.
Experiment_NOTE
search err 15.26, 0.15M

DATASET=CIFAR10
ENCODE="0.3.1-10.11.9-14.14.4-14.13.11-1.6.0-1.12.2-10.13.5-1.13.11-5.0.12-1.13.0-1.0.9-12.2.6-2.11.4<--->1.8.14-9.13.11-0.10.7-1.3.4-7.8.6-1.11.0-3.5.2-3.12.14-10.15.8-2.12.11-6.13.0-2.4.5-1.8.13-6.8.8-2.3.0-1.2.2"

python3 valid_CIFAR.py  --seed=0 \
                        --save_root=./Experiments/ \
                        --data_path=./Res/Dataset/ \
                        --dataset=$DATASET \
                        --num_work=12 \
                        --train_batch_size=128 \
                        --eval_batch_size=256 \
                        --cutout_size=16 \
                        --layers=6 \
                        --channels=36 \
                        --epochs=600 \
                        --device=cuda \
                        --lr_max=0.025 \
                        --lr_min=0.0 \
                        --momentum=0.9 \
                        --keep_prob=0.6 \
                        --drop_path_keep_prob=0.8 \
                        --l2_reg=3e-4 \
                        --encoding_str=$ENCODE