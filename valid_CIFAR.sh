#!/bin/sh
:<<Experiment_NOTE
[Experiment NOTE]
-This area will not execute which only used to record the experiment description.
search err 15.26, 0.15M
Experiment_NOTE


DATASET=CIFAR10
DEVICE=cuda:1
ENCODE="3.13.3-0.0.9-0.4.5-3.8.14-0.0.1-1.9.9-4.14.6-2.1.4-0.4.7-2.7.11-6.7.13-2.2.14-6.0.11-3.2.11-6.7.14-2.2.6-1.3.7-1.11.0-5.4.5-1.5.3-3.8.0-1.5.12<--->0.1.13-5.5.0-2.8.8-6.14.3-0.4.10-2.11.3-2.12.5-2.13.9-0.12.1-3.8.11-2.14.13-1.14.14-1.14.8-0.3.7-1.12.1-0.0.7-1.10.9-1.6.8-3.5.9-1.0.0-3.14.5-2.3.6-5.11.11-1.7.7"

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
                        --device=$DEVICE \
                        --lr_max=0.025 \
                        --lr_min=0.0 \
                        --momentum=0.9 \
                        --keep_prob=0.6 \
                        --drop_path_keep_prob=0.8 \
                        --l2_reg=3e-4 \
                        --encoding_str=$ENCODE