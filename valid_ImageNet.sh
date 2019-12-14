#!/bin/sh
:<<Experiment_NOTE
[Experiment NOTE]
-This area will not execute which only used to record the experiment description.
Experiment_NOTE

DATASET=CIFAR10
ENCODE=

python3 valid_CIFAR.py  --seed=0 \
                        --save_root=./Experiments/ \
                        --encode_str=$ENCODE
                        --data_path=./Dataset/ \
                        --num_work=10 \
                        --train_batch_size=128 \
                        --eval_batch_size=256 \
                        --cutout_size=16 \
                        --layers=6 \
                        --channels=48 \
                        --epochs=600 \
                        --device=cuda \
                        --grad_clip=5.0 \
                        --label_smooth=0.1 \
                        --gamma=0.97 \
                        --momentum=0.9 \
                        --lr_max=0.025 \
                        --keep_prob=0.6 \
                        --drop_path_keep_prob=0.8 \
                        --l2_reg=3e-5 \
                        --encoding_str=$ENCODE