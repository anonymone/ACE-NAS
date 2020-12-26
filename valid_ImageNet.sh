#!/bin/sh
:<<Experiment_NOTE
[Experiment NOTE]
-This area will not execute which only used to record the experiment description.
18354017523187294391178690126379617177,4.1.9-2.1.12-2.6.11-2.3.13-2.12.6-1.2.10-0.7.1-0.2.6-3.8.5-0.10.1-0.7.6-0.5.15-2.3.12-2.5.11-3.11.11-4.6.12-3.6.2-0.4.7-1.11.12-1.9.4-2.13.5-3.1.13-2.8.2-1.11.2-2.11.3<--->3.2.6-2.11.4-0.5.7-2.7.13-1.11.6-2.8.2-0.13.2-2.8.8-1.11.2-2.2.4
Experiment_NOTE

DATASET=CIFAR10
DEVICE=cuda:0
ENCODE="4.1.9-2.1.12-2.6.11-2.3.13-2.12.6-1.2.10-0.7.1-0.2.6-3.8.5-0.10.1-0.7.6-0.5.15-2.3.12-2.5.11-3.11.11-4.6.12-3.6.2-0.4.7-1.11.12-1.9.4-2.13.5-3.1.13-2.8.2-1.11.2-2.11.3<--->3.2.6-2.11.4-0.5.7-2.7.13-1.11.6-2.8.2-0.13.2-2.8.8-1.11.2-2.2.4"

python3 valid_ImageNet.py  --seed=0 \
                        --save_root=./Experiments/ \
                        --data_path=./Res/Dataset/ImageNet/ \
                        --feed_num_work=12 \
                        --load_num_work=10 \
                        --train_batch_size=96 \
                        --eval_batch_size=96 \
                        --layers=6 \
                        --channels=48 \
                        --epochs=600 \
                        --device=$DEVICE \
                        --grad_clip=5.0 \
                        --label_smooth=0.1 \
                        --gamma=0.97 \
                        --momentum=0.9 \
                        --lr_max=0.025 \
                        --keep_prob=0.6 \
                        --drop_path_keep_prob=0.8 \
                        --l2_reg=3e-5 \
                        --encoding_str=$ENCODE
