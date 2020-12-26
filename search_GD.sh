#!/bin/sh
:<<Experiment_NOTE
[Experiment NOTE]
-This area will not execute which only used to record the experiment description.
Experiment_NOTE

MODE=EXPERIMENT
DEVICE=cuda:0

python3 search_GD.py --seed=0 \
                    --search_pop_num=1000 \
                    --save_root=./Experiments/ \
                    --data_path=./Res/Dataset/ \
                    --layers=1 \
                    --channels=16 \
                    --keep_prob=0.6 \
                    --drop_path_keep_prob=0.8 \
                    --classes=10 \
                    --pop_size=300 \
                    --mode=$MODE \
                    --num_work=10 \
                    --train_batch_size=196 \
                    --eval_batch_size=196 \
                    --split_train_for_valid=0.8 \
                    --l2_reg=3e-4 \
                    --momentum=0.8 \
                    --lr_max=0.1 \
                    --lr_min=0 \
                    --epochs=25 \
                    --controller_seed_arch=100 \
                    --controller_new_arch=300 \
                    --controller_source_length=120 \
                    --controller_encoder_length=60 \
                    --controller_decoder_length=120 \
                    --controller_encoder_vocab_size=15 \
                    --controller_decoder_vocab_size=15 \
                    --controller_batch_size=100 \
                    --controller_epochs=500 \
                    --device=$DEVICE