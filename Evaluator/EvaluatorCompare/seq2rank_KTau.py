import sys
sys.path.append('./')

import os
import argparse
import pandas as pd
import logging

from Evaluator.Utils.surrogate import EmbeddingModel as em
from Evaluator.Utils.surrogate import RankNetDataset, Seq2Rank
from Evaluator.Utils.recoder import create_exp_dir

parser = argparse.ArgumentParser('KTau Calculation.')
parser.add_argument('--ckpt_root', type=str, default='./Res/nasbench/seq2seq/checkpoints/')
parser.add_argument('--ckpt_name', type=str, default='2020_02_21_12_50_48')
parser.add_argument('--dataset_root', type=str, default='./Res/nasbench/')
parser.add_argument('--dataset_name', type=str, default='0.001')
parser.add_argument('--exp_path', type=str, default='./Experiments/NASBench/')
args = parser.parse_args()

# use the acc of 108 epochs
# 4 : -4
# 12: -3
# 36: -2
# 108: -1
select_acc = -1

create_exp_dir(args.exp_path)
args.exp_path = os.path.join(args.exp_path, args.dataset_name)
create_exp_dir(args.exp_path)

# logging setting
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s %(message)s")
fh = logging.FileHandler(os.path.join(args.exp_path, 'experiments.log'))
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.addHandler(fh)

# init model
encoder = em(model_path=args.ckpt_root, model_file=args.ckpt_name)
seq2rank = Seq2Rank(encoder, model_save_path=os.path.join(args.exp_path, 'seq2rank_checkpoint'),
                    input_preprocess=lambda x: x,
                    device='cuda:0',
                    model_size=[(64, 32),(32,16), (16, 1)])

# load data set
data_train = pd.read_csv(os.path.join(args.dataset_root, args.dataset_name, 'train.csv'))
data_valid = pd.read_csv(os.path.join(args.dataset_root, args.dataset_name, 'valid.csv'))
train_input, train_label = data_train.values[:,1], data_train.values[:,select_acc]
valid_input, valid_label = data_valid.values[:,1], data_valid.values[:,select_acc]
data_train = [i for i in zip(train_input, train_label)]
data_valid = [i for i in zip(valid_input, valid_label)]
del train_input, train_label, valid_input, valid_label


seq2rank.update_dataset(data_train)
seq2rank.train(train_epoch=50, run_time=int(float(args.dataset_name)*423000),batch_size=96, num_workers=10)

# prepare dataset
eval_data = RankNetDataset()
eval_data.add_data(seq2rank.encoder.encode2numpy(data_valid))

valid_loss, valid_top1, valid_top5 = seq2rank.eval(eval_data,batch_size=96, num_workers=10)

