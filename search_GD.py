import argparse
import time
import logging
import numpy as np
import torch
import random
import sys
import os
import glob
from quotes import Quotes

from Coder.ACE import build_ACE
from SearchEngine.GD_Engine import NAO, GD_population
from SearchEngine.Utils.GD_tools import dataset_utils, nao_train, nao_valid, nao_infer
from Evaluator.EA_evaluator import EA_eval
from Evaluator.Utils import recoder

# Experiments parameter settings
parser = argparse.ArgumentParser(
    "GD based Neural Architecture Search Experiments")
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--save_root', type=str, default='./Experiments/')
# encoding setting
parser.add_argument('--unit_num', default=(20, 20))
parser.add_argument('--value_boundary', default=(0, 15))
# model setting
parser.add_argument('--layers', type=int, default=1)
parser.add_argument('--channels', type=int, default=24)
parser.add_argument('--keep_prob', type=float, default=0.6)
parser.add_argument('--drop_path_keep_prob', type=float, default=0.8)
parser.add_argument('--use_aux_head', type=bool, default=False)
parser.add_argument('--classes', type=int, default=10)
# population setting
parser.add_argument('--pop_size', type=int, default=300)
parser.add_argument('--obj_num', type=int, default=2)
parser.add_argument('--search_pop_num', type=int, default=1000)
# eval setting
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--mode', type=str, default='DEBUG')
parser.add_argument('--data_path', type=str, default='./Res/Dataset/')
parser.add_argument('--cutout_size', type=int, default=None)
parser.add_argument('--num_work', type=int, default=0)
parser.add_argument('--train_batch_size', type=int, default=196)
parser.add_argument('--eval_batch_size', type=int, default=196)
parser.add_argument('--split_train_for_valid', type=float, default=None)
parser.add_argument('--l2_reg', type=float, default=3e-4)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--lr_min', type=float, default=0.001)
parser.add_argument('--lr_max', type=float, default=0.025)
parser.add_argument('--epochs', type=int, default=25)
# Engine Setting
parser.add_argument('--controller_seed_arch', type=int, default=100)
# parser.add_argument('--controller_expand', type=int, default=10)
parser.add_argument('--controller_new_arch', type=int, default=300)
parser.add_argument('--controller_encoder_layers', type=int, default=1)
parser.add_argument('--controller_encoder_hidden_size', type=int, default=64)
parser.add_argument('--controller_encoder_emb_size', type=int, default=32)
parser.add_argument('--controller_mlp_layers', type=int, default=0)
parser.add_argument('--controller_mlp_hidden_size', type=int, default=200)
parser.add_argument('--controller_decoder_layers', type=int, default=1)
parser.add_argument('--controller_decoder_hidden_size', type=int, default=64)
parser.add_argument('--controller_source_length', type=int, default=120)
parser.add_argument('--controller_encoder_length', type=int, default=60)
parser.add_argument('--controller_decoder_length', type=int, default=120)
parser.add_argument('--controller_encoder_dropout', type=float, default=0)
parser.add_argument('--controller_mlp_dropout', type=float, default=0.1)
parser.add_argument('--controller_decoder_dropout', type=float, default=0)
parser.add_argument('--controller_l2_reg', type=float, default=0)
parser.add_argument('--controller_encoder_vocab_size', type=int, default=15)
parser.add_argument('--controller_decoder_vocab_size', type=int, default=15)
parser.add_argument('--controller_trade_off', type=float, default=0.8)
parser.add_argument('--controller_epochs', type=int, default=5)
parser.add_argument('--controller_batch_size', type=int, default=100)
parser.add_argument('--controller_lr', type=float, default=0.001)
parser.add_argument('--controller_grad_bound', type=float, default=5.0)
parser.add_argument('--controller_optimizer', type=str, default='adam')

args = parser.parse_args()

recoder.create_exp_dir(args.save_root)
args.save_root = os.path.join(
    args.save_root, 'GD_SEARCH_{0}'.format(time.strftime("%Y%m%d-%H-%S")))
recoder.create_exp_dir(args.save_root, scripts_to_save=glob.glob('*_GD.*'))

# logging setting
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s %(message)s")
fh = logging.FileHandler(os.path.join(args.save_root, 'experiments.log'))
fh.setLevel(logging.INFO)
fh.setFormatter(formatter)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.addHandler(fh)

# record settings
logging.info("[Experiments Setting]\n"+"".join(
    ["[{0}]: {1}\n".format(name, value) for name, value in args.__dict__.items()]))

# fix seed
torch.cuda.set_device(args.device)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

population = GD_population(ind_params=args,
                           ind_generator=build_ACE,
                           obj_number=2,
                           pop_size=args.pop_size)

evaluator = EA_eval(save_root=args.save_root,
                    mode=args.mode,
                    data_path=args.data_path,
                    cutout_size=args.cutout_size,
                    num_work=args.num_work,
                    train_batch_size=args.train_batch_size,
                    eval_batch_size=args.eval_batch_size,
                    split_train_for_valid=args.split_train_for_valid,
                    l2_reg=args.l2_reg,
                    momentum=args.momentum,
                    lr_min=args.lr_min,
                    lr_max=args.lr_max,
                    epochs=args.epochs)

engine = NAO(
    args.controller_encoder_layers,
    args.controller_encoder_vocab_size,
    args.controller_encoder_hidden_size,
    args.controller_encoder_dropout,
    args.controller_encoder_length,
    args.controller_source_length,
    args.controller_encoder_emb_size,
    args.controller_mlp_layers,
    args.controller_mlp_hidden_size,
    args.controller_mlp_dropout,
    args.controller_decoder_layers,
    args.controller_decoder_vocab_size,
    args.controller_decoder_hidden_size,
    args.controller_decoder_dropout,
    args.controller_decoder_length).cuda()

# Expelliarmus
q = Quotes()

total_time = []
total_run_time = 0
while total_run_time < 100:
    # time cost record
    s_time = time.time()

    logging.info("[Search Epoch {0:>2d}] {2} -- {1}".format(total_run_time, *q.random()))
    evaluator.evaluate(population.get_ind())
    population.save(save_path=os.path.join(
        args.save_root, 'archs'), file_name='arch_{0:_>2d}'.format(total_run_time))

    s_time = time.time() - s_time
    total_time += [s_time]
    logging.info("[Evaluating End] time cost {0:.2f}h".format(s_time/3600.0))

    if population.pop_size > args.search_pop_num:
        logging.info("[All Finished in {0:.2f}d Total evaluate {1:4>d} samples]".format(sum(total_time)/(3600.0*24), population.pop_size))
        break

    s_time = time.time()

    arch_pool, arch_pool_valid_acc = population.to_matrix()
    arch_pool, arch_pool_valid_acc = dataset_utils.sort(
        arch_pool, arch_pool_valid_acc)
    inputs, labels = dataset_utils.normalize(arch_pool, arch_pool_valid_acc)
    trainset, validset = dataset_utils.get_dataset(inputs, labels)
    train_queue, valid_queue = dataset_utils.get_data_loader(train_dataset=trainset,
                                                             valid_dataset=validset,
                                                             train_batch_size=args.controller_batch_size,
                                                             eval_batch_size=args.controller_batch_size)

    optimizer = torch.optim.Adam(
        engine.parameters(), lr=args.controller_lr, weight_decay=args.controller_l2_reg)

    logging.info("[Train Engine] {1} -- {0}".format(*q.random()))
    for epoch in range(1, args.controller_epochs+1):
        loss, mse, ce = nao_train(train_queue, engine, optimizer, controller_trade_off=args.controller_trade_off,
                                  controller_grad_bound=args.controller_grad_bound)
        logging.debug("[Engine Train] [{0:>3d}/{1:>3d}] train loss {2:.6f} mse {3:.6f} ce {4:.6f}".format(
            epoch, args.controller_epochs, loss, mse, ce))
        if epoch % 100 == 0:
            pa, hs = nao_valid(valid_queue, engine)
            logging.info("[Engine Valid] [{0:>3d}/{1:>3d}] pairwise accuracy {2:.6f} hamming distance {3:.6f}".format(
                epoch, args.controller_epochs, pa, hs))
    
    s_time = time.time() - s_time
    total_time += [s_time]
    logging.info("[Train Engine End] time cost {0:.2f}mins".format(s_time/60.0))

    # Generate new encoding
    s_time = time.time()

    new_archs = []
    max_step_size = 100
    predict_step_size = 0
    top100_archs = (inputs[:100], labels[:100])
    infer_queue, _ = dataset_utils.get_data_loader(train_dataset=top100_archs)

    while len(new_archs) < args.controller_new_arch:
        predict_step_size += 1
        new_arch = nao_infer(infer_queue, engine,
                             predict_step_size, direction='+')
        for arch in new_arch:
            if arch not in arch_pool and arch not in new_archs:
                new_archs.append(arch)
            if len(new_archs) >= args.controller_new_arch:
                break
        if predict_step_size > max_step_size:
                break
    population.add_new_inds(new_archs)

    s_time = time.time() - s_time
    total_time += [s_time]
    logging.info("[Generate New Encoding] {0:>2d} new samples are added. in {1:.2f}min".format(len(new_archs), s_time/60.0))
    logging.info("[Search Epoch {0:>2d} End] time cost {1:.2f}h total time cost {2:.2f}d".format(total_run_time, sum(total_time[-3:])/3600.0, sum(total_time)/(3600.0*24)))
    total_run_time += 1

# Select the final Top 5 models
# arch_pool, arch_pool_valid_acc = population.to_matrix()
# arch_pool, arch_pool_valid_acc = dataset_utils.sort(arch_pool, arch_pool_valid_acc)
