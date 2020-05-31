import argparse
import time
import logging
import numpy as np
import torch
import random
import sys
import os
import glob
from copy import deepcopy
#from quotes import Quotes

from Coder.ACE import build_ACE
from SearchEngine.EA_Engine import NSGA2, EA_population
from SearchEngine.Utils import EA_tools
from Evaluator.EA_evaluator import EA_eval
from Evaluator.Utils import recoder

from Evaluator.Utils.surrogate import EmbeddingModel as em
from Evaluator.Utils.surrogate import RankNetDataset, Seq2Rank

# Experiments parameter settings
parser = argparse.ArgumentParser(
    "EA based Neural Architecture Search Experiments")
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--save_root', type=str, default='./Experiments/')
parser.add_argument('--generations', type=int, default=30)
# encoding setting
parser.add_argument('--unit_num', default=(10, 20))
parser.add_argument('--value_boundary', default=(0, 15))
# model setting
parser.add_argument('--layers', type=int, default=1)
parser.add_argument('--channels', type=int, default=16)
parser.add_argument('--keep_prob', type=float, default=0.6)
parser.add_argument('--drop_path_keep_prob', type=float, default=0.8)
parser.add_argument('--use_aux_head', type=bool, default=False)
parser.add_argument('--classes', type=int, default=10)
# population setting
parser.add_argument('--pop_size', type=int, default=30)
parser.add_argument('--obj_num', type=int, default=2)
parser.add_argument('--mutate_rate', type=float, default=0.5)
parser.add_argument('--crossover_rate', type=float, default=1)
# eval setting
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--mode', type=str, default='EXPERIMENT')
parser.add_argument('--data_path', type=str, default='./Res/Dataset/')
parser.add_argument('--cutout_size', type=int, default=None)
parser.add_argument('--num_work', type=int, default=0)
parser.add_argument('--train_batch_size', type=int, default=196)
parser.add_argument('--eval_batch_size', type=int, default=196)
parser.add_argument('--split_train_for_valid', type=float, default=0.9)
parser.add_argument('--small_set', type=float, default=0.2)
parser.add_argument('--l2_reg', type=float, default=3e-4)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--lr_min', type=float, default=0.001)
parser.add_argument('--lr_max', type=float, default=0.025)
parser.add_argument('--epochs', type=int, default=25)

# surrogate
parser.add_argument('--surrogate_allowed', type=recoder.args_bool, default='True')
parser.add_argument('--surrogate_path', type=str,
                    default='./Res/PretrainModel/')
parser.add_argument('--surrogate_premodel', type=str,
                    default='2020_03_10_09_29_34')
parser.add_argument('--surrogate_step', type=int, default=5)
parser.add_argument('--surrogate_search_times', type=int, default=10)
parser.add_argument('--surrogate_preserve_topk', type=int, default=5)

args = parser.parse_args()

recoder.create_exp_dir(args.save_root)
args.save_root = os.path.join(
    args.save_root, 'EA_SEARCH_{0}'.format(time.strftime("%Y%m%d-%H-%S")))
recoder.create_exp_dir(args.save_root, scripts_to_save=glob.glob('*_EA.*'))

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

population = EA_population(obj_number=args.obj_num,
                           pop_size=args.pop_size,
                           mutation_rate=args.mutate_rate,
                           crossover_rate=args.crossover_rate,
                           mutation=EA_tools.ACE_Mutation_V3,
                           crossover=EA_tools.ACE_CrossoverV2,
                           ind_generator=build_ACE,
                           ind_params=args)

evaluator = EA_eval(save_root=args.save_root,
                    mode=args.mode,
                    data_path=args.data_path,
                    cutout_size=args.cutout_size,
                    num_work=args.num_work,
                    train_batch_size=args.train_batch_size,
                    eval_batch_size=args.eval_batch_size,
                    split_train_for_valid=args.split_train_for_valid,
                    small_set=args.small_set,
                    l2_reg=args.l2_reg,
                    momentum=args.momentum,
                    lr_min=args.lr_min,
                    lr_max=args.lr_max,
                    epochs=args.epochs)
engine = NSGA2


if args.surrogate_allowed:
    # init surrogate
    encoder = em(device=args.device,
                 model_path=args.surrogate_path,
                 model_file=args.surrogate_premodel)
    recoder.create_exp_dir(os.path.join(args.save_root, 'seq2rank_checkpoint'))
    seq2rank = Seq2Rank(encoder, model_save_path=os.path.join(args.save_root, 'seq2rank_checkpoint'),
                        input_preprocess=lambda x: x.replace('-', ' ').replace('<   >', ' <---> '),
                        device=args.device)
    surrogate_schedule = [i for i in range(args.generations) if i not in [
        x for x in range(0, args.generations, args.surrogate_step)]]
    surrogate_schedule.remove(args.generations-1)
else:
    surrogate_schedule = []

# Expelliarmus
#q = Quotes()

total_time = 0
for gen in range(args.generations):
    # record time cost
    s_time = time.time()
    logging.info("[Generation{0:>2d}]".format(gen))

    # search by surrogate
    if gen in surrogate_schedule:
        evaluator.set_mode('SURROGATE')
        surrogate_pop = deepcopy(population)
        topk_ind = surrogate_pop.get_topk(k=args.surrogate_preserve_topk)
        surrogate_pop.remove_ind()
        surrogate_pop.add_ind(topk_ind)
        for _ in range(int(args.pop_size/5)):
            surrogate_pop.new_pop()
        for s_gen in range(args.surrogate_search_times):
            surrogate_pop.new_pop()
            evaluator.evaluate(surrogate_pop.get_ind(),
                               surrogate_model=seq2rank)
            _, rm_inds = engine.enviromentalSeleection(
                surrogate_pop.to_matrix()[:, [0, -1]], args.pop_size)
            surrogate_pop.remove_ind(rm_inds)
            surrogate_pop.save(save_path=os.path.join(
                args.save_root, 'sg_populations'), file_name='sg_population_gen{0}_s_gen{1}'.format(gen, s_gen), mode='SURROGATE')
        population.add_ind(surrogate_pop.get_topk(
            k=args.surrogate_preserve_topk, obj_select=2))
        evaluator.set_mode(args.mode)
    else:
        population.new_pop()
    evaluator.set_mode(args.mode)
    evaluator.evaluate(population.get_ind())

    if args.surrogate_allowed:
        # train Seq2Rank
        seq2rank.update_dataset([(seq2rank.input_preprocess(
            ind.to_string()), ind.get_fitness()[0]) for ind in population.get_ind()])
        seq2rank.train(run_time=gen)

    _, rm_inds = engine.enviromentalSeleection(
        population.to_matrix(), args.pop_size)
    logging.info('[Removed Individuals]' +
                 "".join(['\n'+str(i) for i in rm_inds]))
    population.remove_ind(rm_inds)
    population.save(save_path=os.path.join(
        args.save_root, 'populations'), file_name='population_{0:_>2d}'.format(gen))

    s_time = (time.time() - s_time)/3600.0
    total_time += s_time
    logging.info("[Generation{0:>2d} END] time cost {1:.2f}h total time cost {2:.2f}d time left {3:.2f}h\n".format(
        gen, s_time, total_time/24.0, (total_time/(gen+1))*(args.generations-gen-1)))
