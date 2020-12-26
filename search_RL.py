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
from Evaluator.Utils import recoder
from Evaluator.RL_evaluator import RL_eval
from SearchEngine.RL_Engine import RL_population, Q_State_Enumerator, Q_learning, Q_State
from SearchEngine.Utils.RL_tools import ACE_parser_tool_RL as ACE_parser_tool

# Experiments parameter settings
parser = argparse.ArgumentParser(
    "RL based Neural Architecture Search Experiments")
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--save_root', type=str, default='./Experiments/')
# encoding setting
parser.add_argument('--unit_num', default=(15, 25))
parser.add_argument('--value_boundary', default=(0, 10))
# model setting
parser.add_argument('--layers', type=int, default=1)
parser.add_argument('--channels', type=int, default=24)
parser.add_argument('--keep_prob', type=float, default=0.6)
parser.add_argument('--drop_path_keep_prob', type=float, default=0.8)
parser.add_argument('--use_aux_head', type=bool, default=False)
parser.add_argument('--classes', type=int, default=10)
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
# Rl setting
parser.add_argument('--epsilon_list', type=dict, default={1.0: 250,
                                                          0.9: 100,
                                                          0.8: 100,
                                                          0.7: 100,
                                                          0.6: 100,
                                                          0.5: 100,
                                                          0.4: 100,
                                                          0.3: 100,
                                                          0.2: 100,
                                                          0.1: 100})
parser.add_argument('--q_lr', type=float, default=0.1)
parser.add_argument('--q_discount_factor', type=float, default=1.0)
parser.add_argument('--q_random_sample', type=int, default=100)

args = parser.parse_args()

recoder.create_exp_dir(args.save_root)
args.save_root = os.path.join(
    args.save_root, 'RL_SEARCH_{0}'.format(time.strftime("%Y%m%d-%H-%S")))
recoder.create_exp_dir(args.save_root, scripts_to_save=glob.glob('*_RL.*'))

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
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

engine = Q_learning(epsilon=1,
                    save_path=args.save_root,
                    state_format=Q_State_Enumerator(args.value_boundary),
                    q_lr=args.q_lr,
                    q_discount_factor=args.q_discount_factor,
                    q_table=None,
                    max_actions=args.unit_num[1]*2) # we have two cells.

model_gallery = RL_population(obj_number=1,
                              ind_params=args,
                              pop_size=0)

evaluator = RL_eval(save_root=args.save_root,
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
                    epochs=args.epochs,
                    device=args.device)

# Expelliarmus
q = Quotes()

total_time = 0
for epsilon, samples_num in args.epsilon_list.items():
    logging.info('[Epsilon {0:.1f}] {2} -- {1}'.format(epsilon, *q.random()))
    epsilon_total_time = 0
    for s in range(samples_num):
        # record time cost
        s_time = time.time()

        model = build_ACE(fitness_size=1, ind_params=args)
        encoding = ACE_parser_tool.states_to_numpy(
            engine.generate_encode(epsilon=epsilon))
        model.set_dec(encoding)
        results = model_gallery.is_exist(model)
        if results is None:
            results = evaluator.evaluate(model)
            model_gallery.add_ind(model,
                                  epsilon,
                                  results['accTop1'],
                                  results['accTop5'],
                                  results['params'][0],
                                  results['FLOPs'])
        # random sample from existed samples.
        results_seqence = [(ACE_parser_tool.numpy_to_states(model.get_dec(), state_format=Q_State), results['accTop1'])]
        if epsilon < 1:
            for sample_dict in model_gallery.random_sample(sample_num=args.q_random_sample):
                encoding_string, reward = sample_dict['encoding_string'], sample_dict['accTop1']
                results_seqence.append((ACE_parser_tool.string_to_states(encoding_string,state_format=Q_State), reward))

        engine.update_q_table_seqence(results_seqence)

        # time record
        s_time = (time.time() - s_time)/60.0
        epsilon_total_time += s_time
        logging.debug("[Epsilon {0:.1f}][{1:>2d}/{2:>2d}] time cost {3:.2f}mins time left {4:.2f}mins".format(
            epsilon, s+1, samples_num, s_time, epsilon_total_time/(s+1)*(samples_num - (s+1))))

    model_gallery.save(save_path=os.path.join(args.save_root, 'samples'),
                       file_name='samples_after_epsilon{0:_>1f}'.format(epsilon), epsilon=epsilon)

    total_time += epsilon_total_time
    logging.info("[Epsilon {0:.1f} End] total Cost {1:.2f}h".format(
        epsilon, total_time/60.0))

# save final q_values
engine.save_q_table(args.save_root, 'q_value_fianl_', 'csv')
