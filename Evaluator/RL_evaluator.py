import torch.nn as nn
import torch.utils
import torch.nn.functional as F
from torch import Tensor
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import random
import numpy as np
import logging

from Evaluator.Evaluator_Interface import evaluator
from Evaluator.Utils import dataset, train, recoder
from Evaluator.Utils.recoder import AvgrageMeter, error_rate, accuracy

class RL_eval(evaluator):
    def __init__(self,
                 save_root='./Experiments/',
                 mode='EXPERIMENT',
                 data_path='./Dataset/',
                 cutout_size=16,
                 num_work=10,
                 train_batch_size=36,
                 eval_batch_size=36,
                 split_train_for_valid=None,
                 l2_reg=3e-4,
                 momentum=0.9,
                 lr_min=0.0,
                 lr_max=0.025,
                 epochs=30,
                 epoch=-1,
                 optimizer_state_dict=None,
                 device='cpu'):
        super(RL_eval, self).__init__(save_root=save_root,
                                      mode=mode,
                                      data_path=data_path,
                                      cutout_size=cutout_size,
                                      num_work=num_work,
                                      train_batch_size=train_batch_size,
                                      eval_batch_size=eval_batch_size,
                                      split_train_for_valid=split_train_for_valid,  # dataset parameters
                                      l2_reg=l2_reg,
                                      momentum=momentum,
                                      lr_min=lr_min,
                                      lr_max=lr_max,
                                      epochs=epochs,
                                      epoch=epoch,
                                      optimizer_state_dict=optimizer_state_dict,  # train parameters
                                      device=device)
    
    def evaluate(self, samples):
        results = self.eval_model(samples)
        samples.set_fitness(results['fitness'])
        self.save(samples, results=results)
        return results
      
    def eval_model(self, individual, **kwargs):
        # DEBUG MODE
        if self.mode == 'DEBUG':
            model = individual.get_model(2333).to(self.device)
            n_params = recoder.count_parameters(model)
            valid_top1, valid_top5 = np.random.rand(
                1)*100, np.random.rand(1)*100
            model = train.add_flops_counting_methods(model)
            model.eval()
            model.start_flops_count()
            random_data = torch.randn(1, 3, 32, 32)
            model(torch.autograd.Variable(random_data).to(self.device))
            n_flops = np.round(model.compute_average_flops_cost() / 1e6, 4)
            logging.debug("[DEBUG MODE] [{0}] valid Top1 {1:.2f} valid Top5 {2:.2f} Params {3:.2f}".format(individual.get_Id(), valid_top1.item(), valid_top5.item(), n_params))
            return {
                'FLOPs': n_flops,
                'accTop1': valid_top1.item(),
                'accTop5': valid_top5.item(),
                'params': (n_params, model.channels),
                'architecture': model.to_dot(),
                'fitness': np.array([valid_top1, n_flops]).reshape(-1)
            }
        # EXPERIMENT MODE
        cudnn.enabled = True
        cudnn.benchmark = False
        cudnn.deterministic = True

        trainset, validset = dataset.build_cifar10(
            self.data_path, self.cutout_size, self.num_work, self.train_batch_size, self.eval_batch_size, self.split_train_for_valid)
        steps = int(len(trainset)) * self.epochs
        model = individual.get_model(steps=steps)

        n_params = recoder.count_parameters(model)

        train_criterion, eval_criterion, optimizer, scheduler = train.build_train_utils(
            model, epochs=self.epochs, l2_reg=self.l2_reg, momentum=self.momentum, lr_min=self.lr_min, lr_max=self.lr_max)
        # train procedures
        step = 0
        for epoch in range(self.epochs):
            train_loss, train_top1, train_top5, step = train.train(
                trainset, model, optimizer, step, train_criterion, self.device, rate_static=accuracy)
            logging.debug("[Epoch {0:>4d}] [Train] loss {1:.3f} lr {2:.5f} Accuracy Top1 {3:.2f} Accuracy Top5 {4:.2f}".format(
                epoch, train_loss, scheduler.get_lr()[0], train_top1, train_top5))
            scheduler.step()

        valid_loss, valid_top1, valid_top5 = train.valid(
            validset, model, eval_criterion, self.device, rate_static=accuracy)
        # calculate for flopss1
        model = train.add_flops_counting_methods(model)
        model.eval()
        model.start_flops_count()
        random_data = torch.randn(1, 3, 32, 32)
        model(torch.autograd.Variable(random_data).to(self.device))
        n_flops = np.round(model.compute_average_flops_cost() / 1e6, 4)

        logging.info("[Valid Accuracy] [{0}] loss {1:.3f} Accuracy Top1 {2:.2f} Accuracy Top5 {3:.2f} FLOPs {4:.3f} Params {5:.2f}M".format(
            individual.get_Id(), valid_loss, valid_top1, valid_top5, n_flops, n_params))

        return {
            'FLOPs': n_flops,
            'accTop1': valid_top1,
            'accTop5': valid_top5,
            'params': (n_params, model.channels),
            'architecture': model.to_dot(),
            'fitness': np.array([valid_top1, n_flops]).reshape(-1)
        }

    def to_string(self, individual, results=None):
        file_str = ""
        file_str += "//[Network ID] {0:d}\n".format(individual.get_Id())
        file_str += "//[Encoding String] {0}\n".format(individual.to_string())
        file_str += "//[Parameter Size] {0}M with {1} channels\n".format(
            results['params'][0], results['params'][1])
        file_str += "//[Top 1 Accuracy rate] {0}\n".format(results['accTop1'])
        file_str += "//[Top 5 Accuracy rate] {0}\n".format(results['accTop5'])
        file_str += "//"+"="*50 + '\n'
        file_str += '\n'
        file_str += results['architecture']
        return file_str, 'dot' 