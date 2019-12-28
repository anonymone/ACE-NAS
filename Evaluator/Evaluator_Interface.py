import os
import time
import numpy as np
import logging

from Evaluator.Utils.recoder import create_exp_dir


class evaluator:
    def __init__(self, save_root='./Experiments/', mode='EXPERIMENT', **kwargs):
        create_exp_dir(save_root)
        self.save_root = save_root
        self.mode = mode
        create_exp_dir(save_root)
        for name in kwargs.keys():
            exec("self.{0} = kwargs['{0}']".format(str(name)))

    def set_mode(self, mode):
        '''
        Set the evaluate mode.
        '''
        self.mode = mode

    def evaluate(self, samples: 'an iterable collection of code class', **kwargs):
        total = len(samples)
        time_cost = 0
        bar_length = 30
        have_evaluated = 0
        for i,s in enumerate(samples):
            if s.is_evaluated():
                have_evaluated += 1
                continue
            used_t = time.time()
            results = self.eval_model(s)
            s.set_fitness(results['fitness'])
            self.save(s, results=results)
            time_cost += (time.time() - used_t)/60
            # process bar
            ave_time = time_cost/(i+1-have_evaluated)
            left_time = ave_time*(total-(i+1))
            print('[{0:>2d}/{1:>2d}]'.format(i+1, total)+'['+'*'*np.floor(bar_length*((i+1)/total)).astype('int') +
                  '-'*(bar_length-np.floor(bar_length*((i+1)/total)).astype('int'))+']'+ 
                  "{0:.2f} mins/ps, {1:.2f} mins left".format(ave_time, left_time))
        logging.info('\nTotal Evaluated {0:>2d} new samples in {1:.2f} mins'.format(total-have_evaluated, time_cost))

    def eval_model(self, individual, **kwargs):
        print("This method needs to be modified before using it.")

    def to_string(self, individual, results=None) -> 'string, file format':
        print("This method needs to be modified before using it.")
        return "This method needs to be modified before using it.", 'txt'

    def save(self, s, results):
        create_exp_dir(os.path.join(self.save_root, 'models'))
        string, suffix = self.to_string(individual=s, results=results)
        with open(os.path.join(self.save_root, 'models', '{0}.{1}'.format(str(s.get_Id()), suffix)), 'w') as file:
            file.write(string)
