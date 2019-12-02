import os
import numpy as np
from Evaluator.Utils.recoder import create_exp_dir


class evaluator:
    def __init__(self, save_root='./Experiments/', **kwargs):
        self.save_root = save_root
        create_exp_dir(save_root)
        for name in kwargs.keys():
            exec("self.{0} = kwargs['{0}']".format(str(name)))

    def evaluate(self, samples: 'an iterable collection of code class'):
        total = len(samples)
        bar_length = 30
        for i,s in enumerate(samples):
            print('[{0}/{1}]'.format(i, total)+'*'*np.ceil(bar_length*(i/total)).astype('int') +
                  '-'*np.ceil(bar_length*(1-i/total)).astype('int'))
            results = self.eval_model(s)
            s.set_fitness(results['fitness'])
            self.save(s, results=results)

    def eval_model(self, individual):
        print("This method needs to be modified before using it.")

    def to_string(self, individual, results=None) -> 'string, file format':
        print("This method needs to be modified before using it.")
        return "This method needs to be modified before using it.", 'txt'

    def save(self, s, results):
        create_exp_dir(os.path.join(self.save_root, 'models'))
        string, suffix = self.to_string(individual=s, results=results)
        with open(os.path.join(self.save_root, 'models', '{0}.{1}'.format(str(s.get_Id()), suffix)), 'w') as file:
            file.write(string)
