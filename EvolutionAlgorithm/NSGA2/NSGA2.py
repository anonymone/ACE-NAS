from base import EAbase.EABase as EAbase
from Engine import StateControl

class NSGA2(EABase):
    def __init__(self, config):
        super(NSGA2,self).__init__()
        self.evaluate = sum
        self.decoder = StateControl()
        self.mutateRate = int(config['mutateRate'])
        self.crossOver = bool(config['crossOver'])

    def evaluate(self, population, decoder):
        for ind in population:
            model = self.decoder.get_model(ind.dec())
            