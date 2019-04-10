from base import EAbase.EABase as EAbase
#from Engine import *

class NSGA2(EABase):
    def __init__(self):
        super(NSGA2,self).__init__()
        self.evaluate = sum
        