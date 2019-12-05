import math

class State:
    def __init__(self, **kwargs):
        for name in kwargs.keys():
            exec("self.{0} = kwargs['{0}']".format(str(name)))
        self.parames_name = kwargs.keys() 
    
    def to_tuple(self):
        Print('This method needs to defined before use it.')
        return (None)
    
    def to_list(self):
        return list(self.to_tuple())
    
    def copy(self):
        exec("self.s = State({0})".format("".join([str(name) +"="+"self."+str(name)+"," for name in self.parames_name])))
        s = self.s
        del self.s
        return s