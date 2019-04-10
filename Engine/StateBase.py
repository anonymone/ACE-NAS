'''
License
'''
class action():
    def __init__(self):
        self.ADD_CONV = 1
        self.ADD_POOL = 2
        self.ADD_SKIP = 3
        self.ADD_BRANCH = 4
        self.END = 0

class stateBase:
    '''
    This is a basic class of state transition diagraph.
    It uses a dict type to define the state diagraph and the keys are
    the index of state node, values of each keys is a series of tuple 
    including two elements, the index of next state node and a callback function
    which will be called when enter the state node.

    Arguments:
        Graph : State transition diagraph
    
    Methods :
        next_state : recieving a state code and then going to corresponding state.
        re_set : reset the present state to the default state.
    '''

    def __init__(self):
        self.present_state = None
    
    def next_state(self, code=None):
        pass
        
    def re_set(self):
        pass