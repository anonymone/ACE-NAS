'''
License
'''
import threading
import multiprocessing


class action():
    '''
    Including all flags of specific action.
    '''
    def __init__(self):
        self.ADD_CONV = 1
        self.ADD_POOL = 2
        self.ADD_SKIP = 3
        self.ADD_BRANCH = 4
        self.ADD_LINEAR = 5
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

    def get_model(self,code):
        pass


class evalBase:
    def __init__(self, config):
        self.individuals = multiprocessing.Queue(maxsize=30)
        self.threadingNum = int(config['trainning setting']['threadingNum'])
        self.threadingMap = dict()
        # for number in range(self.threadingNum):
        #     self.threadingMap[str(number)] = threading.Thread(
        #         None, target=self.eval, name='Thread{0}'.format(number))
        self.evaluateTool = None

    def insert(self, ind):
        try:
            self.individuals.put(ind)
        except:
            return False
        return True

    def eval(self):
        while True:
            ind = self.individuals.get()
            if ind == 'None':
                break
            print('Get ind {0}'.format(ind.get_dec()))                
            fitness = self.evaluateTool(ind.get_dec())
            print("model fitness : {0} in {1}".format(fitness, threading.current_thread().getName()))
            # self.individuals.task_done()

    def stop(self):
        for i in range(self.threadingNum):
            self.individuals.put('None')
