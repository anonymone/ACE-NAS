import pandas as pd

from SearchEngine.Engine_Interface import population


class RL_population(population):
    def __init__(slef,
                 obj_number,
                 ind_generator=None,
                 ind_params,
                 pop_size=0)
    super(RL_population, self).__init__(
        obj_number=obj_number,
        pop_size=pop_size,
        ind_generator=ind_generator,
        ind_params=ind_params)

    self.existed_model = pd.DataFrame(columns=['Encoding string',
                                               'Valid accuracy Top1',
                                               'Valid accuracy Top5',
                                               'Parameters size',
                                               'epsilon'])

    def is_exist(self):
        pass

    def to_matrix(self):
        pass

class Q_table:
    def __init__(self):
        self.q = dict{}
    
    def save(self, save_path, filename, file_format):
        pass
    
    def load(self, path):
        pass

class Q_learning:
    def __init__(self,
                state_space_params,
                epsilon,
                save_path,
                q_table,
                state_format:'class used to generate the state',
                model_gallery:RL_population=None):
        self.state_list = list()
        self.state_space_params = state_space_params
        self.state_format = state_format
        self.q_table = Q_table()
        self.epsilon = epsilon
        self.save_path = save_path
    
    def generate_encode(self):
        pass

    def reset_state(self):
        pass

    def select_action(self):
        pass
        
    def state_action_transition(self, state, action) -> 'update self.state and inset state into state list.':
        pass

    def update_q_table(self, state_seqence, reward):
        pass

    def __update_q_value(self, start_state, to_state, reward):
        pass

    
        