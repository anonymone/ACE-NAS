'''
The RL_Engine references the following code. Thanks for every contributors related to.
https://github.com/SAGNIKMJR/MetaQNN_ImageClassification_PyTorch
'''

import pandas as pd
import numpy as np

from SearchEngine.Engine_Interface import population
from Coder.ACE import ACTION
from SearchEngine.Utils.RL_tools import State, State_Enumerator


class RL_population(population):
    def __init__(self,
                 obj_number,
                 ind_params,
                 ind_generator=None,
                 pop_size=0):
        super(RL_population, self).__init__(
            obj_number=obj_number,
            pop_size=pop_size,
            ind_generator=ind_generator,
            ind_params=ind_params)

        self.existed_model = pd.DataFrame(columns=['Encoding string',
                                                   'Valid accuracy Top1',
                                                   'Valid accuracy Top5',
                                                   'Parameters size',
                                                   'FLOPs',
                                                   'epsilon'])

    def is_exist(self, ind):
        ind_string = ind.to_string()
        if ind_string in self.existed_model['Encoding string'].values:
            valid_top1 = self.existed_model[self.existed_model['Encoding string']
                                            == ind_string]['Valid accuracy Top1'].values[0]
            valid_top5 = self.existed_model[self.existed_model['Encoding string']
                                            == ind_string]['Valid accuracy Top5'].values[0]
            n_params = self.existed_model[self.existed_model['Encoding string']
                                          == ind_string]['Parameters size'].values[0]
            n_flops = self.existed_model[self.existed_model['Encoding string']
                                         == ind_string]['FLOPs'].values[0]
            epsilon = self.existed_model[self.existed_model['Encoding string']
                                         == ind_string]['epsilon'].values[0]
            model = ind.get_model(1)
            return {
                'FLOPs': n_flops,
                'accTop1': valid_top1,
                'accTop5': valid_top5,
                'params': (n_params, model.channels),
                'architecture': model.to_dot(),
                'fitness': np.array([valid_top1, n_flops]).reshape(-1)
            }
        else:
            return None

    def add_ind(self, ind, epsilon, valid_1, valid_5, param_size, flops):
        super(RL_population, self).add_ind(ind)
        self.existed_model = self.existed_model.append(pd.DataFrame([[ind.to_string(), valid_1, valid_5, param_size, flops, epsilon]],
                                                                    columns=['Encoding string',
                                                                             'Valid accuracy Top1',
                                                                             'Valid accuracy Top5',
                                                                             'Parameters size',
                                                                             'FLOPs',
                                                                             'epsilon']), ignore_index=True)

    def to_matrix(self):
        pass


# define the start and end state.
Q_ACTION = ACTION.copy()
Q_ACTION[-2] = 'start'
Q_ACTION[-1] = 'end'


class Q_State(State):
    def __init__(self,
                 action=None,
                 param0=None,
                 param1=None,
                 param_list=None):
        super(Q_State, self).__init__(action=action,
                                      param0=param0,
                                      param1=param1)
        if param_list is not None:
            self.action = param_list[0]
            self.param0 = param_list[1]
            self.param1 = param_list[2]

    def to_tuple(self):
        return (self.action,
                self.param0,
                self.param1)

    def copy(self):
        return Q_State(self.action, self.param0, self.param1)


class Q_State_Enumerator(State_Enumerator):
    def __init__(self, param_boundary, state_space=Q_ACTION):
        super(Q_State_Enumerator, self).__init__(state_space=state_space)
        self.param_boundary = param_boundary
        self.param_conbination = list()
        self.init_q_value = 0.5
        for index in range(-1, self.param_boundary[1]):
            for i in range(self.param_boundary[0], self.param_boundary[1]):
                for j in range(self.param_boundary[0], self.param_boundary[1]):
                    if (i, j) not in self.param_conbination:
                        self.param_conbination.append((index, i, j))

    def enumerate_state(self, state, q_values):
        actions = list()
        for action, p1, p2 in self.param_conbination:
            actions += [Q_State(action, p1, p2)]
        q_values[state.to_tuple()] = {'action': [s.to_tuple() for s in actions],
                                      'q_value': [self.init_q_value for i in range(len(actions))]}
        return q_values

    def state_action_transition(self, start_s, action):
        to_state = action.copy()
        return to_state


class Q_table:
    def __init__(self):
        '''
        {state:{'action':...,
                'q_value':...}}
        '''
        self.q = dict()

    def save(self, save_path, filename, file_format):
        pass

    def load(self, path):
        pass


class Q_learning:
    def __init__(self,
                 epsilon,
                 save_path,
                 state_format: 'class used to generate the state',
                 q_lr: 'q learning rate' = 0.1,
                 q_discount_factor=1.0,
                 #  state_space_params=Q_ACTION,
                 q_table=None):
        self.epsilon = epsilon
        self.q_learning_rate = q_lr
        self.q_discount_factor = q_discount_factor
        self.state_list = list()
        # self.state_space_params = state_space_params
        self.state_enum = state_format
        if q_table is None:
            self.q_table = Q_table()
        else:
            self.q_table = q_table
        self.save_path = save_path
        self.state = Q_State(-2, 0, 0)
        self.state_change = False

    def generate_encode(self, epsilon=None):
        if epsilon is not None:
            self.epsilon = epsilon
        self.__reset_state()
        while not (self.state.action == -1 and self.state_change):
            if self.state.action == -1 and not self.state_change:
                self.state_change = True
            self.select_action()
        return self.state_list

    def __reset_state(self):
        self.state_list = []
        self.state = Q_State(-2, 0, 0)
        self.state_change = False

    def select_action(self):
        if self.state.to_tuple() not in self.q_table.q:
            self.state_enum.enumerate_state(self.state, self.q_table.q)
        action_values = self.q_table.q[self.state.to_tuple()]
        if np.random.random() < self.epsilon:
            action = Q_State(param_list=action_values['action'][np.random.randint(
                len(action_values['action']))])
        else:
            max_q_value = max(action_values['q_value'])
            max_q_indexes = [i for i in range(
                len(action_values['action'])) if action_values['q_value'][i] == max_q_value]
            max_actions = [action_values['action'][i] for i in max_q_indexes]
            action = Q_State(
                state_list=max_actions[np.random.randint(len(max_actions))])
        while (len(self.state_list) == 0 and action.action == -1) or (len(self.state_list) > 0 and self.state.action == -1 and action.action == -1):
            if np.random.random() < self.epsilon:
                action = Q_State(param_list=action_values['action'][np.random.randint(
                    len(action_values['action']))])
            else:
                max_q_value = max(action_values['q_value'])
                max_q_indexes = [i for i in range(
                    len(action_values['action'])) if action_values['q_value'][i] == max_q_value]
                max_actions = [action_values['action'][i]
                               for i in max_q_indexes]
                action = Q_State(
                    state_list=max_actions[np.random.randint(len(max_actions))])
        self.state_action_transition(self.state, action)

    def state_action_transition(self, state, action) -> 'update self.state and inset state into state list.':
        self.state = self.state_enum.state_action_transition(
            start_s=state, action=action)
        self.state_list.append(self.state.copy())

    def update_q_table(self, state_seqence, reward):
        self.state_change = True
        self.__update_q_value(state_seqence[-2], state_seqence[-1], reward)
        self.state_change = False
        for i in reversed(range(len(state_seqence)-2)):
            self.__update_q_value(state_seqence[i], state_seqence[i+1], 0)

    def __update_q_value(self, start_state, to_state, reward):
        if start_state.to_tuple() not in self.q_table.q:
            self.state_enum.enumerate_state(start_state, self.q_table.q)
        if to_state.to_tuple() not in self.q_table.q:
            self.state_enum.enumerate_state(to_state, self.q_table.q)

        actions = self.q_table.q[start_state.to_tuple()]['action']
        values = self.q_table.q[start_state.to_tuple()]['q_value']

        max_over_next_states = max(
            self.q_table.q[to_state.to_tuple()]['q_value']) if not (to_state.action == -1 and self.state_change) else 0
        action_between_states = self.state_enum.state_action_transition(
            start_state, to_state).to_tuple()

        values[actions.index(action_between_states)] = values[actions.index(action_between_states)] + \
            self.q_learning_rate * \
            (reward + self.q_discount_factor *
             max_over_next_states -
             values[actions.index(action_between_states)])
