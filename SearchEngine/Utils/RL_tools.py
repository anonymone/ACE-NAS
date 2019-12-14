import math
import numpy as np
from Coder.Network.utils import ACE_parser_tool as ACE_parser


class ACE_parser_tool_RL(object):
    @staticmethod
    def states_to_numpy(state_list:list):
        for i in range(len(state_list)):
            if state_list[i].action == -1:
                break
        normal_string = np.array(
            [np.array(s.to_list()).astype("int") for s in state_list[0:i]])
        reduct_string = np.array(
            [np.array(s.to_list()).astype("int") for s in state_list[i+1:-1]])
        return (normal_string, reduct_string)

    @staticmethod
    def string_to_states(encoding_str, state_format:'specify the state type'):
        return ACE_parser_tool_RL.numpy_to_states(ACE_parser.string_to_numpy(encoding_str), state_format)

    @staticmethod
    def numpy_to_states(state_numpy, state_format:'specify the state type'):
        state_list = [state_format(-2, 0, 0)]
        normal_encoding, reduct_encoding = state_numpy
        for action, p1, p2 in normal_encoding:
            state_list.append(state_format(action, p1, p2))
        state_list.append(state_format(-1, 0, 0))
        for action, p1, p2 in reduct_encoding:
            state_list.append(state_format(action, p1, p2))
        state_list.append(state_format(-1, 0, 0))
        return state_list

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

class State_Enumerator:
    def __init__(self, state_space):
        self.state_space = state_space

    def enumerate_state(self, state, q_value):
        pass

    def state_action_transition(self, start_s, action):
        pass