import sys
sys.path.append('./')
import numpy as np

from SearchEngine.RL_Engine import Q_State_Enumerator, Q_learning, Q_State
from SearchEngine.RL_Engine import RL_population
from SearchEngine.Utils.RL_tools import ACE_parser_tool as parser
from Coder.ACE import ACE

ind = ACE(1, (10,15), (0,15))
pop = RL_population(1, None)

q_engine = Q_learning(1, './Experiments/test_module/', Q_State_Enumerator((0,15)))
q_states = q_engine.generate_encode()
q_numpy = parser.states_to_numpy(q_states)
q_states_new = parser.numpy_to_states(q_numpy, Q_State)
q_engine.update_q_table(q_states_new, 0.98)
ind.set_dec(q_numpy)
pop.add_ind(ind,1,89.0,99.0,0.32,37.1)

model = ind.get_model(1)
