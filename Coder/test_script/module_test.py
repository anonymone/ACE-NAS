import sys
sys.path.append('./')

import torch
import torch.nn as nn
import numpy as np

from Coder.ACE import ACE, ACE_Cell
from Coder.Network.nn import Network_CIFAR

ind = ACE(fitness_size=2,
             classes=1000,
             layers=6,
             channels=8)
print(ind.to_string())
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# normal_dec, reduction_dec = ind.get_dec()
# cell = ACE_Cell(
#     code=normal_dec,
#     prev_layers=[[32, 32, 3], [32, 32, 3]],
#     channels=32,
#     reduction=False,
#     layer_id=0,
#     init_layers=2,
#     steps=1000,
#     drop_path_keep_prob=1
# )
# inputs = torch.Tensor(np.random.rand(1, 3, 32, 32))
# cell, inputs = cell.to(device), inputs.to(device)
# y = cell(inputs, inputs, 1)
# final netowrk 
model = ind.get_model(steps=1, imagenet=False)
# model = Network_CIFAR(
#     cell_decoder=ACE_Cell,
#     code= ind.get_dec(),
#     classes =10,
#     layers=3,
#     channels=32,
#     keep_prob=0.8,
#     drop_path_keep_prob=0.8,
#     use_aux_head=True,
#     steps=1
# )
inputs = torch.Tensor(np.random.rand(32, 3, 256, 256))
model, inputs = model.to(device), inputs.to(device)
y,aux = model(inputs, 1)
print(y.shape)