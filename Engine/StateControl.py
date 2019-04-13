'''
License
'''
import torch
import torch.nn as nn
import threading
import queue

from StateBase import *
from Models import layers


class decoder(stateBase):
    '''
    This is a derived class of stateBase
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

    def __init__(self, config):
        super(decoder, self).__init__()
        self.present_state = None
        self.INSTRUCT = action()
        self.actionSize = int(config['ActionCodeSize'])
        self.codeLength = int(config['codeLength'])
        self.parameterSize = self.codeLength - self.actionSize
        # Runtime Flags
        self.FLAG_SKIP = False
        self.FLAG_BRANCH = False
        self.previousOutSize = 3  # initialize to 3 channels RGB
        self.fullConnectLayerSize = 32

    def groupCode(self, code):
        code_cell = list()
        for i in range(int(code.shape[1]/self.codeLength)):
            cell = code[0, i:i+self.codeLength]
            cell = (cell[0:self.actionSize].copy(),
                    cell[self.actionSize:].copy())
            code_cell.append(cell)
        return code_cell

    def get_operator(self, actionCode):
        actionCode = actionCode % 3
        if actionCode[0] == self.INSTRUCT.ADD_CONV:
            return (layers.ConvolutionLayer, 1)
        if actionCode[0] == self.INSTRUCT.ADD_POOL:
            return (layers.PoolingLayer, 2)
        else:
            return (layers.ConvolutionLayer, 1)

    def get_parameters(self, parameters, opType):
        para_dict = {
            'in_channels': int(parameters[0]),  # hold
            'out_channels': int(parameters[1] * 30),
            'kernel_size': int(parameters[2] % 3 + 1),
            'stride': int(parameters[3] % 1 + 1),
            'padding': int(parameters[4] % 1),
            'active_function': int(parameters[5]),  # hold
            'poolingLayerType': int(parameters[6])
        }
        return self.check_parameters(para_dict, opType)

    def check_parameters(self, parameters_dict, opType):
        for param in parameters_dict:
            if param == 'in_channels':
                parameters_dict[param] = self.previousOutSize
                if opType == self.INSTRUCT.ADD_CONV:
                    self.previousOutSize = parameters_dict['out_channels']
            if param == 'active_function':
                if self.FLAG_SKIP:
                    pass
                parameters_dict[param] = nn.functional.relu
        if opType == self.INSTRUCT.ADD_CONV:
            self.fullConnectLayerSize = int((
                self.fullConnectLayerSize - parameters_dict['kernel_size'] + 2*parameters_dict['padding']) / parameters_dict['stride']) + 1
        elif opType == self.INSTRUCT.ADD_POOL:
            self.fullConnectLayerSize = int(
                (self.fullConnectLayerSize + 2*parameters_dict['padding'] - 1*(parameters_dict['kernel_size']-1)-1)/parameters_dict['stride'] + 1
                )
        else:
            pass
        return parameters_dict

    def get_model(self, code):
        code = self.groupCode(code)
        model = list()
        for (actionCode, parameters) in code:
            print(actionCode, parameters)
            (operator, opType) = self.get_operator(actionCode)
            parameters = self.get_parameters(parameters, opType)
            model.append(operator(parameters))
        return nn.Sequential(*model)
