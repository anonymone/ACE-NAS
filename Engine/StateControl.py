'''
License
'''
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
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
        self.actionSize = int(config['individual setting']['ActionCodeSize'])
        self.codeLength = int(config['individual setting']['codeLength'])
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
                (self.fullConnectLayerSize + 2*parameters_dict['padding'] - 1*(
                    parameters_dict['kernel_size']-1)-1)/parameters_dict['stride'] + 1
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
        model.append(layers.linear(self.fullConnectLayerSize, self.previousOutSize))
        return nn.Sequential(*model)


class evaluator(evalBase):
    def __init__(self, config):
        super(evaluator, self).__init__(config)
        self.evaluateTool = self.train
        self.decoder = decoder(config)
        self.batchSize = int(config['trainning setting']['batchSize'])
        self.numberWorkers = int(config['trainning setting']['numberWorkers'])
        self.dataPath = config['trainning setting']['dataPath']
        self.lr = float(config['trainning setting']['learningRate'])
        self.epoch = int(config['trainning setting']['trainEpoch'])

    def train(self, dec):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = self.decoder.get_model(dec)
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=0.9)

        for epoch in range(self.epoch):
            for i,data in enumerate(self.trainloader,0):
                inputs, labels = data
                inputs, labels =  inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                # forward 
                outputs = model(inputs)
                loss = criterion(outputs,labels)
                loss.backward()
                optimizer.step()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.testloader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct/total
        
        

    def initEngine(self, path = None):
        if path is None:
            path = self.dataPath
        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.CIFAR10(root=path, train=True,
                                        download=True, transform=self.transforms)
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batchSize,
                                          shuffle=True, num_workers=self.numberWorkers)
        testset = torchvision.datasets.CIFAR10(root=path, train=False,
                                       download=True, transform=self.transforms)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=self.batchSize,
                                         shuffle=False, num_workers=self.numberWorkers)
        # start evaluator threading
        try:
            for number in self.threadingMap:
                self.threadingMap[number].start()
        except:
            print("Threading {0} starts failed.".format{number}) 