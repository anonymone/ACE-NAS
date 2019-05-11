'''
License
'''
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import math
import threading
import queue
import numpy as np
import logging
FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(filename='./logs/train.log',
                    level=logging.DEBUG, format=FORMAT)


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
        '''
        parameters : 

        '''
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
        actionCode = actionCode % 5
        if actionCode[0] == self.INSTRUCT.ADD_CONV:
            return (layers.ConvolutionLayer, self.INSTRUCT.ADD_CONV)
        elif actionCode[0] == self.INSTRUCT.ADD_POOL:
            return (layers.PoolingLayer, self.INSTRUCT.ADD_POOL)
        elif actionCode[0] == self.INSTRUCT.ADD_SKIP:
            return (layers.SkipContainer, self.INSTRUCT.ADD_SKIP)
        elif actionCode[0] == self.INSTRUCT.ADD_BRANCH:
            return (layers.MultiBranchsContainer, self.INSTRUCT.ADD_BRANCH)
        else:
            return (layers.ConvolutionLayer, self.INSTRUCT.ADD_CONV)

    def get_parameters(self, parameters, opType):
        if opType == self.INSTRUCT.ADD_LINEAR:
            para_dict = {
                'layer_size': (int(parameters[1]) % 4 + 1),  # range(1,4)
                'out_size0': (int(parameters[2]) % 9 + 1) * 400,
                'out_size1': (int(parameters[3]) % 9 + 1) * 300,
                'out_size2': (int(parameters[4]) % 9 + 1) * 200,
                'out_size3': (int(parameters[5]) % 9 + 1) * 100,
                'out_size4': (int(parameters[6]) % 9 + 1) * 100
            }
        elif opType == self.INSTRUCT.ADD_SKIP:
            kernelSize = [1, 3, 5]
            para_dict = {
                'layer_size': (int(parameters[1]) % 4 + 1),  # range(1,4)
                'kernel_size0': kernelSize[int(parameters[2] % 3)],
                'kernel_size1': kernelSize[int(parameters[3] % 3)],
                'kernel_size2': kernelSize[int(parameters[4] % 3)],
                'kernel_size3': kernelSize[int(parameters[5] % 3)],
                'kernel_size4': kernelSize[int(parameters[6] % 3)]
            }
        elif opType == self.INSTRUCT.ADD_BRANCH:
            para_dict = {
                'branch_size': (int(parameters[1]) % 3 + 2),  # range(2,4)
                'type0': int(parameters[2] % 4),
                'type1': int(parameters[3] % 4),
                'type2': int(parameters[4] % 4),
                'type3': int(parameters[5] % 4),
                'type4': int(parameters[6] % 4)
            }
        else:
            para_dict = {
                'in_channels': int(parameters[0]),  # hold
                'out_channels': int(parameters[1] * 30),
                'kernel_size': int(parameters[2] % 3 + 2),
                'stride': int(parameters[3] % 2 + 1),
                'padding': int(parameters[4] % 2),
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
                parameters_dict[param] = nn.ReLU(inplace=True)
        if opType == self.INSTRUCT.ADD_CONV:
            self.fullConnectLayerSize = int((
                self.fullConnectLayerSize - parameters_dict['kernel_size'] + 2*parameters_dict['padding']) / parameters_dict['stride']) + 1

        elif opType == self.INSTRUCT.ADD_POOL:
            self.fullConnectLayerSize = int(
                (self.fullConnectLayerSize + 2*parameters_dict['padding'] - 1*(
                    parameters_dict['kernel_size']-1)-1)/parameters_dict['stride'] + 1
            )
        # In linear type, it returns a list not DICTIONARY
        elif opType == self.INSTRUCT.ADD_LINEAR:
            param_dict = list()
            for x in range(parameters_dict['layer_size']):
                param_dict.append(parameters_dict['out_size'+str(x)])
            # 10 is 10 classes in cafir10
            param_dict.append(10)
            parameters_dict = param_dict
        # In Skip type, it return a list not DICTIONARY
        elif opType == self.INSTRUCT.ADD_SKIP:
            param_dict = list()
            for x in range(parameters_dict['layer_size']):
                param_dict.append(layers.ConvolutionLayer(parameters={
                    'in_channels': self.previousOutSize,  # hold
                    'out_channels': self.previousOutSize,
                    'kernel_size': parameters_dict['kernel_size'+str(x)],
                    'stride': 1,
                    'padding': int((parameters_dict['kernel_size'+str(x)] - 1)/2),
                    'active_function': nn.ReLU(inplace=True)  # hold
                }))
            parameters_dict = param_dict
        elif opType == self.INSTRUCT.ADD_BRANCH:
            param_dict = list()
            modules = {
                'branch0': layers.MultiBase1x1,
                'branch1': layers.MultiBase1x1_3x3,
                'branch2': layers.MultiBase1x1_5x5,
                'branch3': layers.MultiBasePool1x1_5x5
            }
            count = 0
            for x in range(parameters_dict['branch_size']):
                y = parameters_dict['type'+str(x)]
                model = modules['branch'+str(y)](inSize=self.previousOutSize)
                param_dict.append([model])
                count = count + model.outChannelSize
            parameters_dict = param_dict
            self.previousOutSize = count
            # We add this because of the limitation of our DL server is not fierce.
            self.fullConnectLayerSize = int(
                (self.fullConnectLayerSize - 1*(2-1)-1)/1 + 1)
        else:
            pass
        return parameters_dict

    def get_model(self, code):
        code = self.groupCode(code)
        model = list()
        for (actionCode, parameters) in code[:-1]:
            # print(actionCode, parameters)
            (operator, opType) = self.get_operator(actionCode)
            parameters = self.get_parameters(parameters, opType)
            model.append(operator(parameters))
        # init the classifier
        parameters = self.get_parameters(code[-1][1], self.INSTRUCT.ADD_LINEAR)
        model.append(layers.linear(self.fullConnectLayerSize,
                                   self.previousOutSize, parameters))
        return nn.Sequential(*model)


class evaluator(evalBase):
    def __init__(self, config):
        super(evaluator, self).__init__(config)
        self.config = config
        self.evaluateTool = self.train
        self.batchSize = int(config['trainning setting']['batchSize'])
        self.numberWorkers = int(config['trainning setting']['numberWorkers'])
        self.dataPath = config['trainning setting']['dataPath']
        self.lr = float(config['trainning setting']['learningRate'])
        self.epoch = int(config['trainning setting']['trainEpoch'])

        for number in range(self.threadingNum):
            self.threadingMap[str(number)] = threading.Thread(
                None, target=self.eval, name='Thread{0}'.format(number))

    def train(self, dec, Mode=None):
        # Debugs
        if Mode == 'DEBUG':
            return np.random.randint(1, 100, size=(1, 2))
        Decode = decoder(self.config)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        try:
            model = Decode.get_model(dec)
            model.to(device)
            print(model)
        except:
            logging.info("Model is invalid. {0} ".format(dec))
            return np.array([[np.inf, np.inf]]), np.inf
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=0.9)
        # train
        for epoch in range(self.epoch):
            for i, data in enumerate(self.trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                # forward
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        correct = 0
        total = 0
        # test accuracy
        with torch.no_grad():
            for i, data in enumerate(self.testloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        train_correct = 0
        train_total = 0
        # train accuracy
        with torch.no_grad():
            for i, data in enumerate(self.trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
        # cpmputational complexity
        computComplexity = self.getModelComplexity(model)
        torch.cuda.empty_cache()
        return np.array([[1-(correct/total), computComplexity*0.00000001]]), 1-(train_correct/train_total)

    def getModelComplexity(self, model):
        count = 0
        for param in model.parameters():
            paramSize = param.size()
            countEach = 1
            for dis in paramSize:
                countEach *= dis
            count += countEach
        return count

    def initEngine(self, path=None, threadingAble=False):
        if path is not None:
            self.dataPath = path
            # load dataset
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.CIFAR10(root=self.dataPath, train=True,
                                                download=True, transform=self.transform)
        self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batchSize,
                                                       shuffle=True, num_workers=self.numberWorkers)
        testset = torchvision.datasets.CIFAR10(root=self.dataPath, train=False,
                                               download=True, transform=self.transform)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=self.batchSize,
                                                      shuffle=False, num_workers=self.numberWorkers)

        # start evaluator threading
        if threadingAble:
            try:
                for number in self.threadingMap:
                    self.threadingMap[number].start()
            except:
                print("Threading {0} starts failed.".format(number))
