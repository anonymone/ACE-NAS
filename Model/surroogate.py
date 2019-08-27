import sys
# update your projecty root path before running
sys.path.insert(0, './')

import numpy as np
import torch 
import torch.nn as nn
import torch.utils.data as data
import itertools
import random 

from misc import utils
from Model import embeddingModel


class RankNet(nn.Module):
    def __init__(self, sizeList=[(256,128),(128,64),(64,32),(32,1)]):
        super(RankNet,self).__init__()
        self.model = nn.Sequential()
        for inSize, outSize in sizeList[:-1]:
            self.model.add_module('Linear{0}_{1}'.format(inSize,outSize),nn.Linear(inSize,outSize))
            self.model.add_module('ReLU',nn.ReLU())
        self.model.add_module('Linear{0}_{1}'.format(sizeList[-1][0],sizeList[-1][1]),nn.Linear(sizeList[-1][0],sizeList[-1][1]))
        self.P_ij = nn.Sigmoid()
    def forward(self, input1, input2):
        x_i = self.model(input1)
        x_j = self.model(input2)
        S_ij = x_i - x_j
        return self.P_ij(S_ij)
    
    def predict(self, inputs):
        return self.model(inputs)

class RankNetDataset(data.Dataset):
    def __init__(self, dataNumpy, train=True,
                 transform=None, target_transferm=None):
        self.transform = transform
        self.target_transform = target_transferm
        self.train = train
        self.dataset = RankNetDataset.batchData(dataNumpy.shape[0])
        if self.train:
            self.train_data = dataNumpy[:,1:-1].astype(dtype="float32")
            self.train_values = dataNumpy[:,-1].astype(dtype="float32")
        else:
            self.test_data = dataNumpy[:,1:-1].astype(dtype="float32")
            self.test_values = dataNumpy[:,-1].astype(dtype="float32")
    
    @staticmethod
    def batchData(datasetSize, batchSize = 32):
        index = [x for x in range(datasetSize)]
        pairs = [np.array([i,j]) for i,j in itertools.product(index,index)]
        random.shuffle(pairs)
        # pairs = [pairs[i:i+batchSize] for i in range(0,datasetSize - datasetSize%batchSize, batchSize)]
        return np.array(pairs)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        index_ij = self.dataset[index]
        target = np.zeros(1)
        if self.train:
            vector_i, value_i = self.train_data[index_ij[0]], self.train_values[index_ij[0]]
            vector_j, value_j = self.train_data[index_ij[1]], self.train_values[index_ij[1]]            
        else:
            vector_i, value_i = self.test_data[index_ij[0]], self.train_values[index_ij[0]]
            vector_j, value_j = self.test_data[index_ij[1]], self.train_values[index_ij[1]]
        
        target = np.ones(1) if value_i >= value_j else np.zeros(1)
        
        if self.transform is not None:
            vector = self.transform(vector)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return (vector_i,vector_j), target.astype("float32")

    def __len__(self):
        if self.train:
            return len(self.dataset)
        else:
            return len(self.dataset)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        # fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

class Predictor:
    def __init__(self, encoder, modelSize = [(256,128),(128,64),(64,32),(32,1)]):
        self.model = RankNet(modelSize)
        self.criterion = nn.BCELoss()
        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.Adam(parameters)
        self.encoder = encoder
        
    def trian(self, dataset, trainEpoch = 50, printFreqence=1000):
        device = "cuda" if torch.cuda.is_available() else "cpu" 
        self.model = self.model.to(device)

        # Dataset Loader
        dataQueue = torch.utils.data.DataLoader(dataset=dataset, 
                                                batch_size=32, 
                                                shuffle=True)
        for epoch in range(trainEpoch):
            train_loss = 0
            correct = 0
            total = 0
            for step, (inputs, labels) in enumerate(dataQueue):

                inputs_i,inputs_j, labels = inputs[0].to(device), inputs[1].to(device), labels.to(device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs_i, inputs_j)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                outputs = outputs.cpu()
                predicted = outputs.detach().numpy()
                predicted[predicted>=0.5] = 1
                predicted[predicted<0.5] = 0
                total += labels.size(0)
                correct += np.sum(predicted.reshape(1,-1)==labels.cpu().numpy().reshape(1,-1))
                if (step+1)%printFreqence == 0:
                    print("Epoch: {0}, Step: {1} Loss: {2}, Acc: {3}".format(epoch, 
                                                                            step, 
                                                                            train_loss/total, 
                                                                            100.*correct/total))

    def predict(self):
        pass
        
if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    import torch.utils as utils

    dataset = pd.read_csv('./Dataset/encodeData/surrogate.txt')
    dataset = RankNetDataset(dataset.values)

    predictor = Predictor(None)
    predictor.trian(dataset=dataset)    
