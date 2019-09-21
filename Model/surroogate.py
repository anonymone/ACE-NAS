import sys
# update your projecty root path before running
sys.path.insert(0, './')
import random
import torch
import itertools
import torch.utils.data as data
import torch.nn as nn
import numpy as np
import logging
import os

from Model import layers,NAOlayer
from Model import embeddingModel
from misc.flops_counter import add_flops_counting_methods
from misc import utils

LOG_FORMAT = '%(asctime)s%(name)s%(message)s'
logging.basicConfig(format=LOG_FORMAT, level=logging.DEBUG)

device = "cuda" if torch.cuda.is_available() else "cpu"


class RankNet(nn.Module):
    def __init__(self, sizeList=[(256, 128), (128, 64), (64, 32), (32, 1)]):
        super(RankNet, self).__init__()
        self.model = nn.Sequential()
        for layerNumber, (inSize, outSize) in enumerate(sizeList[:-1]):
            self.model.add_module('Linear{0}'.format(
                layerNumber), nn.Linear(inSize, outSize))
            self.model.add_module('ReLU{0}'.format(layerNumber), nn.ReLU())
        self.model.add_module('Linear{0}'.format(
            layerNumber+1), nn.Linear(sizeList[-1][0], sizeList[-1][1]))
        self.P_ij = nn.Sigmoid()

    def forward(self, input1, input2):
        x_i = self.model(input1)
        x_j = self.model(input2)
        S_ij = torch.add(x_i,-x_j)
        return self.P_ij(S_ij)

    def predict(self, inputs):
        outputs = self.model(inputs)
        return outputs


class RankNetDataset(data.Dataset):
    def __init__(self, dataNumpy=None, train=True,
                 transform=None, target_transferm=None, labelsLevel='INC'):  # LabelsLevel DEC label the larger the better, INC the smaller the better
        self.labelsLevel = labelsLevel
        self.transform = transform
        self.target_transform = target_transferm
        self.train = train
        if dataNumpy is None:
            if self.train:
                self.dataset = np.array([])
                self.train_data = np.array([])
                self.train_values = np.array([])
            else:
                self.dataset = np.array([])
                self.test_data = np.array([])
                self.test_data = np.array([])
        else:
            self.dataset = RankNetDataset.batchData(dataNumpy.shape[0])
            if self.train:
                self.train_data = dataNumpy[:, 1:-1].astype(dtype="float32")
                self.train_values = dataNumpy[:, -1].astype(dtype="float32")
            else:
                self.test_data = dataNumpy[:, 1:-1].astype(dtype="float32")
                self.test_values = dataNumpy[:, -1].astype(dtype="float32")

    @staticmethod
    def batchData(datasetSize, batchSize=32):
        index = [x for x in range(datasetSize)]
        pairs = [np.array([i, j]) for i, j in itertools.product(index, index)]
        # pairs = []
        # for i in range(datasetSize):
        #     for j in range(i, datasetSize, 1):
        #         pairs.append(np.array([i,j]))
        random.shuffle(pairs)
        # pairs = [pairs[i:i+batchSize] for i in range(0,datasetSize - datasetSize%batchSize, batchSize)]
        return np.array(pairs)

    def updateData(self, dataset):
        '''
        dataset is a numpy 2darray.
        '''
        self.dataset = RankNetDataset.batchData(dataset.shape[0])
        if self.train:
            self.train_data = dataset[:, :-1].astype(dtype="float32")
            self.train_values = dataset[:, -1].astype(dtype="float32")
        else:
            self.test_data = dataset[:, :-1].astype(dtype="float32")
            self.test_values = dataset[:, -1].astype(dtype="float32")

    def addData(self, newDataset):
        if self.dataset.__len__() == 0:
            self.updateData(newDataset)
        else:
            if self.train:
                self.train_data = np.vstack(
                    [self.train_data, newDataset[:, :-1].astype(dtype="float32")])
                self.train_values = np.hstack(
                    [self.train_values, newDataset[:, -1].astype(dtype="float32")])
                self.dataset = RankNetDataset.batchData(
                    self.train_data.shape[0])
            else:
                self.test_data = np.vstack(
                    [self.train_data, newDataset[:, :-1].astype(dtype="float32")])
                self.test_values = np.hstack(
                    [self.train_values, newDataset[:, -1].astype(dtype="float32")])
                self.dataset = RankNetDataset.batchData(
                    self.test_data.shape[0])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        assert self.dataset.__len__() > 0, "Dataset is empty."
        index_ij = self.dataset[index]
        target = np.zeros(1)
        if self.train:
            vector_i, value_i = self.train_data[index_ij[0]
                                                ], self.train_values[index_ij[0]]
            vector_j, value_j = self.train_data[index_ij[1]
                                                ], self.train_values[index_ij[1]]
        else:
            vector_i, value_i = self.test_data[index_ij[0]
                                               ], self.train_values[index_ij[0]]
            vector_j, value_j = self.test_data[index_ij[1]
                                               ], self.train_values[index_ij[1]]

        if self.labelsLevel == "DEC":
            target = np.ones(1) if value_i >= value_j else np.zeros(1)
        elif self.labelsLevel == "INC":
            target = np.ones(1) if value_i <= value_j else np.zeros(1)
        else:
            # Defult INC
            target = np.ones(1) if value_i <= value_j else np.zeros(1)

        if self.transform is not None:
            vector = self.transform(vector)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return (vector_i, vector_j), target.astype("float32")

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
        fmt_str += '{0}{1}\n'.format(
            tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(
            tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class Predictor:
    def __init__(self, encoder, modelSavePath, args, modelSize=[(256, 128), (128, 64), (64, 32), (32, 1)]):
        self.saveModelPath = modelSavePath
        self.modelSize = modelSize
        self.model = RankNet(modelSize)
        self.criterion = nn.BCELoss()
        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.Adam(parameters)
        self.encoder = encoder
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.args = args

    def predict(self, codeString):
        if type(codeString) is not list:
            codeString = [codeString]
        vector = self.encoder.encode(codeString)
        vector = torch.from_numpy(vector).to(self.device)
        self.model = self.model.to(self.device)
        with torch.no_grad():
            outputs = self.model.predict(vector)
        outputs = outputs.to('cpu')
        outputs = outputs.detach().numpy()
        return outputs.reshape(1)

    # Normal evaluation
    def evaluation(self,args ,individuals):
        result = []
        initChannel = self.args.trainSearch_initChannel
        CIFAR_CLASSES = self.args.trainSearchDatasetClassNumber

        for Id, ind in enumerate(individuals):
            # channels = [(3, initChannel)] + [((2**(i-1))*initChannel, (2**i)
            #                                   * initChannel) for i in range(1, len(ind.getDec()))]
            
            steps = int(np.ceil(40000 / 96)) * args.trainSearch_epoch

            model = NAOlayer.SEEArchitecture(args=args,
                                     classes=CIFAR_CLASSES,
                                     layers=2,
                                     channels=initChannel,
                                     code= ind.getDec(), 
                                     keepProb=args.trainSearch_keep_prob, 
                                     dropPathKeepProb=args.trainSearch_dropPathProb,
                                     useAuxHead=False, 
                                     steps=steps).to(device)
            # calculate for flopss1
            model = add_flops_counting_methods(model)
            model.eval()
            model.start_flops_count()
            # when the dataset changed it would be changed.
            random_data = torch.randn(1, 3, 32, 32).to(device)
            model(torch.autograd.Variable(random_data).to(device))
            n_flops = np.round(model.compute_average_flops_cost() / 1e6, 4)
            # calculate for predict value
            fitnessSG = self.predict(ind.toString(displayUsed=False))
            individuals[Id].setFitnessSG(fitnessSG)
            individuals[Id].setFitness([0., n_flops])
            result.append(np.hstack([[Id], fitnessSG, [n_flops]]))
        return np.array(result)

    def trian(self, dataset=None, trainEpoch=50, printFreqence=100, newModel=False):
        if newModel:
            self.model = RankNet(self.modelSize)
            parameters = filter(lambda p: p.requires_grad,
                                self.model.parameters())
            self.optimizer = torch.optim.Adam(parameters)
        if os.path.exists(os.path.join(self.saveModelPath, "model.ckpt")) and dataset is None:
            self.model.load_state_dict(torch.load(
                self.saveModelPath + "model.ckpt"))
            self.model.eval()
            logging.info("RankNet model find in {0}".format(
                self.saveModelPath+"model.ckpt"))
        else:
            if not os.path.exists(self.saveModelPath):
                os.makedirs(self.saveModelPath)
                os.chmod(self.saveModelPath, mode=0o777)
            logging.warning(
                "No pretrained model. Model will start trainning from scratch...")
            # device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = self.model.to(self.device)

            # Dataset Loader
            dataQueue = torch.utils.data.DataLoader(dataset=dataset,
                                                    batch_size=32,
                                                    shuffle=True)
            for epoch in range(trainEpoch):
                train_loss = 0
                correct = 0
                total = 0
                for step, (inputs, labels) in enumerate(dataQueue):

                    inputs_i, inputs_j, labels = inputs[0].to(
                        self.device), inputs[1].to(self.device), labels.to(self.device)
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs_i, inputs_j)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()

                    train_loss += loss.item()
                    outputs = outputs.cpu()
                    predicted = outputs.detach().numpy()
                    predicted[predicted >= 0.5] = 1
                    predicted[predicted < 0.5] = 0
                    total += labels.size(0)
                    correct += np.sum(predicted.reshape(1, -1)
                                      == labels.cpu().numpy().reshape(1, -1))
                    # if newCorrect > correct and epoch > trainEpoch*0.5:
                    #     torch.save(self.model.state_dict(), os.path.join(self.saveModelPath,"model_acc{0}.ckpt".format(newCorrect)))
                    #     logging.info("Save new Model (acc:{0}) successfully!".format(newCorrect))
                    # correct = newCorrect
                    if (step+1) % printFreqence == 0:
                        logging.info("Epoch: {0}, Step: {1} Loss: {2}, Acc: {3}".format(epoch,
                                                                                        step+1,
                                                                                        train_loss/total,
                                                                                        100.*correct/total))
            torch.save(self.model.state_dict(), os.path.join(
                self.saveModelPath, "model.ckpt"))


if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    import torch.utils as utils
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--EmbeddingTrainPath', action='store', dest='train_path', default='./Dataset/encodeData/data.txt',
                        help='Path to train data')
    parser.add_argument('--EmdeddingDevPath', action='store', dest='dev_path', default='./Dataset/encodeData/data_val.txt',
                        help='Path to dev data')
    parser.add_argument('--EmbeddingExptDir', action='store', dest='expt_dir', default='./Dataset/encodeData/',
                        help='Path to experiment directory. If load_checkpoint is True, then path to checkpoint directory has to be provided')
    parser.add_argument('--EmdeddingLoadCheckpoint', action='store', dest='load_checkpoint', default='2019_08_26_07_35_34',
                        help='The name of the checkpoint to load, usually an encoded time string')
    parser.add_argument('--EmdeddingResume', action='store_true', dest='resume',
                        default=False,
                        help='Indicates if training has to be resumed from the latest checkpoint')
    args = parser.parse_args()

    dataset = pd.read_csv('./Dataset/encodeData/surrogate.txt')
    #newdataset = np.ndarray(shape=dataset.shape)
    dataset = RankNetDataset(dataset.values)
    print(dataset.__len__())
    #dataset.addData(newdataset)
    #print(dataset.__len__())
    encoder = embeddingModel.EmbeddingModel(opt=args)

    predictor = Predictor(
        encoder=encoder, modelSavePath="./Dataset/encodeData/RankModel_test/",args=args)
    predictor.trian(dataset=dataset, trainEpoch=20)
    while True:
        code = input("Enter the input code: ")
        code = code.replace('Phase:',"").split('-')
        input_code = []
        for unit in code:
            substr = "-".join([bit for bit in unit])
            input_code.append(substr)
        code = " ".join(input_code)
        value = predictor.predict([code])
        print(value)
