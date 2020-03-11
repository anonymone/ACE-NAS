import random
import torch
import itertools
import torch.utils.data as data
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import torchtext
import numpy as np
import logging
import os

import seq2seq
from seq2seq.trainer import SupervisedTrainer
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq
from seq2seq.loss import Perplexity
from seq2seq.optim import Optimizer
from seq2seq.dataset import SourceField, TargetField
from seq2seq.evaluator import Predictor
from seq2seq.util.checkpoint import Checkpoint

from Evaluator.Utils.recoder import count_parameters, create_exp_dir
from Evaluator.Utils.train import train, valid


class EmbeddingModel:
    def __init__(self, device='cuda', model_path='./PretrainModel/Seq2Rank/', model_file='seq2rank.ckpt'):
        try:
            checkpoint = Checkpoint.load(os.path.join(model_path, model_file))
        except:
            logging.error('[ERROR] [Seq2Rank] Pretrain Encode model load failed on {0}'.format(
                os.path.join(model_path, model_file)))
        self.seq2seq = checkpoint.model
        self.input_vocab = checkpoint.input_vocab
        self.output_vocab = checkpoint.output_vocab
        self.device = device

    def encode(self, code):
        '''
        Inputs: a string array, each row is an input sequence.
        Outputs: a numpy array, each row is an output vector.
        '''
        vectors = []
        for seqStr in code:
            seq = seqStr.strip().split()
            src_id_seq = torch.LongTensor(
                [self.input_vocab.stoi[tok] for tok in seq]).view(1, -1)
            src_id_seq = src_id_seq.cuda()
            with torch.no_grad():
                output, hiden = self.seq2seq.encoder(src_id_seq, [len(seq)])
            hiden_cpu = hiden.cpu()
            vectors.append(hiden_cpu.data.numpy().reshape(-1))
        return np.array(vectors)

    def encode2numpy(self, code, with_fitness=True):
        if with_fitness:
            decList = []
            values = []
            for decString, value in code:
                decList.append(decString)
                values.append(value)
            return np.hstack([self.encode(decList), np.array(values).reshape(-1,1)])
        else:
            decList = []
            for decString in code:
                decList.append(decString)
            return np.array(decList)


class RankNet(nn.Module):
    def __init__(self, sizeList=[(128, 64), (64, 32), (32, 1)]):
        super(RankNet, self).__init__()
        self.model = nn.Sequential()
        for layerNumber, (inSize, outSize) in enumerate(sizeList[:-1]):
            self.model.add_module('Linear{0}'.format(
                layerNumber), nn.Linear(inSize, outSize))
            self.model.add_module('ReLU{0}'.format(layerNumber), nn.ReLU())
        self.model.add_module('Linear{0}'.format(
            layerNumber+1), nn.Linear(sizeList[-1][0], sizeList[-1][1]))
        self.P_ij = nn.Sigmoid()

    def forward(self, input, global_step=None):
        input1, input2 = input[:,0,:], input[:,1,:]
        x_i = self.model(input1)
        x_j = self.model(input2)
        S_ij = torch.add(x_i, -x_j)
        return self.P_ij(S_ij), None

    def predict(self, inputs):
        outputs = self.model(inputs)
        return outputs


class RankNetDataset(data.Dataset):
    def __init__(self, data_numpy=None, train=True,
                 transform=None, target_transferm=None, labelsLevel='INC'):  # LabelsLevel DEC label the larger the better, INC the smaller the better
        self.labelsLevel = labelsLevel
        self.transform = transform
        self.target_transform = target_transferm
        self.train = train
        if data_numpy is None:
            if self.train:
                self.dataset = np.array([])
                self.train_data = np.array([])
                self.train_values = np.array([])
            else:
                self.dataset = np.array([])
                self.test_data = np.array([])
                self.test_data = np.array([])
        else:
            self.dataset = RankNetDataset.batchData(data_numpy.shape[0])
            if self.train:
                self.train_data = data_numpy[:, 1:-1].astype(dtype="float32")
                self.train_values = data_numpy[:, -1].astype(dtype="float32")
            else:
                self.test_data = data_numpy[:, 1:-1].astype(dtype="float32")
                self.test_values = data_numpy[:, -1].astype(dtype="float32")

    @staticmethod
    def batchData(datasetSize, batchSize=32):
        index = [x for x in range(datasetSize)]
        pairs = [np.array([i, j]) for i, j in itertools.product(index, index)]
        random.shuffle(pairs)
        return np.array(pairs)

    def update_data(self, dataset):
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

    def add_data(self, newDataset):
        if self.dataset.__len__() == 0:
            self.update_data(newDataset)
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
        return np.vstack([vector_i, vector_j]), target.astype("float32")

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


class Seq2Rank:
    def __init__(self,
                 encoder,
                 model_save_path,
                 input_preprocess: 'used to format the encoding string' = lambda x: x,
                 model_size=[(64, 64), (64, 32), (32, 1)],
                 device='cpu'):
        self.save_model_path = model_save_path
        self.model_size = model_size
        self.model = RankNet(model_size)
        self.criterion = nn.BCELoss()
        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.Adam(parameters)
        self.encoder = encoder
        self.device = device
        self.input_preprocess = input_preprocess
        self.data_set = RankNetDataset()

    def update_dataset(self, data : '[(input string, label)]'):
        self.data_set.add_data(self.encoder.encode2numpy(data))

    def predict(self, codeString):
        if type(codeString) is not list:
            codeString = [codeString]
        vector = self.encoder.encode(codeString)
        vector = torch.from_numpy(vector).cuda()
        self.model = self.model.cuda()
        with torch.no_grad():
            outputs = self.model.predict(vector)
        outputs = outputs.to('cpu')
        outputs = outputs.detach().numpy()
        return outputs.reshape(1)

    # Normal evaluation
    def evaluation(self, individuals):
        result = []
        for Id, ind in enumerate(individuals):
            # calculate for predict value
            fitnessSG = self.predict(ind.to_string(
                callback=self.input_preprocess))
            individuals[Id].set_fitnessSG(fitnessSG)
        
    def eval(self, dataset=None, batch_size=32, num_workers=0):
        if dataset is None:
            dataset = self.data_set
        data_queue = torch.utils.data.DataLoader(dataset=dataset,
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 num_workers=num_workers)
        
        def rate_fun(outputs, labels, topk=(1, 1)):
            outputs = outputs.cpu()
            predicted = outputs.detach().numpy()
            predicted[predicted >= 0.5] = 1
            predicted[predicted < 0.5] = 0
            total = labels.size(0)
            correct = np.sum(predicted.reshape(1, -1)
                             == labels.cpu().numpy().reshape(1, -1))
            return torch.from_numpy(np.array([correct/total])), torch.from_numpy(np.array([correct/total]))
        valid_loss, valid_top1, valid_top5  = valid(data_queue,
                                                    self.model, 
                                                    self.criterion,
                                                    device='cuda', rate_static=rate_fun)
        logging.info("[Valid] [Train] loss {0:.3f} error Top1 {1:.2f} error Top5 {2:.2f}".format(
                valid_loss, valid_top1, valid_top5))
        return valid_loss, valid_top1, valid_top5

    def train(self, dataset=None, train_epoch=50, newModel=False, run_time=0,
                batch_size=32, num_workers=0):
        if newModel:
            self.model = RankNet(self.model_size)
            parameters = filter(lambda p: p.requires_grad,
                                self.model.parameters())
            self.optimizer = torch.optim.Adam(parameters)
        create_exp_dir(self.save_model_path)

        if dataset is None:
            dataset =self.data_set

        data_queue = torch.utils.data.DataLoader(dataset=dataset,
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 num_workers=num_workers)

        def rate_fun(outputs, labels, topk=(1, 1)):
            outputs = outputs.cpu()
            predicted = outputs.detach().numpy()
            predicted[predicted >= 0.5] = 1
            predicted[predicted < 0.5] = 0
            total = labels.size(0)
            correct = np.sum(predicted.reshape(1, -1)
                             == labels.cpu().numpy().reshape(1, -1))
            return torch.from_numpy(np.array([correct/total])), torch.from_numpy(np.array([correct/total]))

        # train model
        step = 0
        for epoch in range(train_epoch):
            train_loss, train_top1, train_top5, step = train(
                data_queue, self.model, self.optimizer, step, self.criterion, rate_static=rate_fun)
            logging.info("[Epoch {0:>4d}] [Train] loss {1:.3f} error Top1 {2:.2f} error Top5 {3:.2f}".format(
                epoch, train_loss, train_top1, train_top5))
        # save model
        torch.save(self.model.state_dict(), os.path.join(
            self.save_model_path, "Seq2Rank_run_{0:>2d}.ckpt".format(run_time)))

# Following code are used to train the seq2seq


class auto_seq2seq:
    def __init__(self, data_path, model_save_path, model_load_path, hidden_size=32, max_vocab=4000,device='cuda'):
        self.src = SourceField()
        self.tgt = TargetField()
        self.max_length = 90
        self.data_path = data_path
        self.model_save_path = model_save_path
        self.model_load_path = model_load_path

        def len_filter(example):
            return len(example.src) <= self.max_length and len(example.tgt) <= self.max_length
        self.trainset = torchtext.data.TabularDataset(path=os.path.join(self.data_path, 'train'),
                                                      format='tsv',
                                                      fields=[
                                                          ('src', self.src), ('tgt', self.tgt)],
                                                      filter_pred=len_filter)
        self.devset = torchtext.data.TabularDataset(path=os.path.join(self.data_path, 'eval'), format='tsv',
                                                    fields=[
                                                        ('src', self.src), ('tgt', self.tgt)],
                                                    filter_pred=len_filter)
        self.src.build_vocab(self.trainset, max_size=max_vocab)
        self.tgt.build_vocab(self.trainset, max_size=max_vocab)
        weight = torch.ones(len(self.tgt.vocab))
        pad = self.tgt.vocab.stoi[self.tgt.pad_token]
        self.loss = Perplexity(weight, pad)
        self.loss.cuda()
        self.optimizer = None
        self.hidden_size = hidden_size
        self.bidirectional = True
        encoder = EncoderRNN(len(self.src.vocab), self.max_length, self.hidden_size,
                             bidirectional=self.bidirectional, variable_lengths=True)
        decoder = DecoderRNN(len(self.tgt.vocab), self.max_length, self.hidden_size * 2 if self.bidirectional else self.hidden_size,
                             dropout_p=0.2, use_attention=True, bidirectional=self.bidirectional,
                             eos_id=self.tgt.eos_id, sos_id=self.tgt.sos_id)
        self.device = device
        self.seq2seq = Seq2seq(encoder, decoder).cuda()
        for param in self.seq2seq.parameters():
            param.data.uniform_(-0.08, 0.08)

    def train(self, epoch=20, resume=False):
        t = SupervisedTrainer(loss=self.loss, batch_size=96,
                              checkpoint_every=1000,
                              print_every=1000, expt_dir=self.model_save_path)
        self.seq2seq = t.train(self.seq2seq, self.trainset,
                               num_epochs=epoch, dev_data=self.devset,
                               optimizer=self.optimizer,
                               teacher_forcing_ratio=0.5,
                               resume=resume)