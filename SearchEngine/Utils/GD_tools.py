import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from Evaluator.Utils.recoder import AvgrageMeter

SOS_ID = 0
EOS_ID = 0

class ACEDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets=None, train=True, sos_id=0, eos_id=0):
        super(ACEDataset, self).__init__()
        if targets is not None:
            assert len(inputs) == len(targets)
        self.inputs = copy.deepcopy(inputs)
        self.targets = copy.deepcopy(targets)
        self.train = train
        self.sos_id = sos_id
        self.eos_id = eos_id
    
    def __getitem__(self, index):
        encoder_input = self.inputs[index]
        encoder_target = None
        if self.targets is not None:
            encoder_target = [self.targets[index]]
        if self.train:
            decoder_input = [self.sos_id] + encoder_input[:-1]
            sample = {
                'encoder_input': torch.LongTensor(encoder_input),
                'encoder_target': torch.FloatTensor(encoder_target),
                'decoder_input': torch.LongTensor(decoder_input),
                'decoder_target': torch.LongTensor(encoder_input),
            }
        else:
            sample = {
                'encoder_input': torch.LongTensor(encoder_input),
                'decoder_target': torch.LongTensor(encoder_input),
            }
            if encoder_target is not None:
                sample['encoder_target'] = torch.FloatTensor(encoder_target)
        return sample
    
    def __len__(self):
        return len(self.inputs)

class dataset_utils(object):
    @staticmethod
    def normalize(inputs, labels):
        min_val = min(labels)
        max_val = max(labels)
        encoder_target = [(i - min_val) / (max_val - min_val) for i in labels]
        return inputs, encoder_target

    @staticmethod
    def sort(inputs, labels):
        sorted_indices = np.argsort(labels)[::-1]
        inputs = [inputs[i] for i in sorted_indices]
        labels = [labels[i] for i in sorted_indices]
        return inputs, labels

    @staticmethod
    def get_dataset(inputs, labels):
        dataset = list(zip(inputs, labels))
        np.random.shuffle(dataset)
        encoder_input, encoder_target = list(zip(*dataset))
        train_encoder_input = encoder_input
        train_encoder_target = encoder_target
        valid_encoder_input = encoder_input
        valid_encoder_target = encoder_target
        return (train_encoder_input, train_encoder_target), (valid_encoder_input, valid_encoder_target)
    
    @staticmethod
    def get_data_loader(train_dataset=None,
                        valid_dataset=None,
                        data_container=ACEDataset, 
                        sos_id=0, 
                        eos_id=0, 
                        train_batch_size=36, 
                        eval_batch_size=36):
        if train_dataset is not None:
            train_encoder_input, train_encoder_target = train_dataset 
            train_dataset = ACEDataset(train_encoder_input, train_encoder_target,True, sos_id=sos_id, eos_id=eos_id)
            train_queue = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, pin_memory=True)
        else:
            train_queue = None
        if valid_dataset is not None:
            valid_encoder_input, valid_encoder_target = valid_dataset
            valid_dataset = ACEDataset(valid_encoder_input, valid_encoder_target, False, sos_id=sos_id, eos_id=eos_id)
            valid_queue = torch.utils.data.DataLoader(valid_dataset, batch_size=eval_batch_size, shuffle=False, pin_memory=True)
        else:
            valid_queue = None
        return train_queue, valid_queue
    
    @staticmethod
    def data_augmentation(dataset, labels):
        '''
        :TODO use the redundant encoding strings to expend the data size.
        '''
        return dataset, labels


class utils(object):
    @staticmethod
    def pairwise_accuracy(la, lb):
        n = len(la)
        assert n == len(lb)
        total = 0
        count = 0
        for i in range(n):
            for j in range(i+1, n):
                if la[i] >= la[j] and lb[i] >= lb[j]:
                    count += 1
                if la[i] < la[j] and lb[i] < lb[j]:
                    count += 1
                total += 1
        return float(count) / total
    
    @staticmethod
    def hamming_distance(la, lb):
        N = len(la)
        assert N == len(lb)
    
        def _hamming_distance(s1, s2):
            n = len(s1)
            assert n == len(s2)
            c = 0
            for i, j in zip(s1, s2):
                if i != j:
                    c += 1
            return c
    
        dis = 0
        for i in range(N):
            line1 = la[i]
            line2 = lb[i]
            dis += _hamming_distance(line1, line2)
        return dis / N


def nao_train(train_queue, model, optimizer, controller_trade_off=0.8, controller_grad_bound=5.0, device='cuda'):
    objs = AvgrageMeter()
    mse = AvgrageMeter()
    nll = AvgrageMeter()
    model.train()
    for step, sample in enumerate(train_queue):
        encoder_input = sample['encoder_input']
        encoder_target = sample['encoder_target']
        decoder_input = sample['decoder_input']
        decoder_target = sample['decoder_target']
        
        encoder_input = encoder_input.to(device)
        encoder_target = encoder_target.to(device).requires_grad_()
        decoder_input = decoder_input.to(device)
        decoder_target = decoder_target.to(device)
        
        optimizer.zero_grad()
        predict_value, log_prob, arch = model(encoder_input, decoder_input)
        loss_1 = F.mse_loss(predict_value.squeeze(), encoder_target.squeeze())
        loss_2 = F.nll_loss(log_prob.contiguous().view(-1, log_prob.size(-1)), decoder_target.view(-1))
        loss = controller_trade_off * loss_1 + (1 - controller_trade_off) * loss_2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), controller_grad_bound)
        optimizer.step()
        
        n = encoder_input.size(0)
        objs.update(loss.data, n)
        mse.update(loss_1.data, n)
        nll.update(loss_2.data, n)
        
    return objs.avg, mse.avg, nll.avg

def nao_valid(queue, model, device='cuda'):
    inputs = []
    targets = []
    predictions = []
    archs = []
    with torch.no_grad():
        model.eval()
        for step, sample in enumerate(queue):
            encoder_input = sample['encoder_input']
            encoder_target = sample['encoder_target']
            decoder_target = sample['decoder_target']
            
            encoder_input = encoder_input.to(device)
            encoder_target = encoder_target.to(device)
            decoder_target = decoder_target.to(device)
            
            predict_value, logits, arch = model(encoder_input)
            n = encoder_input.size(0)
            inputs += encoder_input.data.squeeze().tolist()
            targets += encoder_target.data.squeeze().tolist()
            predictions += predict_value.data.squeeze().tolist()
            archs += arch.data.squeeze().tolist()
    pa = utils.pairwise_accuracy(targets, predictions)
    hd = utils.hamming_distance(inputs, archs)
    return pa, hd

def nao_infer(queue, model, step, direction='+', device='cuda'):
    new_arch_list = []
    model.eval()
    for i, sample in enumerate(queue):
        encoder_input = sample['encoder_input']
        encoder_input = encoder_input.to(device)
        model.zero_grad()
        new_arch = model.generate_new_arch(encoder_input, step, direction=direction)
        new_arch_list.extend(new_arch.data.squeeze().tolist())
    return new_arch_list

class Encoder(nn.Module):
    def __init__(self,
                 layers,
                 vocab_size,
                 hidden_size,
                 dropout,
                 length,
                 source_length,
                 emb_size,
                 mlp_layers,
                 mlp_hidden_size,
                 mlp_dropout,
                 ):
        super(Encoder, self).__init__()
        self.layers = layers
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.length = length
        self.source_length = source_length
        self.mlp_layers = mlp_layers
        self.mlp_hidden_size = mlp_hidden_size
        
        self.embedding = nn.Embedding(self.vocab_size, self.emb_size)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, self.layers, batch_first=True, dropout=dropout)
        self.mlp = nn.Sequential()
        for i in range(self.mlp_layers):
            if i == 0:
                self.mlp.add_module('layer_{}'.format(i), nn.Sequential(
                    nn.Linear(self.hidden_size, self.mlp_hidden_size),
                    nn.ReLU(inplace=False),
                    nn.Dropout(p=mlp_dropout)))
            else:
                self.mlp.add_module('layer_{}'.format(i), nn.Sequential(
                    nn.Linear(self.mlp_hidden_size, self.mlp_hidden_size),
                    nn.ReLU(inplace=False),
                    nn.Dropout(p=mlp_dropout)))
        self.regressor = nn.Linear(self.hidden_size if self.mlp_layers == 0 else self.mlp_hidden_size, 1)
    
    def forward(self, x):
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        if self.source_length != self.length:
            assert self.source_length % self.length == 0
            ratio = self.source_length // self.length
            embedded = embedded.view(-1, self.source_length // ratio, ratio * self.emb_size)
        out, hidden = self.rnn(embedded)
        out = F.normalize(out, 2, dim=-1)
        encoder_outputs = out
        encoder_hidden = hidden
        
        out = torch.mean(out, dim=1)
        out = F.normalize(out, 2, dim=-1)
        arch_emb = out
        
        out = self.mlp(out)
        out = self.regressor(out)
        predict_value = torch.sigmoid(out)
        return encoder_outputs, encoder_hidden, arch_emb, predict_value
    
    def infer(self, x, predict_lambda, direction='-'):
        encoder_outputs, encoder_hidden, arch_emb, predict_value = self(x)
        grads_on_outputs = torch.autograd.grad(predict_value, encoder_outputs, torch.ones_like(predict_value))[0]
        if direction == '+':
            new_encoder_outputs = encoder_outputs + predict_lambda * grads_on_outputs
        elif direction == '-':
            new_encoder_outputs = encoder_outputs - predict_lambda * grads_on_outputs
        else:
            raise ValueError('Direction must be + or -, got {} instead'.format(direction))
        new_encoder_outputs = F.normalize(new_encoder_outputs, 2, dim=-1)
        new_arch_emb = torch.mean(new_encoder_outputs, dim=1)
        new_arch_emb = F.normalize(new_arch_emb, 2, dim=-1)
        return encoder_outputs, encoder_hidden, arch_emb, predict_value, new_encoder_outputs, new_arch_emb

class Attention(nn.Module):
    def __init__(self, input_dim, source_dim=None, output_dim=None, bias=False):
        super(Attention, self).__init__()
        if source_dim is None:
            source_dim = input_dim
        if output_dim is None:
            output_dim = input_dim
        self.input_dim = input_dim
        self.source_dim = source_dim
        self.output_dim = output_dim
        self.input_proj = nn.Linear(input_dim, source_dim, bias=bias)
        self.output_proj = nn.Linear(input_dim + source_dim, output_dim, bias=bias)
        self.mask = None
    
    def set_mask(self, mask):
        self.mask = mask
    
    def forward(self, input, source_hids):
        batch_size = input.size(0)
        source_len = source_hids.size(1)

        # (batch, tgt_len, input_dim) -> (batch, tgt_len, source_dim)
        x = self.input_proj(input)

        # (batch, tgt_len, source_dim) * (batch, src_len, source_dim) -> (batch, tgt_len, src_len)
        attn = torch.bmm(x, source_hids.transpose(1, 2))
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))
        attn = F.softmax(attn.view(-1, source_len), dim=1).view(batch_size, -1, source_len)
        
        # (batch, tgt_len, src_len) * (batch, src_len, source_dim) -> (batch, tgt_len, source_dim)
        mix = torch.bmm(attn, source_hids)
        
        # concat -> (batch, tgt_len, source_dim + input_dim)
        combined = torch.cat((mix, input), dim=2)
        # output -> (batch, tgt_len, output_dim)
        output = torch.tanh(self.output_proj(combined.view(-1, self.input_dim + self.source_dim))).view(batch_size, -1, self.output_dim)
        
        return output, attn


class Decoder(nn.Module):
    KEY_ATTN_SCORE = 'attention_score'
    KEY_LENGTH = 'length'
    KEY_SEQUENCE = 'sequence'
    
    def __init__(self,
                 layers,
                 vocab_size,
                 hidden_size,
                 dropout,
                 length,
                 encoder_length,
                 ):
        super(Decoder, self).__init__()
        self.layers = layers
        self.hidden_size = hidden_size
        self.length = length
        self.encoder_length = encoder_length
        self.vocab_size = vocab_size
        self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, self.layers, batch_first=True, dropout=dropout)
        self.sos_id = SOS_ID
        self.eos_id = EOS_ID
        self.init_input = None
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.attention = Attention(self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.vocab_size)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def forward_step(self, x, hidden, encoder_outputs):
        batch_size = x.size(0)
        output_size = x.size(1)
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        output, hidden = self.rnn(embedded, hidden)
        output, attn = self.attention(output, encoder_outputs)
        
        predicted_softmax = F.log_softmax(self.out(output.contiguous().view(-1, self.hidden_size)), dim=1)
        predicted_softmax = predicted_softmax.view(batch_size, output_size, -1)
        return predicted_softmax, hidden, attn
    
    def forward(self, x, encoder_hidden=None, encoder_outputs=None):
        ret_dict = dict()
        ret_dict[Decoder.KEY_ATTN_SCORE] = list()
        if x is None:
            inference = True
        else:
            inference = False
        x, batch_size, length = self._validate_args(x, encoder_hidden, encoder_outputs)
        assert length == self.length
        decoder_hidden = self._init_state(encoder_hidden)
        decoder_outputs = []
        sequence_symbols = []
        lengths = np.array([length] * batch_size)
        
        def decode(step, step_output, step_attn):
            decoder_outputs.append(step_output)
            ret_dict[Decoder.KEY_ATTN_SCORE].append(step_attn)
            if step % 2 == 0:  # sample index, should be in [1, index-1]
                index = step // 2 % 10 // 2 + 3
                symbols = decoder_outputs[-1][:, 1:index].topk(1)[1] + 1
            else:  # sample operation, should be in [7, 11]
                symbols = decoder_outputs[-1][:, 7:].topk(1)[1] + 7
            
            sequence_symbols.append(symbols)
            
            eos_batches = symbols.data.eq(self.eos_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > step) & eos_batches) != 0
                lengths[update_idx] = len(sequence_symbols)
            return symbols
        
        decoder_input = x[:, 0].unsqueeze(1)
        for di in range(length):
            if not inference:
                decoder_input = x[:, di].unsqueeze(1)
            decoder_output, decoder_hidden, step_attn = self.forward_step(decoder_input, decoder_hidden, encoder_outputs)
            step_output = decoder_output.squeeze(1)
            symbols = decode(di, step_output, step_attn)
            decoder_input = symbols
        
        ret_dict[Decoder.KEY_SEQUENCE] = sequence_symbols
        ret_dict[Decoder.KEY_LENGTH] = lengths.tolist()
        
        return decoder_outputs, decoder_hidden, ret_dict
    
    def _init_state(self, encoder_hidden):
        """ Initialize the encoder hidden state. """
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([h for h in encoder_hidden])
        else:
            encoder_hidden = encoder_hidden
        return encoder_hidden
    
    def _validate_args(self, x, encoder_hidden, encoder_outputs):
        if encoder_outputs is None:
            raise ValueError("Argument encoder_outputs cannot be None when attention is used.")
        
        # inference batch size
        if x is None and encoder_hidden is None:
            batch_size = 1
        else:
            if x is not None:
                batch_size = x.size(0)
            else:
                batch_size = encoder_hidden[0].size(1)
        
        # set default input and max decoding length
        if x is None:
            x = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1).to(self.device)
            max_length = self.length
        else:
            max_length = x.size(1)
        
        return x, batch_size, max_length
    
    def eval(self):
        return
    
    def infer(self, x, encoder_hidden=None, encoder_outputs=None):
        decoder_outputs, decoder_hidden, _ = self(x, encoder_hidden, encoder_outputs)
        return decoder_outputs, decoder_hidden