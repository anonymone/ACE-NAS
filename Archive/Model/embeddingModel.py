import sys
import os
import argparse
import logging
import torch
from torch.optim.lr_scheduler import StepLR
import torchtext
import numpy as np

import seq2seq
from seq2seq.trainer import SupervisedTrainer
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq
from seq2seq.loss import Perplexity
from seq2seq.optim import Optimizer
from seq2seq.dataset import SourceField, TargetField
from seq2seq.evaluator import Predictor
from seq2seq.util.checkpoint import Checkpoint

log_format = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')

device = 'cuda' if torch.cuda.is_available() else "cpu"

class EmbeddingModel:
    def __init__(self, opt):
        logging.info("loading checkpoint from {}".format(os.path.join(opt.expt_dir, opt.load_checkpoint)))
        checkpoint_path = os.path.join(opt.expt_dir, opt.load_checkpoint)
        checkpoint = Checkpoint.load(checkpoint_path)
        self.seq2seq = checkpoint.model
        self.input_vocab = checkpoint.input_vocab
        self.output_vocab = checkpoint.output_vocab

    def encode(self, code):
        '''
        Inputs: a string array, each row is an input sequence.
        Outputs: a numpy array, each row is an output vector.
        '''
        vectors = []
        for seqStr in code:
            seq = seqStr.strip().split()
            src_id_seq = torch.LongTensor([self.input_vocab.stoi[tok] for tok in seq]).view(1,-1)
            src_id_seq = src_id_seq.to(device)
            with torch.no_grad():
                output, hiden = self.seq2seq.encoder(src_id_seq, [len(seq)])
            hiden_cpu = hiden.cpu()
            vectors.append(hiden_cpu.data.numpy().reshape(-1))
        return np.array(vectors)
    
    def encode2numpy(self, code, withfitness =True):
        if withfitness:
            decList = []
            values = []
            for decString, value in code:
                decList.append(decString)
                values.append(value)
            return np.hstack([self.encode(decList),np.array(values)])
        else:
            decList = []
            for decString in code:
                decList.append(decString)
            return np.array(decList)


def trainEmbeddingModel(opt):
    # Prepare dataset
    src = SourceField()
    tgt = TargetField()
    max_len = 90
    def len_filter(example):
        return len(example.src) <= max_len and len(example.tgt) <= max_len
    train = torchtext.data.TabularDataset(
        path=opt.train_path, format='tsv',
        fields=[('src', src), ('tgt', tgt)],
        filter_pred=len_filter
    )
    dev = torchtext.data.TabularDataset(
        path=opt.dev_path, format='tsv',
        fields=[('src', src), ('tgt', tgt)],
        filter_pred=len_filter
    )
    src.build_vocab(train, max_size=1000)
    tgt.build_vocab(train, max_size=1000)
    input_vocab = src.vocab
    output_vocab = tgt.vocab

    # NOTE: If the source field name and the target field name
    # are different from 'src' and 'tgt' respectively, they have
    # to be set explicitly before any training or inference
    # seq2seq.src_field_name = 'src'
    # seq2seq.tgt_field_name = 'tgt'

    # Prepare loss
    weight = torch.ones(len(tgt.vocab))
    pad = tgt.vocab.stoi[tgt.pad_token]
    loss = Perplexity(weight, pad)
    if torch.cuda.is_available():
        loss.cuda()

    seq2seq = None
    optimizer = None
    if not opt.resume:
        # Initialize model
        hidden_size=128
        bidirectional = True
        encoder = EncoderRNN(len(src.vocab), max_len, hidden_size,
                            bidirectional=bidirectional, variable_lengths=True)
        decoder = DecoderRNN(len(tgt.vocab), max_len, hidden_size * 2 if bidirectional else hidden_size,
                            dropout_p=0.2, use_attention=True, bidirectional=bidirectional,
                            eos_id=tgt.eos_id, sos_id=tgt.sos_id)
        seq2seq = Seq2seq(encoder, decoder)
        if torch.cuda.is_available():
            seq2seq.cuda()

        for param in seq2seq.parameters():
            param.data.uniform_(-0.08, 0.08)

        # Optimizer and learning rate scheduler can be customized by
        # explicitly constructing the objects and pass to the trainer.
        #
        # optimizer = Optimizer(torch.optim.Adam(seq2seq.parameters()), max_grad_norm=5)
        # scheduler = StepLR(optimizer.optimizer, 1)
        # optimizer.set_scheduler(scheduler)

    # train
    t = SupervisedTrainer(loss=loss, batch_size=96,
                        checkpoint_every=1000,
                        print_every=1000, expt_dir=opt.expt_dir)

    seq2seq = t.train(seq2seq, train,
                    num_epochs=20, dev_data=dev,
                    optimizer=optimizer,
                    teacher_forcing_ratio=0.5,
                    resume=opt.resume)
# try:
#     raw_input          # Python 2
# except NameError:
#     raw_input = input  # Python 3

# predictor = Predictor(seq2seq, input_vocab, output_vocab)

# while True:
#     seq_str = raw_input("Type in a source sequence:")
#     seq = seq_str.strip().split()
#     print(predictor.predict(seq))

if __name__ == "__main__":
    import sys
    sys.path.insert(0,"./")
    from Model import surroogate
    from Model import individual
    import numpy as np
    import pandas as pd
    import argparse

    parser = argparse.ArgumentParser(
        "Embedding Test")
    # Predictor model setting 
    parser.add_argument('--PredictorModelDataset', dest='predictDataset', default='./Dataset/encodeData/surrogate.txt')
    parser.add_argument('--PredictorModelPath', dest='predictPath', default='./Dataset/encodeData/RankModel/')
    parser.add_argument('--PredictorModelEpoch', dest='predictEpoch', default= 20)


    parser.add_argument('--EmbeddingTrainPath', action='store', dest='train_path', default='./Dataset/encodeData/data.txt',
                    help='Path to train data')
    parser.add_argument('--EmdeddingDevPath', action='store', dest='dev_path', default='./Dataset/encodeData/data_val.txt',
                        help='Path to dev data')
    parser.add_argument('--EmdeddingExptDir', action='store', dest='expt_dir', default='./Dataset/encodeData',
                        help='Path to experiment directory. If load_checkpoint is True, then path to checkpoint directory has to be provided')
    parser.add_argument('--EmdeddingLoadCheckpoint', action='store', dest='load_checkpoint', default='2019_08_26_07_35_34',
                        help='The name of the checkpoint to load, usually an encoded time string')
    parser.add_argument('--EmdeddingResume', action='store_true', dest='resume',
                        default=False,
                        help='Indicates if training has to be resumed from the latest checkpoint') 
    parser.add_argument('--crossoverRate', type=float,
                        default=0.2, help='The propability rate of crossover.')
    parser.add_argument('--mutationRate', type=float,
                        default=0.2, help='The propability rate of crossover.')
    args = parser.parse_args()

    pop = individual.SEEPopulation(popSize=30, objSize=2, args=args)
    popString = pop.toString()

    model = EmbeddingModel(args)
    encodeNumpy = model.encode2numpy(popString)
    PredicDataset = pd.read_csv("./Dataset/encodeData/surrogate.txt")
    PredicDataset = surroogate.RankNetDataset(PredicDataset.values)
    PredicDataset.addData(encodeNumpy[:,:-1])
    predictor = surroogate.Predictor(encoder=model, modelSavePath=args.predictPath)
    predictor.trian(dataset=PredicDataset, trainEpoch=args.predictEpoch)