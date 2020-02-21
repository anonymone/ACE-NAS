import sys
sys.path.append('./')

import seq2seq
from seq2seq.trainer import SupervisedTrainer
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq
from seq2seq.loss import Perplexity
from seq2seq.optim import Optimizer
from seq2seq.dataset import SourceField, TargetField
from seq2seq.evaluator import Predictor
from seq2seq.util.checkpoint import Checkpoint

class eval_tool:
    def __init__(self, ckpt_path='./Res/PretrainModel/2019_12_27_08_48_21/'):
        checkpoint = Checkpoint.load(ckpt_path)
        self.seq2seq = checkpoint.model
        self.input_vocab = checkpoint.input_vocab
        self.output_vocab = checkpoint.output_vocab
        self.predictor = Predictor(self.seq2seq, self.input_vocab, self.output_vocab)
    
    def predict(self, input_str):
        return self.predictor.predict(input_str.strip().split())