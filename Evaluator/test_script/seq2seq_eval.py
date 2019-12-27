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

checkpoint = Checkpoint.load('./Res/PretrainModel/2019_12_27_08_48_21/')
seq2seq = checkpoint.model
input_vocab = checkpoint.input_vocab
output_vocab = checkpoint.output_vocab
predictor = Predictor(seq2seq, input_vocab, output_vocab)
seq_str = "2.2.12 8.12.4 10.2.1 5.0.0 12.13.10 5.6.10 8.12.7 1.9.2 13.13.13 3.11.0"
seq = seq_str.strip().split()
print(predictor.predict(seq))