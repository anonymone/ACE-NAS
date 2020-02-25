import sys
sys.path.append('./')
import argparse
import os

from Evaluator.test_script import seq2seq_eval

parser = argparse.ArgumentParser('Eval seq2seq.')
parser.add_argument('--ckpt_root', type=str, default='./Res/nasbench/seq2seq/checkpoints/')
parser.add_argument('--ckpt_name', type=str, default='2020_02_21_12_50_48')
args = parser.parse_args()

eval_tool = seq2seq_eval.eval_tool(ckpt_path=os.path.join(args.ckpt_root, args.ckpt_name))

input_str = input("Enter INPUT: ")

while input_str != 'exit':
    print("OUTPUT: {0}".format(" ".join(eval_tool.predict(input_str))))
    input_str = input("Enter INPUT: ")
