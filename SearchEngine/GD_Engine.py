import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from SearchEngine.Utils.GD_tools import Encoder, Decoder
from SearchEngine.Engine_Interface import population
from Coder.ACE import code_parser, ACTION, OPERATIONS_large, OPERATIONS, build_ACE


class GD_population(population):
    def __init__(self,
                 ind_params,
                 obj_number=1,
                 ind_generator=build_ACE,
                 pop_size=0):
        super(GD_population, self).__init__(
            obj_number=obj_number,
            pop_size=pop_size,
            ind_generator=ind_generator,
            ind_params=ind_params)
        self.code_parser = code_parser(
            node_search_space=OPERATIONS_large, action_search_space=ACTION)

    def add_new_inds(self, dec_list):
        for dec in dec_list:
            ind = self.ind_generator(self.obj_number, self.ind_params)
            dec = np.array(dec).reshape(2,-1)
            ind.set_dec((dec[0,:], dec[1,:]))
            logging.info("[New sample] [{0}] {1}".format(ind.get_Id(), ind.to_string()))
            self.add_ind(ind)

    def __parser(self, code: 'code'):
        unit_list = list()
        for action, p1, p2 in code:
            # action = self.code_parser.get_action_token(action)
            # if self.code_parser.get_action(action) == ACTION[1]:  # add_node_C
            #     p1, p2 = self.code_parser.get_op_token(p1), p2
            # # substitute_node_B_for_type
            # elif self.code_parser.get_action(action) == ACTION[3]:
            #     p1, p2 = p1, self.code_parser.get_op_token(p2)
            # else:
            #     pass
            unit_list.extend([action, p1, p2])
        return unit_list

    def to_matrix(self) -> '[[dec..., fitness...]]':
        matrix = list(map(lambda x: self.__parser(x.get_dec()[0]) + self.__parser(x.get_dec()[1]), self.individuals.values()))
        values = list(map(lambda x: x.get_fitness()[0], self.individuals.values()))
        return matrix, values


class NAO(nn.Module):
    def __init__(self,
                 encoder_layers,
                 encoder_vocab_size,
                 encoder_hidden_size,
                 encoder_dropout,
                 encoder_length,
                 source_length,
                 encoder_emb_size,
                 mlp_layers,
                 mlp_hidden_size,
                 mlp_dropout,
                 decoder_layers,
                 decoder_vocab_size,
                 decoder_hidden_size,
                 decoder_dropout,
                 decoder_length,
                 ):
        super(NAO, self).__init__()
        self.encoder = Encoder(
            encoder_layers,
            encoder_vocab_size,
            encoder_hidden_size,
            encoder_dropout,
            encoder_length,
            source_length,
            encoder_emb_size,
            mlp_layers,
            mlp_hidden_size,
            mlp_dropout,
        )
        self.decoder = Decoder(
            decoder_layers,
            decoder_vocab_size,
            decoder_hidden_size,
            decoder_dropout,
            decoder_length,
            encoder_length
        )

        self.flatten_parameters()

    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    def forward(self, input_variable, target_variable=None):
        encoder_outputs, encoder_hidden, arch_emb, predict_value = self.encoder(
            input_variable)
        decoder_hidden = (arch_emb.unsqueeze(0), arch_emb.unsqueeze(0))
        decoder_outputs, decoder_hidden, ret = self.decoder(
            target_variable, decoder_hidden, encoder_outputs)
        decoder_outputs = torch.stack(decoder_outputs, 0).permute(1, 0, 2)
        arch = torch.stack(ret['sequence'], 0).permute(1, 0, 2)
        return predict_value, decoder_outputs, arch

    def generate_new_arch(self, input_variable, predict_lambda=1, direction='-'):
        encoder_outputs, encoder_hidden, arch_emb, predict_value, new_encoder_outputs, new_arch_emb = self.encoder.infer(
            input_variable, predict_lambda, direction=direction)
        new_encoder_hidden = (new_arch_emb.unsqueeze(0),
                              new_arch_emb.unsqueeze(0))
        decoder_outputs, decoder_hidden, ret = self.decoder(
            None, new_encoder_hidden, new_encoder_outputs)
        new_arch = torch.stack(ret['sequence'], 0).permute(1, 0, 2)
        return new_arch
