# -*- coding: utf-8 -*-
#
# Copyright © 2016 Clay L. McLeod <clay.l.mcleod@gmail.com>
#
# Distributed under terms of the MIT license.

from __future__ import print_function


from keras.models import Sequential, Graph
from keras.layers.core import Dense, TimeDistributedDense, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM

class nntools(object):

    @staticmethod
    def get_current_model(freq_dim, hidden_dim):
        return nntools.build_lstm_network(freq_dim, hidden_dim)
        #return nntools.build_bidirectional_lstm_network(freq_dim, hidden_dim)


    @staticmethod
    def build_lstm_network(freq_dim, hidden_dim=64):

        # Define vanilla sequential model
        model = Sequential()

        model.add(TimeDistributedDense(output_dim=hidden_dim, input_dim=freq_dim))
        model.add(LSTM(output_dim=hidden_dim, input_dim=hidden_dim, return_sequences=False))
        model.add(Dense(output_dim=freq_dim, input_dim=hidden_dim))
        model.compile(loss='mean_squared_error', optimizer='rmsprop')
        return model

    @staticmethod
    def build_bidirectional_lstm_network(freq_dim, hidden_dim=64):

        model = Graph()
        model.add_input(name='input', input_shape=(freq_dim,))
        model.add_node(Embedding(freq_dim, 128, input_shape=freq_dim), input='input', name='embed')
        model.add_node(LSTM(hidden_dim), name='forward', input='embed')
        model.add_node(LSTM(hidden_dim, go_backwards=True), name='backward', input='embed')
        model.add_node(Dropout(0.5), name='dropout', inputs=['forward', 'backward'])
        model.add_node(Dense(freq_dim, activation='relu'), name='relu', input='dropout')
        model.add_output(name='output', input='relu')
        return model
