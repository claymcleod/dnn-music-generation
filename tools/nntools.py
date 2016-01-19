# -*- coding: utf-8 -*-
#
# Copyright © 2016 Clay L. McLeod <clay.l.mcleod@gmail.com>
#
# Distributed under terms of the MIT license.

from __future__ import print_function

import os, glob

from keras.models import Sequential
from keras.layers.core import Dense, TimeDistributedDense
from keras.layers.recurrent import LSTM

class nntools(object):

    @staticmethod
    def get_current_model(freq_dim, hidden_dim):
        return nntools.build_lstm_network(freq_dim, hidden_dim)


    @staticmethod
    def build_lstm_network(freq_dim, hidden_dim):

        # Define vanilla sequential model
        model = Sequential()

        model.add(TimeDistributedDense(output_dim=hidden_dim, input_dim=freq_dim))
        model.add(LSTM(output_dim=hidden_dim, input_dim=hidden_dim, return_sequences=False))
        model.add(Dense(output_dim=freq_dim, input_dim=hidden_dim))
        model.compile(loss='mean_squared_error', optimizer='rmsprop')
        return model
