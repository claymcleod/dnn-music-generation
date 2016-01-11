#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2016 Clay L. McLeod <clay.l.mcleod@gmail.com>
#
# Distributed under terms of the MIT license.

from __future__ import print_function

import os, glob
import numpy as np

from sets import Set
from config import config
from tools.datatools import datatools
from tools.nntools import nntools

_config = config.get_config()
data_dir = _config['data_dir']

fft_dir = os.path.join(data_dir, 'fft')
fft_glob = os.path.join(fft_dir, '*.npy')

filenames = Set()

for g in glob.glob(fft_glob):
    filenames.add(g.replace('_x.npy','').replace('_y.npy',''))

for f in filenames:
    X_train = np.load(f+'_x.npy')
    y_train = np.load(f+'_y.npy')

    model = nntools.build_lstm_network(X_train.shape[2], 2048)
    model.fit(X_train, y_train, nb_epoch=20, batch_size=128)
