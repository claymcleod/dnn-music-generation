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
seql = _config['seql']

fft_dir = os.path.join(data_dir, 'fft')
fft_glob = os.path.join(fft_dir, '*.npy')

weights_dir = os.path.join(data_dir, 'weights')
datatools.ensure_dir_exists(weights_dir)

filenames = Set()

for g in glob.glob(fft_glob):
    filenames.add(g.replace('_x.npy','').replace('_y.npy',''))

for f in filenames:
    X_train = np.load(f+'_x.npy')
    y_train = np.load(f+'_y.npy')
    filename = f.split('/')[-1]
    weight_file = os.path.join(weights_dir, filename+'.hdf5')

    i = 0
    l = len(X_train)
    output = np.zeros(X_train.shape)
    output = np.concatenate(X_train[0:20], output)

    model = nntools.build_lstm_network(X_train.shape[2], 2048)
    if os.path.exists(weight_file):
	model.load_weights(weight_file)

    while True:
	next_val = model.predict(output[i:i+seql])
	print(next_val.shape)
        for k in range(0, seql-1):
	    for x in range(0, output.shape[2]):
	        output[i+seql+1][k][x] = output[i+seql][k-1][x]

        for x in range(0, output.shape[2]):
            output[i+seql+1][seql][x] = next_val[x]

