#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2016 Clay L. McLeod <clay.l.mcleod@gmail.com>
#
# Distributed under terms of the MIT license.

from __future__ import print_function

import os, sys, glob
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
gen_dir = os.path.join(data_dir, 'gen')

datatools.ensure_dir_exists(weights_dir)
datatools.ensure_dir_exists(gen_dir)

filenames = Set()

for g in glob.glob(fft_glob):
    filenames.add(g.replace('_x.npy','').replace('_y.npy',''))

for f in filenames:
    X_train = np.load(f+'_x.npy')
    y_train = np.load(f+'_y.npy')
    filename = f.split('/')[-1]
    weight_file = os.path.join(weights_dir, filename+'.hdf5')
    trained_file_location = os.path.join(gen_dir, filename)

    print()
    file_str = "# Generating for '{}' #".format(filename)
    header_str = '#' * len(file_str)
    print(header_str)
    print(file_str)
    print(header_str)
    print()

    print("-- Preparing data...")
    output = np.zeros(X_train.shape)	
    output = np.append(X_train[0:seql], output, axis=0)
    fft_output = np.zeros((X_train.shape[0]-seql, X_train.shape[2]))

    print("-- Building model...")
    model = nntools.build_lstm_network(X_train.shape[2], 2048)

    print("-- Loading weights...")
    if os.path.exists(weight_file):
	model.load_weights(weight_file)
    
    i = 0
    l = len(X_train)
    while True:
	sys.stdout.write("-- Generating... ({}/{})\r".format(i+seql+1, output.shape[0]))
	sys.stdout.flush()
	if i+seql+1 >= output.shape[0]:
	    break

	next_val = model.predict(output[i:i+seql])

        for k in range(0, seql-1):
	    for x in range(0, output.shape[2]):
		output[i+seql+1][k][x] = next_val[k][x]
		fft_output[i][x] = next_val[seql-1][x]

	i = i + 1

    print("\n-- Saving numpy array...")
    print("Shape of output: {}".format(fft_output.shape))
    np.save(trained_file_location, fft_output)
