#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2016 Clay L. McLeod <clay.l.mcleod@gmail.com>
#
# Distributed under terms of the MIT license.

from __future__ import print_function

import os, glob, sys
import numpy as np
import matplotlib.pyplot as plt

from config import config
from tools.datatools import datatools
from scipy.io import wavfile

def write_flush(s):
    '''
    Quick helper method to write to std then flush
    '''

    sys.stdout.write(s)
    sys.stdout.flush()

_config = config.get_config()
data_dir = _config['data_dir']

out_dir = os.path.join(data_dir, 'out')
out_glob_file_path = os.path.join(out_dir, '*.wav')

plot_dir = os.path.join(data_dir, 'plot')

datatools.ensure_dir_exists(out_dir)
datatools.ensure_dir_exists(plot_dir)

for wav_data_file in glob.glob(out_glob_file_path):
    filename = wav_data_file.split('/')[-1]
    write_flush('Loading '+str(filename)+'...')

    png_filename = filename.split('/')[-1].replace('.wav','.png')
    savepath = os.path.join(plot_dir, png_filename)
    td_arr = wavfile.read(wav_data_file)
    write_flush('plotting spectrogram...')

    plt.specgram(td_arr[1])
    plt.savefig(savepath)
    plt.close()
    write_flush('finished.\n')




