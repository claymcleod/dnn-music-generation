# -*- coding: utf-8 -*-
#
# Copyright © 2016 Clay L. McLeod <clay.l.mcleod@gmail.com>
#
# Distributed under terms of the MIT license.

from __future__ import print_function

import os, sys


# Supress GPU and backend declarations from keras
suppress_errors = True

if suppress_errors:
    sys.stderr = open(os.devnull, 'w')

def get_config():
    config = {}
    config['data_dir'] = './data' # This should stay the same
    config['block_size'] = 44100 // 8 # Block size (default 44100 // 8, or an eigth of a second)
    config['seql'] = 40 # How far back to 'remember' for each training example (default 40, or 5 seconds)
    config['max_training_iterations'] = 500 # Max training for dnn (default 500)
    config['epochs_per_round'] = 10 # Epochs per round (default 10)
    config['generate_x_blocks'] = 320 # Blocks to generate (default 320, or 40 seconds)
    return config
