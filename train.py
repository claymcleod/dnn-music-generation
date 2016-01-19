#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2016 Clay L. McLeod <clay.l.mcleod@gmail.com>
#
# Distributed under terms of the MIT license.

from __future__ import print_function

import os, glob
import numpy as np

from config import config
from tools.datatools import datatools

_config = config.get_config()
data_dir = _config['data_dir']
epochs_per_round = _config['epochs_per_round']
max_training_iterations = _config['max_training_iterations']

datatools.train_dnn(data_dir, epochs_per_round, max_training_iterations)
