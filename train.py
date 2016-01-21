#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2016 Clay L. McLeod <clay.l.mcleod@gmail.com>
#
# Distributed under terms of the MIT license.

from __future__ import print_function


from config import config
from tools.datatools import datatools

_config = config.get_config()
data_dir = _config['data_dir']
epochs_per_round = _config['epochs_per_round']
max_training_iterations = _config['max_training_iterations']
hidden_dim = _config['hidden_dim']

datatools.train_dnn(data_dir, epochs_per_round, max_training_iterations, hidden_dim)
