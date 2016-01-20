#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2016 Clay L. McLeod <clay.l.mcleod@gmail.com>
#
# Distributed under terms of the MIT license.


from config import config
from tools.datatools import datatools

_config = config.get_config()
data_dir = _config['data_dir']
seql = _config['seql']
generate_x_blocks = _config['generate_x_blocks']

datatools.generate_from_dnn(data_dir, seql, generate_x_blocks)
