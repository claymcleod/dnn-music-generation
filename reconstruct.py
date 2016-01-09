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
datatools.convert_gen_to_out(data_dir)
