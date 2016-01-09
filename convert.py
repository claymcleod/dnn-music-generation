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
datatools.convert_mp3s_to_wav(data_dir)
datatools.convert_wav_to_fft(data_dir)
