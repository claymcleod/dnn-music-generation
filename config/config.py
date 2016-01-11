# -*- coding: utf-8 -*-
#
# Copyright © 2016 Clay L. McLeod <clay.l.mcleod@gmail.com>
#
# Distributed under terms of the MIT license.

def get_config():
    config = {}
    config['data_dir'] = './data'
    config['block_size'] = 44100 // 8
    config['seql'] = 20
    return config
