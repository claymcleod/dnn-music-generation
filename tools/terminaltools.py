#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2016 Clay L. McLeod <clay.l.mcleod@gmail.com>
#
# Distributed under terms of the MIT license.

from sys import stdout

def write_flush(sentence):
    stdout.write(sentence)
    stdout.flush()
