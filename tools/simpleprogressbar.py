# -*- coding: utf-8 -*-
#
# Copyright © 2016 Clay L. McLeod <clay.l.mcleod@gmail.com>
#
# Distributed under terms of the MIT license.

import sys, math

class simpleprogressbar(object):

    def __init__(self, width=10):
        self.width = width
        self.percentage = 0.0

    def update(self, percentage):
        self.percentage = percentage

    def render(self):
        frac = self.percentage / 100.0
        pos = int(math.floor(frac * self.width))
        sys.stdout.write('['+'='*pos+' '*(self.width-pos)+']')
        backspaces = '\b' * (self.width + 2)
        sys.stdout.write(backspaces)
        sys.stdout.flush()
