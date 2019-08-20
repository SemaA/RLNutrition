#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 10:24:52 2019

@author: sema
"""
import numpy as np
s = 1
for i in range(10):
    print('Before the operations {}'.format(s))
    t = s+1
    nextS = np.random.randint(1,50)
    
    s = nextS
    print('After the operations {}'.format(s))