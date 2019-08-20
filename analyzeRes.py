#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 12:19:24 2019

@author: sema
"""

import pickle

with open('results/19_08_05_11_56.p', 'rb') as handle:
    resDict = pickle.load(handle)
    
userDiet = resDict['userDiet']


pastPandiet = userDiet.userDiet.pastPandietScores
