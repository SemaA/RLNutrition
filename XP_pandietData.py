#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 11:45:24 2019

@author: sema
"""
import pickle

with open('data/conso_ad.p', 'rb') as handle:
    conso = pickle.load(handle)

with open('data/references.p', 'rb') as handle:
    adequacy, moderation = pickle.load(handle)


