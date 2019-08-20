#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 09:50:54 2019

@author: sema
"""

import pandas as pd 
import pickle
from userModel import *
from preprocessData import createData, computePortions

level = 'codsougr'
# Import data
with open("data/conso_ad.p", 'rb') as fp:
    df_conso = pickle.load(fp)

with open("data/adults.p", 'rb') as fp:
    df_users = pickle.load(fp)

# Generate Sequence Objects

df_conso['qte_nette'] = df_conso.qte_nette.astype('float64')
meanQuantities = df_conso.groupby(['codsougr', 'codsougr_name'])['qte_nette'].mean().to_dict()

with open("data/meanQuantitiesDict.p", 'wb') as fp:
    pickle.dump(meanQuantities, fp)


users, seqs = createData(df_conso, df_users)
portionsDict = computePortions(seqs, level)

with open("data/portionsDict.p", 'wb') as fp:
    pickle.dump(portionsDict, fp)