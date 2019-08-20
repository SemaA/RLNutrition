#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 15:25:55 2019

@author: sema
"""
import pandas as pd
from nutritionalScore import *


compo = pd.read_csv('benchmarkData/table_compo_sema.csv', header=0, sep=";")
socio = pd.read_csv('benchmarkData/table_indiv_sema.csv', header=0, sep=";",
                    dtype = {'sexe_ps':int, 'menopause':int})
nut_ref = pd.read_csv('benchmarkData/adequacyReferences.csv', header=0, sep=";", index_col=[0,1])
conso_test = pd.read_csv('benchmarkData/table_conso_sema1.csv', header=0, sep=";", 
                         dtype= {'qte_nette':float})

moderationRef = pd.read_csv('benchmarkData/moderationReferences.csv', header=0, sep=";", 
                            index_col=[0,1])

penRef = pd.read_csv('benchmarkData/penalties.csv', header=0, sep=";", 
                            index_col=0)

res = pd.read_csv('benchmarkData/PANDiet_sema.csv', header=0, sep=";", index_col=0, 
                  dtype= {'nomen':int})

# Change the type of the variable codal
compo.dropna(axis=0, how='all', inplace=True)
compo['codal'] = compo.codal.astype('int')
compo.set_index('codal', inplace=True)
compo = compo.apply(pd.to_numeric)
cols_compo = compo.columns


conso = pd.read_pickle('benchmarkData/conso_ad.p')
conso['codal']  = conso.codal.astype('int')
conso['qte_nette']  = conso.qte_nette.astype('float64')

# Select users for testing 
ids = res.index.tolist()[0:100]

#ids = res[res.Penalty == 1].index.tolist()
#idx = pd.IndexSlice
#conso_ = conso_test.loc[idx[ids,:,:], :]
conso_ = conso_test[conso_test.nomen.isin(ids)]
#conso_.reset_index(inplace=True)
c = conso_.rename(columns={'nojour':'jour'})
c.set_index(['nomen', 'jour', 'tyrep'], inplace=True)
c['codal']  = c.codal.astype('int')
c['qte_nette']  = c.qte_nette.astype('float64')
nj = 7

diet = computePandiet(c, compo, socio, nut_ref, moderationRef, nj, 'codal')