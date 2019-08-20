#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Sema
"""

import os
import pickle
import pandas as pd
import numpy as np
from collections import Counter 

from userModel import Food, createUserTable, getConsumptionSequences

def importData(path):
    with open(path, "rb") as fp:
        return pickle.load(fp)
    
def importPickle(path):
    with open(path, "rb") as fp:
        X = pickle.load(fp)
    return X

def importSubMatrix(path):
    with open(path, "rb") as fp:
        sub_dict = pickle.load(fp)
    S = {}
    for key, sub_mat in sub_dict.items():
        sub_mat.set_index(sub_mat.index.astype(int), inplace=True)
        sub_mat.columns = sub_mat.columns.astype(int)
        S[key] = sub_mat
    return S

def createItems(df):
    I = []
    for index, row in df.iterrows():
        i = Food(codsougr = row.codsougr, codsougr_name = row.codsougr_en)
        I.append(i)
    return I
        

def createData(df_conso, df_user):
    # Create user data
    print('Importing user data...')
    users = createUserTable(df_user)
    
    # Create conso data 
    print('Import conso data...')
    conso = df_conso.reset_index()
    U = getConsumptionSequences(conso, users)
    return users, U


def getConsoFromSequence(seq):
    """
    From a MealSequence object, get a pandas dataframe for the computation of 
    pandiet and the User object.
    
    Input :
        seq (MealSequence object)
    Output:
        df (pd.DataFrame) 
        user (User object)
    """
    L = []
    cols = ['nomen','jour','tyrep','codsougr','codsougr_name', 'qte_nette']
    
    nomen = seq.nomen
    for m in seq.meal_list:
        tyrep = m.tyrep
        jour = m.jour

        for food in m.meal:
            codsougr = food.codsougr
            codsougr_name = food.codsougr_name
            qte_nette = food.qte_nette
            L.append([nomen, jour, tyrep, codsougr, codsougr_name, qte_nette])
    df = pd.DataFrame(L, columns=cols)
    df_user = pd.Series(seq.user.as_dict()).to_frame().T
    return df, df_user




def computePortions(seqs, level):
    """
    Compute the portions given on the consumed qty in the database. 
    
    Input : 
        seqs (dict) key nomen, value MealSequence
        level (str) in ['codgr', 'codsougr', 'codal']
    """ 
    if level in ['codgr', 'codsougr', 'codal']:
        levelName = level +'_name'
        # Get two dataframes for men and women
        users = seqs.keys()
        df_men = []
        df_women = []
        for u in users:
            seq = seqs[u]
            df = getConsoFromSequence(seq)[0]
            if seq.user.sexe_ps == 1:
                df_men.append(df)
            else:
                df_women.append(df)
            
        women = pd.concat(df_women)
        men = pd.concat(df_men)
        
        women_portions = women.groupby([level, levelName]).qte_nette.mean().to_dict()
        men_portions = men.groupby([level, levelName]).qte_nette.mean().to_dict()
        
        women_portions = {(str(i[0]), i[1]):v for i,v in women_portions.items()}
        men_portions = {(str(i[0]), i[1]):v for i,v in men_portions.items()}

        return {'women':women_portions, 'men':men_portions}
    
    else:
        print('Enter a valid level (codgr, codsougr, codal) !')

def computeConsumptionFrequency(df_conso):
    """
    Given the level of classification, return the consumption frequency of items.
    """
    df_conso['codal'] = pd.to_numeric(df_conso['codal'])
    items = df_conso['codal']
    return Counter(items)

    
    
def computeGroupComposition(df_conso, df_composition, df_nomenclature, level):
    """
    Each food contributes to the mean of the group based on its consumption 
    frequency database.
    
    Input:
        itemFrequencies (dict) counter object
    
    """
    new_composition = df_composition.copy()
    new_composition.set_index('codal', inplace=True)
    cols = new_composition.columns.tolist() 
    
    
    df1 = pd.merge(df_composition, df_nomenclature[['codgr', 'codsougr', 'codal']], 
                   on='codal')
    
    itemFrequencies = computeConsumptionFrequency(df_conso)
    df1['counts'] = df1['codal'].map(itemFrequencies)
    
    levelCounts = df1.groupby(level).counts.sum()
    df1['level_count'] = df1[level].map(levelCounts)
    df1['weight'] = df1['counts'] / df1['level_count']
    
    
    df1.set_index('codal', inplace=True)
    
    df1.groupby(level).apply(lambda x:(x[cols] * x['counts']).sum())
    for col in cols:
        new_composition[col] *= df1['weight'] 
    new_composition[level] = df1[level]
    res = new_composition.groupby(level).sum()
    return res

