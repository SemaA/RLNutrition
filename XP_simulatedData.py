#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate synthetic consumption data.
Given a set of items chosen from a list of items, the script generates a set of
meals (i.e a set of items). 
From a user dataframe, it creates a user profile. 

If you want to change the way the meals are generated, the function "createMeal"
in the script generateSyntheticData must be altered. 

Output : UserDiet object.  

@author: Sema
"""

import pickle
import pandas as pd
from generateSyntheticData import createUserObject, createUserDietObject

#with open('data/adults.p', 'rb') as handle:
#    df_user = pickle.load(handle)
#df_user.reset_index(inplace=True)
#df_user.rename(columns = {'menopaus':'menopause'}, inplace=True)
#columns = ['menopause', 'regmaig', 'regmedic', 'regrelig', 'regvegr', 'regvegt',
#           'ordi', 'bonalim', 'agglo9']
#for col in columns:
#    df_user[col].fillna(0, inplace = True)
#df_user.to_csv('data/socio.csv')



#Import data
socio = pd.read_csv('data/socio.csv', header=0)
adequacyRef = pd.read_csv('data/adequacyReferences.csv', index_col=[0,1], header=0,
                          delimiter=";")
moderationRef = pd.read_csv('data/moderationReferences.csv', index_col=[0,1], header=0,
                          delimiter=";")
with open('data/compoCodsougr.p', 'rb') as handle:
    composition = pickle.load(handle) 

with open('data/dict_codsougrUpdated.p', 'rb') as handle:
    dict_codsougr = pickle.load(handle) 

with open('data/portionsDict.p', 'rb') as handle:
    portions = pickle.load(handle) 


users = createUserObject(socio)
user = users[110007]

itemsName = ['yoghurt', 'bread', 'fromage blanc', 'butter', 'margarine', 
             'poultry rabbit', 'pizza', 'beans', 'potatoes', 'sodas colas']
itemsDict = {v:k for k,v in dict_codsougr.items() if v in itemsName}
minItem = 1
maxItem = 5
nMeal = 21
nLastDays = 7

userDiet = createUserDietObject(user, itemsName, itemsDict, minItem, maxItem, 
                                nMeal, portions, composition, socio, 
                                adequacyRef, moderationRef, nLastDays)

with open('simulatedData/userDiet.p', 'wb') as handle:
    pickle.dump(userDiet, handle)