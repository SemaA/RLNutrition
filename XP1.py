#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 20:10:14 2019

@author: sema
"""
import datetime
import logging
date = datetime.datetime.now().strftime('%y_%m_%d_%H_%M')
logging.basicConfig(filename='logs/XP1_'+date+'.log', 
                    format = '%(asctime)s %(message)s',
                    level=logging.INFO)

console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
logging.getLogger('').addHandler(console)


import pickle
import random
import numpy as np
import pandas as pd
from itertools import permutations
from userModel import User, Food, Meal, UserDiet, Substitution
from policies import System
from XP_rl import xp, learningPhase
from generateSyntheticData import generateNoisyMatrix, createUserObject, createConsumptionSeq, generateAcceptabilityMatrix



#################### Import data
#conso = pd.read_csv('data/testConso.txt')       
adequacyRef = pd.read_csv('data/adequacyReferences.csv', index_col=[0,1], header=0,
                          delimiter=";")
moderationRef = pd.read_csv('data/moderationReferences.csv', index_col=[0,1], header=0,
                          delimiter=";")
with open('data/conso_ad.p', 'rb') as handle:
    consumptions = pickle.load(handle) 
    
with open('data/compoCodsougr.p', 'rb') as handle:
    composition = pickle.load(handle) 

with open('data/meanQuantitiesDict.p', 'rb') as handle:
    meanQuantities = pickle.load(handle) 
    
with open('data/dict_cod.p', 'rb') as handle:
    dict_cod = pickle.load(handle) 

dict_codsougr = dict_cod['codsougr']
#dict_codsougr = {int(k):v for k,v in dict_codsougr.items()}
dict_codsougr.pop('4499')
dict_codsougr['238'] = 'vegetable mix'
with open('data/portionsDict.p', 'rb') as handle:
    portions = pickle.load(handle) 

nomen = 110006
conso = consumptions[consumptions.index.get_level_values('nomen') == nomen]
conso.reset_index(inplace=True)
conso['codsougr_name'] = conso['codsougr'].map(dict_codsougr)
#portions = {i+1:v for i,v in enumerate(portions)}
itemsDict = {v:k for k,v in dict_codsougr.items()}
itemsName = list(dict_codsougr.values())


socio = pd.read_csv('data/userInfo.txt', header=0, delimiter = ";")
users = createUserObject(socio)
user = users[1]
nomen = user.nomen
nLastDays = 7
nbItem = len(itemsName)
nMeal = 21

#################### User initialization
diet = createConsumptionSeq(conso)
meals = [x.nameTuple for x in diet if x.nameTuple]
ms = list(set(meals))
A = generateAcceptabilityMatrix(ms, itemsName)
#################### Expert initialization
mu = 0
sigma = 0.01
epsilon = 0.5
alpha = 0.2
gamma = 0.4

A0 = generateNoisyMatrix(A, mu, sigma)
actions  = list(permutations(itemsName,2)) 
subScores = {action:np.random.random_sample() for action in actions}
T = 1000
filename = 'results/'+date+'.p'

#################### Learning phase 
logging.info('Starting the learning phase...')
#userDiet, expert, pandietRewards = xp(A0, actions, epsilon, alpha, gamma, subScores,
#       nomen, user, diet, nbItem, A, meanQuantities, itemsDict, 
#                    portions, composition, socio, adequacyRef, moderationRef, 
#                    nLastDays, T, filename)

logging.info('-------------------------------------------------------')
logging.info('Initialization of user...')
userDiet = UserDiet(user, diet, nbItem, A, itemsDict, 
                portions, composition, socio, adequacyRef, moderationRef, 
                nLastDays)
logging.debug('Setting the initial PandietScore...')
p = userDiet.computePandietScore()
userDiet.updatePandiet(0, p)

#Expert initialization
logging.info('-------------------------------------------------------')

logging.info('Initialization of expert...')
meals = [x.nameTuple for x in diet if x.nameTuple]
ms = list(set(meals))
expert = System(A0, actions, epsilon, alpha, gamma, ms, subScores)


# Learning phase
logging.info('-------------------------------------------------------')
logging.info('Learning phase...')
userDiet, expert, pandietRewards = learningPhase(userDiet, 
                                                 expert, 
                                                 T, 
                                                 filename,
                                                 utility='mixed')