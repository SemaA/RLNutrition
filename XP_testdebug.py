#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 10:36:00 2019

@author: sema
"""
import logging
logging.basicConfig(filename='logs/XP_testDebug.log', level=logging.DEBUG)


import pickle
import random
import numpy as np
import pandas as pd
from itertools import permutations
from userModel import User, Food, Meal, UserDiet, Substitution
from policies import System
from XP_rl import xp
from XP_data import generateNoisyMatrix, createUserObject, createConsumptionSeq, generateAcceptabilityMatrix


### Import data 
conso = pd.read_csv('data/testConso.txt')
adequacyRef = pd.read_csv('data/adequacyReferences.csv', index_col=[0,1], header=0,
                          delimiter=";")
moderationRef = pd.read_csv('data/moderationReferences.csv', index_col=[0,1], header=0,
                          delimiter=";")

with open('data/compoCodsougr.p', 'rb') as handle:
    composition = pickle.load(handle) 

with open('data/meanQuantitiesDict.p', 'rb') as handle:
    meanQuantities = pickle.load(handle) 
    
with open('data/dictCodsougr.p', 'rb') as handle:
    dict_codsougr = pickle.load(handle) 

with open('data/portions.p', 'rb') as handle:
    portions = pickle.load(handle) 

portions = {i+1:v for i,v in enumerate(portions)}


itemsDict = {v:k for k,v in dict_codsougr.items()}
itemsName = list(dict_codsougr.values())
#itemsName = ['bread', 'coffee', 'yoghurt', 'tea infusion', 'butter', 
#             'jam honey', 'rice', 'beef', 'spring water', 'beer']


socio = pd.read_csv('data/userInfo.txt', header=0, delimiter = ";")
nLastDays = 7
nbItem = len(itemsName)
minItem = 2
maxItem = 4
nMeal = 21
nomen = 1

users = createUserObject(socio)
user = users[1]

#diet = createDiet(itemsName, itemsDict, minItem, maxItem, nMeal, meanQuantities)


diet = createConsumptionSeq(conso)
meals = [x.nameTuple for x in diet if x.nameTuple]
ms = list(set(meals))
A = generateAcceptabilityMatrix(ms, itemsName)


userDiet = UserDiet(nomen, user, diet, nbItem, A, meanQuantities, itemsDict, 
                    portions, composition, socio, adequacyRef, moderationRef, 
                    nLastDays)
p, df = userDiet.computePandietScore(verbose=1)
userDiet.updatePandiet(p)

meal = userDiet.generateNextMeal(jour=1, tyrep=1)
print('---------------------------')
userDiet.addMeal(meal)
p1 = userDiet.computePandietScore()
userDiet.updatePandiet(p1)




#################### Expert initialization
mu = 0
sigma = 0.01
epsilon = 0.5
alpha = 0.2
gamma = 0.4

A0 = generateNoisyMatrix(A, mu, sigma)
actions  = list(permutations(itemsName,2)) 
timestep = 1
substitutabilityScores = {action:np.random.random_sample() for action in actions}


expert = System(A0, actions, epsilon, alpha, gamma, ms, substitutabilityScores)
state = userDiet.getLastMeal()
stateName = state.nameTuple
utilityDistribution = expert.provideSubstitutions(state, userDiet, function='mixed')
chosenAction = expert.selectAction(state, 'epsilonGreedy', utilityDistribution)
acceptanceProb = userDiet.computeAcceptabilityProb(state, chosenAction)


if acceptanceProb == 1:
    prevPandiet = userDiet.computePandietScore()
    x, y, newMeal = userDiet.substituteItemInDiet(state, chosenAction)
    userDiet.replaceLastMeal(newMeal, verbose=1)
    p = userDiet.computePandietScore()
    deltaPandiet = prevPandiet.pandiet - p.pandiet
    
    subScore = expert.Q[chosenAction].loc[[stateName]].values[0]
    
    s = Substitution(x, y, subScore, p.pandiet, deltaPandiet, timestep)
    userDiet.addToSubstitutionTrack(s)
    
    rewardUser = deltaPandiet
    rewardExpert = 1 
    prevMeal = userDiet.getLastMeal()
    jour = prevMeal.jour
    tyrep = prevMeal.tyrep 
    
    newTyrep = 1 + tyrep%3
    if newTyrep == 1:
        newJour = jour + 1
    else:
        newJour = jour
    nextS = userDiet.generateNextMeal(jour=newJour, tyrep=newTyrep)
    nextS.reorder()
    
    nextSname = nextS.nameTuple
            
    nextUtilityDistribution = expert.provideSubstitutions(nextS,userDiet, function='mixed')
    nextChosenAction = expert.selectAction(nextS, 
                                   'greedy', 
                                   nextUtilityDistribution)
    maxNextValue = expert.Q[nextChosenAction].loc[[nextSname]].values[0]
    expert.qlearningUpdate(state, chosenAction, rewardExpert, 
                                   maxNextValue)
    # No user update
    #user.updateMealDistribution(lastMeal, mealAfterInteraction, reward)