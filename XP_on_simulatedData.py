#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiences on simulated data

@author: Sema
"""
#################### Import packages
import datetime
import logging
date = datetime.datetime.now().strftime('%y_%m_%d_%H_%M')
logging.basicConfig(filename='logs/XP1_SIM'+date+'.log', 
                    format = '%(asctime)s %(message)s',
                    level=logging.INFO)

console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
logging.getLogger('').addHandler(console)

import pickle
import random
import numpy as np
from itertools import permutations
from policies import System
from XP_rl import learningPhase
from generateSyntheticData import generateNoisyMatrix

random.seed(10)


#################### Import data 
with open('data/subScoreAllMeals.p', 'rb') as handle:
    subScores = pickle.load(handle) 
    
logging.info('-------------------------------------------------------')
logging.info('Importing the user object...')
with open('simulatedData/userDiet.p', 'rb') as handle:
    userDiet = pickle.load(handle)
logging.debug('Setting the initial PandietScore...')
p = userDiet.computePandietScore()
userDiet.updatePandiet(0,p)

#################### Expert variables
mu = 0
sigma = 0.01
epsilon = 0.5
alpha = 0.2
gamma = 0.4

itemsName = list(userDiet.foodDict.keys())
A0 = generateNoisyMatrix(userDiet.A, mu, sigma)
actions  = list(permutations(itemsName,2)) 
subScores = {action:np.random.random_sample() for action in actions}
T = 500
filename = 'results/simulatedData'+date+'.p'

#################### Expert initialization
logging.info('-------------------------------------------------------')
logging.info('Initialization of expert...')
meals = [m.nameTuple for m in userDiet.meal_list]
ms = list(set(meals))
expert = System(A0, actions, epsilon, alpha, gamma, ms, subScores)

#################### Learning phase
logging.info('-------------------------------------------------------')
logging.info('Learning phase...')
userDiet, expert, pandietRewards = learningPhase(userDiet, 
                                                 expert, 
                                                 T, 
                                                 filename,
                                                 utility='mixed')