#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 16:02:39 2019

@author: sema
"""
import pandas as pd

from userModel import Food, Meal, UserDiet
from policies import System
from nutritionalScore import computePandiet

# Import 
meanQuantities = importPickle('/data/meanQuantitiesDict.p')



nLastDays = 7

# Acceptability matrix generation





def XP(userDiet, nStep, nLastDays, composition, socio, ref, modref, epsilon, 
       alpha, gamma):
    """
    Perform the experience of RL for coaching. 
    """
    
    #Generate A0 with noise of A
    #A0 = 
    
    expert = System(A0, action, epsilon alpha, gamma)
    userDiet.pandiet0 = userDiet.computePandiet(nLastDays, composition, socio, ref, modref)
    
    for i in range(nStep):
        userDiet.timestep += 1
        meal = userDiet.generateNextMeal()
        userDiet.addMeal(meal, composition, socio, ref, modref, nLastDays)
        currentPandiet = userDiet.currentPandiet
        
        #Policy chooses action
        epsilonGreedyAction = expert.selectAction(meal, 'epsilonGreedy')
        # Compute probability of acceptance of the substitution 
        # action = tuple ('name1', 'name2')
        acceptanceProb = userDiet.computeAcceptabilityProb(epsilonGreedyAction)
        
        # Compute the reward (negative reward)
        if acceptanceProb == 1:
            
            toSubstitute, substitutedBy = userDiet.substituteItemInDiet(epsilonGreedyAction, portions)
            accepScore = userDiet.A.at[meal, action]
            
            curPandiet, deltaPandiet = userDiet.computeDeltaPandietAfterSub(composition, socio, 
                                                          ref, modref, nLastDays)
            reward = deltaPandiet * accepScore
            
            s = Substitution(toSubstitute, substitutedBy, subScore, curPandiet, deltaPandiet)
            userDiet.addToSubstitutionTrack(s)
            
        else:
            reward = 0
        
        expert.rewards{timestep} = reward
        # Update user and expert values
        
        maxNextQvalue = userDiet.generateNextMeal()
        
        
        userDiet.updateMealDistribution(meal, mealAfterInteraction)
        expert.qlearningUpdate(meal, action, reward, maxNextQvalue)
        expert.updateAcceptability(meal, action, acceptanceProb, maxNextQvalue)