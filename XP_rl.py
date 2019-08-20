#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Sema
"""
import datetime
date = datetime.datetime.now().strftime('%y_%m_%d_%H_%M')
import logging
import pickle
import random
import numpy as np
import pandas as pd
from itertools import permutations
from userModel import User, Food, Meal, UserDiet, Substitution
from policies import System


def xp(A0, actions, epsilon, alpha, gamma, subScores,
       nomen, user, diet, nbItem, A, meanQuantities, itemsDict, 
                    portions, composition, socio, adequacyRef, moderationRef, 
                    nLastDays, T, filename):
    
    #User initialization 
    logging.info('-------------------------------------------------------')
    logging.info('Initialization of user...')
    userDiet = UserDiet(nomen, user, diet, nbItem, A, meanQuantities, itemsDict, 
                    portions, composition, socio, adequacyRef, moderationRef, 
                    nLastDays)
    logging.debug('Setting the initial PandietScore...')
    p = userDiet.computePandietScore()
    userDiet.updatePandiet(p)
    
    #Expert initialization
    logging.info('-------------------------------------------------------')

    logging.info('Initialization of expert...')
    meals = [x.nameTuple for x in diet if x.nameTuple]
    ms = list(set(meals))
    expert = System(A0, actions, epsilon, alpha, gamma, ms, subScores)
    
    
    # Learning phase
    logging.info('-------------------------------------------------------')
    logging.info('Learning phase...')
    userDiet, expert, pandietRewards = learningPhase(userDiet, expert, T, utility='mixed')
    with open('res2.p', 'wb') as handle:
        pickle.dump({'user':userDiet, 'expert':expert, 
                     'pandietRewards':pandietRewards}, handle)
    return userDiet, expert, pandietRewards


def learningPhase(userDiet, expert, T, saveFileName, utility='mixed'):
    
    pandietRewards = {}
    
    lastMeal = userDiet.getLastMeal()
    jour = lastMeal.jour
    tyrep = lastMeal.tyrep
    state = userDiet.generateNextMeal(jour=jour, tyrep=tyrep)
    print('The first state is {}'.format(state.nameTuple))
    userDiet.addMeal(state)
#    p1 = userDiet.computePandietScore()
#    userDiet.updatePandiet(p1)

    for timestep in range(1,T):
        userDiet.timestep = timestep
        
        logging.info('LEARNING TIMESTEP {}'.format(timestep))
        logging.info('State : '.format(state))
        distributionDict = expert.provideSubstitutions(state, userDiet, 
                                                       function=utility)
        utilityDistribution = distributionDict[utility]
        baselineDistribution = distributionDict['baseline']
        chosenAction = expert.selectAction(state, 'epsilonGreedy', 
                                           utilityDistribution)
        baselineAction = expert.selectAction(state, 'greedy', baselineDistribution)
        
        logging.info('Current state: {}, chosen action: {}'.format(state.nameTuple, chosenAction))
        acceptanceProb = userDiet.computeAcceptabilityProb(state, chosenAction)
        baselineAcceptance = userDiet.computeAcceptabilityProb(state, baselineAction)
        logging.info('Action {}, probability of acceptance: {}'.format(chosenAction, acceptanceProb))
        logging.info('Baseline action {}, probability of acceptance: {}'.format(baselineAction, 
                     baselineAcceptance))
        
        # Track 
        expert.trackGreedyBaselineResults(timestep, baselineDistribution, 
                                          baselineAction, baselineAcceptance)

        # Generating the next meal
        prevMeal = userDiet.getLastMeal()
        jour = prevMeal.jour
        tyrep = prevMeal.tyrep 
        
        newTyrep = 1 + tyrep%3
        if newTyrep == 1:
            newJour = jour + 1
        else:
            newJour = jour
        nextS = userDiet.generateNextMeal(jour=newJour, tyrep=newTyrep)
        logging.info('The next state is: {}'.format(nextS.nameTuple))
        
        # Create the substitution object
        # 
        # s = Substitution(x, y, subScore, p.pandiet, deltaPandiet, timestep)
        
        prevPandiet = userDiet.computePandietScore()

        if acceptanceProb == 1:
            stateName = state.nameTuple
            x, y, newMeal = userDiet.substituteItemInDiet(state, chosenAction)
            userDiet.replaceLastMeal(newMeal, verbose=1)
            nextPandiet = userDiet.computePandietScore()
            deltaPandiet = prevPandiet.pandiet - nextPandiet.pandiet
                        
            subScore = expert.Q[chosenAction].loc[[stateName]].values[0]
            
            s = Substitution(x, y, subScore, nextPandiet.pandiet, deltaPandiet, timestep)
            userDiet.addToSubstitutionTrack(s)
            
            rewardUser = deltaPandiet
            rewardExpert = 1 
            
            pandietRewards[timestep] = deltaPandiet

            # Updates user and coach models
            nextSname = nextS.nameTuple
            
            nextDistributionDict = expert.provideSubstitutions(nextS, 
                                                          userDiet, 
                                                          function=utility)
            nextUtilityDistribution = nextDistributionDict[utility]
            nextChosenAction = expert.selectAction(state, 
                                           'greedy', 
                                           nextUtilityDistribution)
            
            maxNextValue = expert.Q[nextChosenAction].loc[[nextSname]].values[0]
            expert.qlearningUpdate(state, chosenAction, rewardExpert, maxNextValue)
            userDiet.updateMealDistribution(stateName, newMeal.nameTuple, rewardUser)
            logging.info('Timestep {0}, accepted action {1} with PandietGain {2:.4f} '.format(timestep, 
                  chosenAction, deltaPandiet))
        else:
            logging.info('Timestep {0}, refused action {1} '.format(timestep, chosenAction))
            nextPandiet = prevPandiet
        state = nextS
        userDiet.updatePandiet(timestep, nextPandiet)


    logging.info('-------------------------------------------------------')
    with open(saveFileName, 'wb') as handle:
        res = {'userDiet':userDiet, 'expert':expert, 'rewards':pandietRewards}
        pickle.dump(res, handle)
    return userDiet, expert, pandietRewards