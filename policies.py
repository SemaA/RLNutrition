#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 16:02:02 2019

@author: Sema
"""
import logging
import numpy as np
import pandas as pd
from copy import deepcopy
from collections import defaultdict

from userModel import *
from nutritionalScore import *

import matplotlib.pyplot as plt
#logging.basicConfig(level=logging.DEBUG)

#from matplotlib import plot


class System(object):
    def __init__(self, A0, actions, epsilon, alpha, gamma, meals,
                 substitutabilityScores):
        """
        A (pd.DataFrame) acceptability matrix
        actions (list of tuples) (a,b)
        
        Q table (pd.DataFrame) rows - states and col - actions
        substitutabilityScores (dict) key action, value substitutabilityScore
        
        """
        self.Q = pd.DataFrame(np.random.uniform(0,1, (len(meals), len(actions))), 
                              index=meals, columns=actions)
        self.A = A0
        self.actions = actions
        self.substitutabilityScores = substitutabilityScores
        
        
        self.n_arms = len(actions)
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.rewards = {}
        self.n_iteration = 0
        
        self.baselineResults = {}
        logging.debug('Expert object created: {} arms.'.format(self.n_arms))
        
        
    
    # Policies
    def provideSubstitutions(self, state, userModel, function='mixed'):
        """
        Computes the Pandiet gain of choosing the substitution. 
        Returns a dict of key-value (subs, pandietGain*Q-table value)
        
        Only considers substitutions where the first item is in the state.
        
        state (Meal object)
        userModel (UserDiet object)
        function (str) in ['mixed', 'subs', 'pandiet']
        """
        stateName = state.nameTuple
        subsToTest = [s for s in self.actions if (s[0] in stateName and s[1] not in stateName)]

        if stateName not in self.Q.index:# a new meal is considered
            logging.debug('While computing utility values, add a new state...')
            #Insert new row in dataframe for the new meal
            self.Q = self.Q.append(pd.Series(self.substitutabilityScores, name=stateName))
            
        acceptability = self.Q[subsToTest].loc[[stateName]]
        logging.debug('Computing utiliy value for {} actions (with {} function).'.format(len(subsToTest), 
                      function))
        
        # Computation of utility values
        utilityValues = pd.DataFrame(index = acceptability.index,
                                     columns = acceptability.columns)
        previousPandiet = userModel.current_pandiet
        for action in subsToTest:
            userModelCopy = deepcopy(userModel)
            logging.debug('Testing the action {}'.format(action))
            
            x,y, modifiedMeal = userModelCopy.substituteItemInDiet(state, action)
            userModelCopy.replaceLastMeal(modifiedMeal, verbose=1)
            logging.debug('Now current meal is {}'.format(userModelCopy.getLastMeal()))
            p = userModelCopy.computePandietScore()
            utilityValues.at[[stateName], [action]] = previousPandiet - p.pandiet
        
        baselineDist = utilityValues.iloc[0,:].to_dict()
        if function == 'pandiet':
            p = utilityValues.iloc[0,:].to_dict()
        elif function == 'subs':
            p = acceptability.iloc[0,:].to_dict()
        else:
            product = utilityValues * acceptability
            p = product.iloc[0,:].to_dict()
        logging.debug('Successfully computed utility values for actions')
        
        return {function:p, 'baseline':baselineDist}
            
    
    def epsilonGreedy(self, state, armDistribution):
        """
        Choose randomly an arm with probability p or the arm with the highest 
        estimated reward with probability 1-p.
        
        state (Meal object)
        armDistribution (dict object) where the key is the substitution and 
        the value is the utility associated to the substitution.
        """
        p = np.random.rand()
        
        if p > self.epsilon: # Pick the best arm
            logging.debug('Policy: epsilonGreedy, choosing the max value...')
            return max(armDistribution, key=armDistribution.get)
        else:
            # Randomly choose an arm with uniform probability
            arms = list(armDistribution.keys())
            idx = np.random.choice(len(arms), size=1)[0]
            return arms[idx]
    
    def greedy(self, state, armDistribution):
        """
        Pick the best arm regarding the Q-values
        """
        return max(armDistribution, key=armDistribution.get)
    
        
    def selectAction(self, state, policy, armDistribution):
        """
        Select action according to state.
        """
        if policy == 'greedy':
            return self.greedy(state, armDistribution)
        elif policy == 'epsilonGreedy':
            return self.epsilonGreedy(state, armDistribution)
        else:
            logging.debug('Policy name error!')
            print('Enter a valid policy')
    
    def trackGreedyBaselineResults(self, timestep, baselineDistribution, action, 
                                   acceptanceProb):
        """
        Add 
        """
        gain = baselineDistribution[action]
        self.baselineResults[timestep]= {'action':action,'gain': gain, 
                            'acceptanceProb': acceptanceProb}
        logging.info('Baseline nutGreedy: action {}, gain {}, accepted {}'.format(action, gain, acceptanceProb))
        
    
    
    def qlearningUpdate(self, state, action, reward, maxNextQvalue):
        """
        Update Q-table with QLearning equation (TD update rule)
        
        Q(s,a) <- Q(s,a) + alpha [r + gamma max_a' Q(s',a') - Q(s,a)]
        """
        state.reorder()
        stateName = state.nameTuple
        prevVal = self.Q[action].loc[[stateName]].values[0]
        logging.debug(prevVal)
        updateValue = prevVal + self.alpha * (reward + self.gamma * maxNextQvalue - prevVal)
        self.Q.at[state, [action]] = updateValue
        
        logging.debug('Updated Q-value of state {} and action {}'.format(stateName, action))
        logging.debug('Old value : {}, updated value : {}'.format(prevVal, updateValue))
        
        
#    def updateAcceptability(self, state, action, acceptanceProb, maxNextQvalue):
#        """
#        Update the acceptability matrix.
#        """
#        updateValue = self.A.at[state, action] + self.alpha [acceptanceProb + self.gamma * maxNextQvalue - self.A.at[state, action]]
#        self.A.at[state, action] = updateValue
#        print('Updated Acceptability-value of state {} and action {}'.format(state, action))  
        
        
    def plotBaselineGains(self):
        """
        Plot cumulative rewards over time.
        """
        baselineRes = self.baselineResults
        values = [(t,v['gain']) for t,v in baselineRes.items()]
        plt.plot(*zip(*values))
        plt.title("Evolution of PandietScore gains (baseline nutritional greedy)")
        plt.xlabel("Timestep")
        plt.ylabel("PandietScore")
        plt.show()