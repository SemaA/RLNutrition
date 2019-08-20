#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Sema
"""
import logging
import numpy as np
import pandas as pd
from copy import deepcopy


from numpy.random import choice
from collections import Counter
import matplotlib.pyplot as plt

from nutritionalScore import computePandiet
#logging.basicConfig(level=logging.DEBUG)


class User(object):
    def __init__(self, nomen, sexe_ps, v2_age, poidsd, poidsm, bmi, 
                 menopause, regmaig, regmedic, regrelig, regvegr, regvegt, 
                 ordi, bonalim, agglo9):
        self.nomen = int(nomen)
        self.sexe_ps = int(sexe_ps)
        self.v2_age = int(v2_age)
        self.poidsd = float(poidsd)
        self.poidsm = float(poidsm)
        self.bmi = float(bmi)
        self.menopause = int(menopause)
        
        self.regmaig = int(regmaig)
        self.regmedic = int(regmedic)
        self.regrelig = int(regrelig)
        self.regvegr = int(regvegr)
        self.regvegt = int(regvegt)
        
        self.ordi = float(ordi)
        self.bonalim = int(bonalim)
        self.agglo9 = int(agglo9)
        logging.debug('User object created: {}, {} years old.'.format(self.nomen,
                      self.v2_age))
        

    def as_dict(self):
        return {'nomen':int(self.nomen), 
                'sexe_ps':int(self.sexe_ps), 
                'v2_age':int(self.v2_age),
                'poidsd':self.poidsd, 
                'poidsm':self.poidsm, 
                'bmi':self.bmi,
                'menopause':int(self.menopause), 
                'regmaig':int(self.regmaig), 
                'regvegr':int(self.regvegr)}

class Food(object):
    def __init__(self, codsougr, codsougr_name, qte_nette):
        self.codsougr = int(codsougr)
        self.codsougr_name = str(codsougr_name)
        self.qte_nette = float(qte_nette)
        
#    def __repr__(self):
#        return 'Item {}, qty {}'.format(self.codsougr_name, self.qte_nette)

class Meal(object):
    """
    A meal is characterized by the set of food items, the type of meal, the day
    the place, the companion, the duration. 
    
    food_set (list) : list of Food objects
    """
    def __init__(self, food_set, jour, tyrep):
        self.meal = food_set
        self.codsougr = [x.codsougr for x in self.meal]
        self.codsougr_name = [x.codsougr_name for x in self.meal]
        self.nameTuple = tuple(sorted(self.codsougr_name))
        self.jour = jour
        self.tyrep = tyrep
        
        logging.debug('Meal object created with items <{}>, jour {}, tyrep {}'.format(self.codsougr_name,
                      self.jour, self.tyrep))
        
    def __str__(self):
        return ", ".join(str(x) for x in self.codsougr_name)
    
    def __repr__(self):
        return ", ".join(str(x) for x in self.codsougr_name)
    
    def findItemIndex(self, value):
        return next(i for i,m in enumerate(self.meal) 
                    if getattr(m, 'codsougr_name') == value)
        
    def reorder(self):
        self.codsougr = [x.codsougr for x in self.meal]
        self.codsougr_name = [x.codsougr_name for x in self.meal]
        self.nameTuple = tuple(sorted(self.codsougr_name))

class MealSequence(object):
    def __init__(self,user,meal_list):
        self.nomen = user.nomen
        self.user = user
        self.meal_list = meal_list
        self.codsougr_name_list = [x.codsougr_name for x in self.meal_list]
    

class Substitution(object):
    """
    Input :
        toSubstitute (int) corresponding to the code at the level.
        substitutedBy (Item object)
    """
    def __init__(self, toSubstitute, substitutedBy, subScore, 
                 pandiet, deltaPandiet, timestep):
        self.toSubstitute = toSubstitute
        self.substitutedBy = substitutedBy
        self.subScore = subScore
        self.pandiet = pandiet
        self.deltaPandiet = deltaPandiet
        self.objectif = subScore * deltaPandiet
        self.timestep = timestep
        logging.debug('Timestep {0} : Substitution object created: {1} --> {2} with deltaPandiet {3:.4f} and subScore {4:.4f}.'.format(self.timestep, 
                      self.toSubstitute.codsougr_name,
                      self.substitutedBy.codsougr_name, 
                      self.deltaPandiet, 
                      self.subScore))
    
    def __str__(self):
        return 'Substitution of {} by {} with deltaPandiet {} and subScore {}'.format(self.toSubstitute,
                               self.substitutedBy, self.deltaPandiet, self.subScore)


class PandietSubScores(object):
    def __init__(self, nomen, pandiet, adequacy, moderation, timestep,
                 penalty=None):
        self.nomen = nomen
        self.pandiet = float(pandiet)
        self.adequacy = adequacy
        self.moderation= moderation
        self.timestep = timestep
        self.penalty = penalty

###############################################################################

class UserDiet(object):
    def __init__(self, user, meal_list, nbItem, A, foodDict, portions, 
                 composition, socio, adequacyRef, moderationRef, nLastDays):
        """
        A : pd.Dataframe
        
        foodDict (dict) {foodName:foodCod}
        
        Input : 
            user (userObject)
            meal_list (list of Meal objects)
            nbItem (int)
            A (pd.DataFrame) row (tuple of meal), col (tuple of action) 
            portions (dict) 
                keys ['men', 'women'] 
                value dict(idItem:qty)
            composition (pd.DataFrame)  row(itemId), col (nutrientName)
            socio (pd.DataFrame) row(nomen), col (attributeName)
            adequacyRef (pd.DataFrame)
            moderationRef (pd.DataFrame)
            nLastDays (int) 
        
        """
        self.nomen = user.nomen
        self.user = user
        self.mealList0 = meal_list
        self.meal_list = meal_list
        self.nbItem = nbItem
        
        self.timestep = 0
        self.foodDict = foodDict
        self.portions = portions
        self.composition = composition
        self.socio = socio
        self.adequacyRef = adequacyRef
        self.moderationRef = moderationRef
        self.nLastDays = nLastDays
        
        if user.sexe_ps == 1:
            self.meanQuantities = portions['men']
        else:
            self.meanQuantities = portions['women']
        
        self.codsougr_name_list = [x.codsougr_name for x in self.meal_list]
        self.substitutions = []
        self.possibleSubstitutions = []
        self.pandiet0 = 0
        self.current_pandiet = 0
        self.pastPandietScores = []
        self.possibleSubstitutions = []
        self.mealsAfterRec = {}
        self.mealsBeforeRec = {}
        
        # Optimal substitutability matrix (row state, col actions)
        self.mealRepertoire = [meal.nameTuple for meal in self.meal_list if meal.nameTuple]
        self.A = A
        self.mealCounts0 = self.getMealCounts() 
        self.mealCounts = self.mealCounts0
        self.V0 = self.getMealDistribution()
        self.V = self.V0
        self.rewards = {}
        self.baselineChoices = {}

        logging.debug('UserDiet object created: {}.'.format(self.nomen))
    
    def length(self):
        return sum([len(m.meal) for m in self.meal_list])
    
    def getConsoFromSequence(self, n_meals):
        """
        From a MealSequence object, get a pandas dataframe for the computation 
        of pandiet and the User object.
        
        Input :
            seq (MealSequence object)
        Output:
            df (pd.DataFrame) 
            user (User object)
        """
        L = []
        cols = ['nomen','jour', 'codsougr','qte_nette']
        
        nomen = self.nomen
        for m in self.meal_list[-n_meals:]:
            jour = m.jour
            for food in m.meal:
                codsougr = food.codsougr
                qte_nette = food.qte_nette
                L.append([nomen, jour, codsougr, qte_nette])
        df = pd.DataFrame(L, columns=cols)
        df_user = pd.Series(self.user.as_dict()).to_frame().T
        logging.debug('Transformed MealSequence object to dataframe by considering the last {} meals.'.format(n_meals))
        return df, df_user
    
    def computePandietScore(self, verbose=0):
        """
        Compute the pandiet score given the last nLastDays.
        """
        nLastMeals = self.nLastDays * 3
        df, df_user = self.getConsoFromSequence(nLastMeals)
        
        pandietScore, adequacy, moderation = computePandiet(df, self.composition, 
                                                            self.socio, 
                                                            self.adequacyRef, 
                                                            self.moderationRef, 
                                                            self.nLastDays)
        
        p = PandietSubScores(nomen = self.nomen, 
                                 pandiet = pandietScore.values[0],
                                 adequacy = adequacy,
                                 moderation = moderation, 
                                 timestep = self.timestep)
        logging.debug('Timestep {0}: computed PandietScore = {1:.4f}.'.format(self.timestep,
                      pandietScore.values[0]))
        if verbose == 1:
            return p, df
        else:
            return p
        
    
    def updatePandiet(self, timestep, pandietObject):
        """
        Set the current pandiet score. 
        """
        previousPandiet = self.current_pandiet
        self.current_pandiet = pandietObject.pandiet
        self.pastPandietScores.append((timestep, pandietObject))
        logging.info('Updating PandietScore...')
        logging.info('Previous PandietScore: {0:.4f}. Current pandietScore {1:.4f}'.format(previousPandiet, self.current_pandiet))  
    
    def getMealCounts(self, update=0):
        """
        Count the meals occured in the mealList.
        If update = 0, this means that the user does not update his behavior. 
        He keeps generating his meals according to the meals he did at the 
        beginning of the learning phase. 
        
        If update = 1, the user incorporates new meals to his repertoire. 
        """
        if update == 0:
            counts = Counter([meal.nameTuple for meal in self.mealList0])
        else:
            counts = Counter([meal.nameTuple for meal in self.meal_list])
        # Drop empty meals 
        del counts[()]
        return counts
        
    
    def getMealDistribution(self):
        """
        Compute the distribution over meals (with frequencies)
        """
        total = float(sum(self.mealCounts.values()))
        mealDistribution = {key:value/total for key, value in self.mealCounts.items()}
        return mealDistribution
    
    
    def generateNextMeal(self, jour, tyrep):
        """
        Generate next meal from the meal distribution.
        Once the name of the items determined, we have to compute a quantity 
        for each item. 
        """
        nextMeal = choice(list(self.V.keys()), 1, p=list(self.V.values()))[0]
        logging.debug('{}'.format(nextMeal))
        nextMealValues = [x for item in nextMeal for x in self.meanQuantities.keys() if x[1] == item]
        mealList = [Food(cod, name, self.meanQuantities[(cod,name)]) for cod, name in nextMealValues]
        logging.debug('{}'.format(nextMealValues))
        meal = Meal(mealList, jour=jour, tyrep=tyrep)
        logging.debug('Generated the next meal: <jour:{}, tyrep {}, {}>'.format(meal.jour, meal.tyrep, 
                      meal.nameTuple))
        return meal
    
    def addMeal(self, mealToAdd):
        """
        Add next meal to sequence, compute the current pandiet.
        """
        self.meal_list.append(mealToAdd)
        logging.debug('Added meal {}'.format(mealToAdd.nameTuple))
    
    def trackMeal(self, mealBeforeRec, mealAfterRec):
        """
        Add meal before recommendation and meal after recommendation.
        """
        self.mealsBeforeRec.append(mealBeforeRec)
        self.mealsAfterRec.append(mealAfterRec)
        logging.debug('Meal before recommendation {}'.format(mealBeforeRec.nameTuple))
        logging.debug('Meal after recommendation {}'.format(mealAfterRec.nameTuple))
        
    
    def replaceLastMeal(self, newMeal, verbose=0):
        """
        After operating substitution, replace the last meal by the new one.
        """
        deletedMeal = self.meal_list.pop(-1)
        self.meal_list.append(newMeal)
        logging.debug('Replaced meal {} by new meal {}'.format(deletedMeal.nameTuple, newMeal.nameTuple))
    
    def replaceLastMealAfterRecommendation(self, newMeal, verbose=0):
        """
        After operating substitution, replace the last meal by the new one.
        """
        deletedMeal = self.newMeals.pop(-1)
        self.newMeals.append(newMeal)
        logging.debug('Replaced meal {} by new meal {}'.format(deletedMeal.nameTuple, newMeal.nameTuple))

    
    def computeAcceptabilityProb(self, state, action):
        """
        Compute the probability of accepting the proposed action by the system.
        
        action (tuple) (a,b) : substitute a by b
        """
        stateName = state.nameTuple
        if stateName in self.A.index:
            acceptabilityProb = self.A[action].loc[[stateName]].values[0]
        else:
            acceptabilityProb = 0
            logging.info('State not in acceptability matrix !')
        logging.debug('Compute the probability of accepting the action {}.'.format(action))
        return np.random.binomial(1, acceptabilityProb)
    
    def baselineNutGreedy(self):
        """
        Select the action following a nutritional score greedy approach.
        """
#    
#    def computeDeltaPandietAfterSub(self):
#        """
#        Compute reward from action.
#        
#        acceptabilityProb * deltaPandiet
#        """
#        p = self.computePandietScore()
#        curPandiet = p.pandiet 
#        pastPandiet= self.current_pandiet
#        deltaPandiet = curPandiet - pastPandiet
#        
#        # Update Pandiet values
#        self.current_pandiet = p.pandiet
#        self.pastPandietScores.append(p)
#        
#
#        return curPandiet, deltaPandiet
    
    
    def updateMealDistribution(self, lastMeal, mealAfterInteraction, reward,
                               updateEq='default'):
        """
        If the mealAfterInteraction did not exist in the repertoire of the user,
        add the new meal. 
        
        Input :
            lastMeal (tuple)
            mealAfterInteraction (tuple)
            reward (float)
            updateEq (str) : 'default' no update to distribution
                            'rlDefault' usual updates in RL
        Output:
            self.V updated
                    
        
        Update the meal distribution with the RL formula. 
        V(S) <- V(S) + alpha * [(R_ss') + V(s')-V(s)]
        """
        s = lastMeal
        s_ = mealAfterInteraction
        
        #Update meal counts after addition of the newMeal in the mealList
        self.mealCounts = self.getMealCounts() 
        self.rewards[self.timestep] = reward
        
        if updateEq == 'default':
            # Simply convert the counts to a distribution
            logging.debug('Updated user model when converting counts to distrib.')
            
            self.V = self.getMealDistribution()
        else: # Update tules
            if s_ not in self.V.keys():
                self.V[s_] = min(self.V.values())
    
    def getLastMeal(self):
        return self.meal_list[-1]
        
    def substituteItemInDiet(self, state, action):
        """
        Operate the substitution on the last meal. 
        
        action (tuple) : (name1, name2)
        portions 
        """
        x_name = action[0]
        y_name = action[1]
#        men_portions = self.portions['men']
#        women_portions = self.portions['women']
        
        
        currentMeal = deepcopy(state)
        toBeSubstitutedIndex = currentMeal.findItemIndex(x_name)
        toBeSubstitutedFood = currentMeal.meal[toBeSubstitutedIndex]
        x_qty = toBeSubstitutedFood.qte_nette
        
        x_cod = self.foodDict[x_name]
        y_cod = self.foodDict[y_name]
        
#        if self.user.sexe_ps == 1: # male user
#            x_portion = men_portions[int(x_cod)]
#            y_portion = men_portions[int(y_cod)]
#        else:
#            x_portion = women_portions[int(x_cod)]
#            y_portion = women_portions[int(y_cod)]
        
        x_portion = self.meanQuantities[(x_cod, x_name)]
        y_portion = self.meanQuantities[(y_cod, y_name)]
        
        # Compute the equivalent portion 
        y_qty = y_portion * x_qty / x_portion
        substituteFood = Food(y_cod, y_name, y_qty)
        logging.debug('Before {0}, {1:.4f}, After {2}, {3:.4f}'.format(x_name, x_qty,y_name, y_qty))
        
        # Operate substitution
        x = deepcopy(toBeSubstitutedFood)
        mealList = deepcopy(currentMeal.meal)
        mealList.pop(toBeSubstitutedIndex)
        mealList.append(substituteFood)
        newMeal = Meal(mealList, currentMeal.jour, currentMeal.tyrep)
        newMeal.reorder()
        logging.debug('From current meal replaced item {} with {}'.format(toBeSubstitutedFood.codsougr_name, 
                      substituteFood.codsougr_name))
        
#        currentMeal.meal.pop(toBeSubstitutedIndex)
#        currentMeal.meal.append(y)
#        currentMeal.reorder()
        return x, substituteFood, newMeal
        
    
    def addToSubstitutionTrack(self, substitutionObject):
        self.substitutions.append(substitutionObject)
    
    def addToPossibleSubstitutions(self, substitutionList):
        self.possibleSubstitutions.append(substitutionList)
    
    def plotPandietScores(self):
        """
        Plot the pandietScore by timestep.
        """
        scores = self.pastPandietScores
        values = [(t,score.pandiet) for t,score in scores]
        plt.plot(*zip(*values))
        plt.title("Evolution of PandietScore")
        plt.xlabel("Timestep")
        plt.ylabel("PandietScore")
        plt.show()
        


#def substituteItemInDiet(df, sexe_ps, women_portions, men_portions, X, y, level):
#    """
#    Substitute the item x in diet by the item y with the equivalent portions.
#    As items are subgroups, we repeat the operation for each aliment in the 
#    subgroup. 
#    
#    Input:
#        df (pd.DataFrame)
#        user (User object) specify if user is mal or female
#        df_portion (pd.DataFrame) maps aliment to average portion
#        X (dict) keys = ["codsougr", "day", "tyrep"] 
#        y (Item object)
#    """
#    df.loc[(df[list(X)] == pd.Series(X)).all(axis=1)] 
#    x = X['codsougr']
#    x_qty = df.loc[(df[list(X)] == pd.Series(X)).all(axis=1)].qte_nette.values[0]
#    y_name = getattr(y, level)
#    
#    y_copy = deepcopy(y)
#    if sexe_ps == 1: # male user
#        x_portion = men_portions[x]
#        y_portion = men_portions[y_name]
#    else:
#        x_portion = women_portions[x]
#        y_portion = women_portions[y_name]
#    
#    # Compute the equivalent portion 
#    y_qty = y_portion * x_qty / x_portion
#    y_copy.qte_nette = y_qty
#    
#    # Operate the substitution
#    df_s = df.copy()
#    df_s.loc[(df_s[list(X)] == pd.Series(X)).all(axis=1), 
#             ['codsougr', 'qte_nette']] =[y_name, y_qty]
#    return df_s, y_copy





def createUserTable(df):
    """
    Read the user table and create the objects.
    
    Input :
        df (pd.DataFrame)
    Output : 
        dict (key nomen, value objects)
    """
    df.menopaus.fillna(0, inplace=True)
    df.regmaig.fillna(0, inplace=True)
    df.regmaig.fillna(0, inplace=True)
    df.regmedic.fillna(0, inplace=True)
    df.regrelig.fillna(0, inplace=True)
    df.regvegr.fillna(0, inplace=True)
    df.regvegt.fillna(0, inplace=True)
    df.bonalim.fillna(0, inplace=True)
    df.agglo9.fillna(0, inplace=True)
    
    U = {}
    for index, row in df.iterrows(): 
        u = User(index, row.sexe_ps, row.v2_age, row.poidsd, row.poidsm, row.bmi, 
             row.menopaus, row.regmaig, row.regmedic, row.regrelig, row.regvegr, 
             row.regvegt, row.ordi, row.bonalim, row.agglo9)
        U[index] = u
    return U

def getConsumptionSequences(df1, users):
    """
    Read the table of consumption and create objects.
    
    Input :
        df (pd.DataFrame) consumption dataframe
        users (dict) of User objects
    Output : 
        I_full 
        U (dict)
    """
    nomen_list = df1.nomen.unique().tolist()
    
    
    I = {n:[] for n in nomen_list} # Contient les séquences de consommation des individus
    M = {} #Contient les aliments d'un repas
    
    nomen = df1.nomen.iloc[0]
    jour = df1.jour.iloc[0]
    tyrep = df1.tyrep.iloc[0]
    
    for index, row in df1.iterrows(): # Scan tous les aliments de consommation
        #print(row.nomen, row.jour, row.tyrep)
        if (row.nomen == nomen) & (row.jour == jour) & (row.tyrep == tyrep):
            name = str(row.nomen) + '_' + str(index)
            M[name] = Food(row.codsougr, row.codsougr_name, row.qte_nette, ) # Crée l'objet FoodItem

        else:
            m = list(set(M.values())) #Set of food items in a meal
            
            meal = Meal(m, jour = jour, tyrep = tyrep)
            #print(meal)

            I[nomen].append(meal)

            M = {}
            nomen = row.nomen
            jour = row.jour
            tyrep = row.tyrep
            
    U = {}
    for u,seq in I.items():
        user = users[u]
        u_seq = MealSequence(user, seq)
        U[u] = u_seq
    
    return U
