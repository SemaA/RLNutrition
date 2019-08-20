#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pandiet computation
How to use it :
    From consumption database, get all the consumptions of a user and then 
    compute the pandiet score. 
"""
import numpy as np
import pandas as pd
from scipy.stats import norm


def getConsoUser(df, user_id):
    """
    Get all the consumptions of a user given his user id. 
    
    Input:
        df(pd.DataFrame) multindex
        user_id (int)
    Output:
        pd.DataFrame with multindex
    """
    u_id = int(user_id)
    if isinstance(df.index, pd.core.index.MultiIndex):
        return df.xs(u_id, level='nomen')
    else:
        return df.loc[df.nomen == u_id]

def addNutrientValues(df, composition, df_socio, level='codsougr'):
    """
    Compute the nurtitional intake of each food item with the nutritional 
    composition table and the quantity. 
    
    Input:
        df(pd.DataFrame) multindex, consumption df
        composition (df.DataFrame)
    
    Output:
        df (df.DataFrame) with nutrient columns in addition
        
    """
    cols = composition.columns.tolist()
    if isinstance(df.index, pd.core.index.MultiIndex):
        df.reset_index(inplace=True)
    df1 = pd.merge(df, composition, on=level)
    #df1.update(df1[cols].mul(df1.qte_nette,0))
    for col in cols:
        df1[col] *= df1['qte_nette'] 
    df1[cols] *= 0.01
    
    ## Add Nutrient Value By User Weight for proteines and mg
    cols_ = ['mg','mg_dis','proteines_d','proteines_N_d']
    cols_w = [col+'_kg' for col in cols_]
    df1 = pd.merge(df1, df_socio[['nomen', 'poidsm']], on='nomen')
    df1[cols_w] = df1[cols_].copy()
    for col in cols_w:
        df1[col] /= df1['poidsm'] 
    cols += cols_w
    return df1, cols

def sumDailyIntakes(df, cols):
    """
    Sum intakes into daily intakes per user. 
    
    Input:
        df(pd.DataFrame) multindex, consumption df
    
    Output:
        df_ (df.DataFrame) daily intakes per user (nb user x nb days rows)
    """
    df_ = df.groupby(['nomen', 'jour'])[cols].sum()  
    return df_

def convertInNeedUnit(df):
    """
    Convert several nutrients in need units after summing over the day. 
    """
    df['energie_woa'] = df['energie'] - df['alcool'] * 7
    df['glut_ei'] = df['glucides'] * 400 / df['energie_woa']
    df['lip_ei'] = df['lipides'] * 900 / df['energie_woa']
    df['LAom6_ei'] = df['LAom6'] * 900 / df['energie_woa']
    df['ALAom3_ei'] = df['ALAom3'] * 900 / df['energie_woa']
    df['ags_ei'] = df['ags'] * 900 / df['energie_woa']
    df['agp_ei'] = df['agp'] * 900 / df['energie_woa']
    df['vitb1_ei'] = df['vitb1'] / df['energie_woa'] * 1000
    df['vitb2_ei'] = df['vitb2'] / df['energie_woa'] * 1000
    df['vitb3_ei'] = df['vitb3_en'] / df['energie_woa'] * 1000
    return df

def absorb_nh(row):
    if (row['fer_nonhem'] != 0) & (row['Ca'] != 0) & (row['Phytates'] != 0) & (row['vitc'] != 0):
        log_val = 6.294 - 0.709 * np.log(15) + 0.119 * np.log(row['vitc']) + 0.006 * np.log(row['MFP'] + 0.1) - 0.055 * np.log(row['eq_thenoir'] + 0.1) - 0.247 * np.log(row['Phytates']) - 0.137 * np.log(row['Ca']) - 0.083 * np.log(row['fer_nonhem']) 
        val = np.exp(log_val)
    else:
        val = 0
    return val

def log_fe_biodis(row):
    if (row['fe_biodis'] != 0):
        val = np.log(row['fe_biodis'])
    else:
        val = 0
    return val
        

def biodisponibility(df):
    """
    Code the biodisponibility of iron and zinc.
    (cf Armah et al 2013 et Miller et al 2007)
    """
    #For zn
    df['zn_mol'] = df['Zn'] / 65.38
    df['phytates_mol'] = df['Phytates'] / 660.04 
    df['zn_biodis'] = 0.5 * (0.13 + df['zn_mol'] + 0.10*(1+ df['phytates_mol']/1.2)-
      np.sqrt((0.13+df['zn_mol']+0.10*(1+df['phytates_mol']/1.2))**2 - 4*0.13*df['zn_mol']))
    df['zn_biodis'] *= 65.38
    df['Zn_abs'] = df['zn_biodis'] *100 / df['Zn']
    
    df['absorb_nh']	 = df.apply(absorb_nh, axis=1)
    df['log10_absorb_h'] = 1.9897 - 0.3092 * np.log10(15)
    df['absorb_h'] = 10**(df['log10_absorb_h'])
    df['fe_biodis'] = df['absorb_nh']/100 * df['fer_nonhem'] + df['absorb_h']/100 * df['fer_hem']
    df['fe_abs'] = df['fe_biodis'] / df['Fe'] *100
    df['log_fe_biodis']	 = df.apply(log_fe_biodis, axis=1)
    
    df['K_mol']= df['k'] / 39.1
    df['Na_mol']= df['na'] / 23
    df['P_mol']= df['P'] / 30.97
    df['Ca_mol']= df['Ca'] / 40.08
    return df


def averageDay(df, cols): 
    """
    """
    new_cols = ['ALAom3_ei','Ca_mol','K_mol','LAom6_ei','Na_mol', 'P_mol',
                'Zn_abs','absorb_h','absorb_nh','agp_ei','ags_ei','energie_woa',
                'fe_abs','fe_biodis','glut_ei','lip_ei','log10_absorb_h',
                'log_fe_biodis','phytates_mol','vitb1_ei','vitb2_ei',
                'vitb3_ei','zn_biodis','zn_mol']
    cols += new_cols
    df_ = df.groupby(['nomen'])[cols].mean() 
    std = df.groupby(['nomen'])[cols].std() 
    std.rename(columns=lambda x:x+'_std', inplace =True)
    df__ = df_.merge(std, left_index=True, right_index=True)
    return df__


######## Pandiet scores
def solve(m1,std1,m2,std2):
    """
    Computes the intersection of two gaussians. 
    """
    if m1 > m2:
        min_m = m2
        min_m_std = std2
        max_m = m1
        max_m_std = std1
    else:
        min_m = m1
        min_m_std = std1
        max_m = m2
        max_m_std = std2
        
    a = 1/(2*min_m_std**2) - 1/(2*max_m_std**2)
    b = max_m/(max_m_std**2) - min_m/(min_m_std**2)
    c = min_m**2 /(2*min_m_std**2) - max_m**2 / (2*max_m_std**2) - np.log(max_m_std/min_m_std)
    r = np.roots([a,b,c])[0] 
    area = norm.cdf(r,max_m,max_m_std) + (1.-norm.cdf(r,min_m,min_m_std))
    return area

def probnorm(m1, std1, m2, std2, nj):
    """
    Compute the equivalent of probnorm in SAS.
    """
    x = (m1-m2) / np.sqrt(std1**2/nj + std2**2)
    return norm(0, 1).cdf(x)

def adequacy(col, ref):
    m = ref.loc[(col, 0), 'mean']
    std = ref.loc[(col,0), 'std']
    return m, std*m

def adequacyBySex(row, col, ref):
    """
    Selects the arguments for the computation based on the sex of the user
            
    ref (pd.DataFrame) of two columns ['mean', 'std']
    """
    if row['sexe_ps'] == 1:
        m = ref.loc[(col, 1), 'mean']
        std = ref.loc[(col, 1), 'std']
    else:
        m = ref.loc[(col, 2), 'mean']
        std = ref.loc[(col, 2), 'std']
    return m, std*m

def adequacyCa(row, col, ref):
    """
    Selects the values for the computation of the adequacy of calcium.
    """
    if row['v2_age'] <= 24:
        m = ref.loc[(col, 24), 'mean']
        std = ref.loc[(col, 24), 'std']
    else:
        m = ref.loc[(col, 25), 'mean']
        std = ref.loc[(col, 25), 'std']
    return m, std*m

def adequacyFe(row, col, ref, row2):
    """
    Selects the values (mean, std) for the compuration of the adequacy of iron
    """
    if (row['sexe_ps'] == 2) & (row['menopause'] == 2):
        m = ref.loc[(col, 2), 'mean']
        std = ref.loc[(col,2), 'std']
        m1 = row2['log_fe_biodis']
        std1 = row2['log_fe_biodis_std']
        return m1, std1, m, std
    elif (row['sexe_ps'] == 1) | (row['menopause'] == 1):
        m = ref.loc[(col, 1), 'mean']
        std = ref.loc[(col,1), 'std']
        m1 = row2['fe_biodis']
        std1 = row2['fe_biodis_std']
        return m1, std1, m, std*m

def adequacyZn(row, col, ref):
    m = ref.loc[col, 'mean']
    std = ref.loc[col, 'std']
    m += 0.038*row['poidsm']
    return m, m*std

def moderation(row, col, ref):
    """
    Selects the arguments for the computation of the moderation score. 
    """
    if 0 in ref.loc[col].index: # no condition for the computation
        m = ref.loc[(col, 0), 'mean']
        std = ref.loc[(col,0), 'std']
    else:
        # For computation based on sex
        if row['sexe_ps'] == 1:
            m = ref.loc[(col, 1), 'mean']
            std = ref.loc[(col,1), 'std']
        else:
            m = ref.loc[(col, 2), 'mean']
            std = ref.loc[(col,2), 'std']
    return m, std*m

def penalty(col,ref):
    return ref.loc[col, 'pen']


### Scores    
def computeAdequacyScore(df_, socio, ref, nj):
    """
    Compute the adequacy score. 
    """
    df = pd.DataFrame(index=df_.index)
    
    # For unchanged nutrients
    u_nutrients = ['proteines_N_d_kg', 'LAom6_ei', 'ALAom3_ei', 'om3dha', 'epadha',
                   'fibres', 'vitb1_ei', 'vitb2', 'vitb3_ei', 'vitb9', 'vitb12',
                   'vitc', 'vitd', 'Iode', 'mg_kg', 'k', 'Se']
    
    #print('Computing the adequacy score for unchanged nutrients')
    for col in u_nutrients:
        if col in df_.columns: 
            new_col = 'P'+ col
            std_col = col + '_std'
            #df[new_col] = np.vectorize(probnorm)(df_[col], df_[std_col], ref.loc[col], default_std, nj)
            
            df[new_col] = df_.apply(lambda x: probnorm(x[col], x[std_col], 
              *adequacy(col, ref), nj = nj), axis=1)
        else:
            print("{} is not in the table".format(col))
    
    # For unchanged nutrients and specific computation
    #nutrients = ['P', 'zn_biodis']
    df['PP'] = df_.apply(lambda x: probnorm(x['P_mol'], 
      (x['P_mol_std'] + x['Ca_mol_std'])/nj,
      x['Ca_mol']/1.65, 0.075*x['Ca_mol']/1.65, nj = nj), axis=1)
    

    df['Pzn_biodis'] = df_.apply(lambda x: probnorm(x["zn_biodis"], x['zn_biodis_std'], 
              *adequacyZn(socio.loc[x.name], 'zn_biodis', ref), nj = nj), axis=1)

    # For nurtrients with different requirements by sex
    s_nutrients = ['vita', 'vitb5', 'vitb6', 'vite', 'Cu', 'Mn']
    #print('Computing the adequacy score for nutrients for different requirements by sex...')
    for col in s_nutrients:
        if col in df_.columns:
            new_col = 'P'+ col
            std_col = col + '_std'
            df[new_col] = df_.apply(lambda x: probnorm(x[col], x[std_col], 
              *adequacyBySex(socio.loc[x.name], col, ref), nj = nj), axis=1)
    
    # For calcium
    #print('Computing the adequacy score for Ca...')
    df['PCa'] = df_.apply(lambda x: probnorm(x['Ca'], x['Ca_std'], 
      *adequacyCa(socio.loc[x.name], 'Ca', ref), nj = nj), axis=1)
    
    # For iron 
    #print('Computing the adequacy score for Fe...')
    df['PFe'] = df_.apply(lambda x: probnorm(*adequacyFe(socio.loc[x.name], 
      'Fe', ref, x), nj = nj), axis=1)
    
    # Additional operations before the summation
    df['Pepadha'] += df['Pom3dha']
    df['Pom3dha'] *= 0.5
    df['Pepadha'] *= 0.5 
    
    # Final mean 
    df['adequacy_score'] = df.sum(axis=1) *100 /26
    return df


def computeModerationScore(df_, socio, ref, nj):
    """
    Computes the moderation score
    """
    df = pd.DataFrame(index = df_.index)
    m_nutrients = ['proteines_N_d_kg', 'glut_ei', 'lip_ei', 'ags_ei', 'glus_slac', 'na']
    
    for col in m_nutrients:
        if col in df_.columns:
            new_col = 'P'+ col
            std_col = col + '_std'
            df[new_col] = df_.apply(lambda x: 1 - probnorm(x[col], x[std_col], 
              *moderation(socio.loc[x.name], col, ref), nj=nj), axis=1)
    df['moderation_score'] = df.sum(axis=1) *100 / 6
    return df


def computePenalty(df_, ref):
    """
    """
    p_nutrients = ['ret', 'vitb6', 'vitb9', 'vitd', 'vite', 'Ca', 'Cu', 'Iode', 'mg_dis',
                   'Zn', 'Se']
    df_['penalty'] = 0
    
    for col in p_nutrients:
        if col in df_.columns:
            df_['penalty'] += df_.apply(lambda x:1 if x[col] > penalty(col, ref) else 0, axis=1)
        else:
            print('Not in index')
    #df_['penalty'] += 6
    return df_
            
            

def computePandiet(df, composition, socio, ref, modref, nj, level='codsougr'):
    """
    Computes the Pandiet score :
        pandiet = (adequacy + moderation) / 2 
    
    Input : 
        df (pd.DataFrame) consumption database
        composition (pd.DataFrame) 
        socio (pd.DataFrame) 
        ref (pd.DataFrame) nutritional adequacy references for nutriments
        modref (pd.DataFrame) : nutritional moderation references
    
    Output:
        df_pandiet (pd.DataFrame) : 
    """
    df2, cols = addNutrientValues(df, composition, socio, level)
    #print('Summing for daily intakes')
    df3 = sumDailyIntakes(df2, cols)
    #print('Converting to need unit')
    df4 = convertInNeedUnit(df3)
    #print('Computing biodisponibility')
    df4 = biodisponibility(df4)
    #print('Averaging to day')
    df5 = averageDay(df4, cols)
    s = socio.set_index('nomen')
    #print('Computing the adequacy score...')
    adequacy = computeAdequacyScore(df5, s, ref, nj)
    #print('Computing the moderation score...')
    moderation = computeModerationScore(df5, s, modref, nj)
    #print('Computing the pandiet...')
    #penal = computePenalty(df5,penref)
    diet = (adequacy.adequacy_score + moderation.moderation_score) / 2
    #print('Finished')
    return diet, adequacy, moderation

  
#def main(df, compo, socio, ref, modref):
#    nj = 7
#    print('Adding nutrient values')
#    df2, cols = addNutrientValues(df, compo, socio)
#    print('Summing for daily intakes')
#    df3 = sumDailyIntakes(df2, cols)
#    print('Converting to need unit')
#    df4 = convertInNeedUnit(df3)
#    print('Computing biodisponibility')
#    df4 = biodisponibility(df4)
#    print('Averaging to day')
#    df5= averageDay(df4, cols)
#    s = socio.set_index('nomen')
#    print('Computing the adequacy score...')
#    adequacy = computeAdequacyScore(df5, s, ref, nj)
#    print('Computing the moderation score...')
#    moderation = computeModerationScore(df5, s, modref, nj)
#    print('Computing the pandiet...')
#    diet = (adequacy + moderation) / 2
#    print('Finished')
#    return df2, df3, df4, df5, adequacy, moderation, diet
#
#df2, df3, df4, df5, df6, df7, diet = main(c, compo, socio, nut_ref, nutRefSpe,
#                                    moderationRef)


#a_cols = ['PA1protkg', 'PALAom6_ei', 'PAALAom3_ei', 'PAom3dha', 'PAepadha', 
#          'PAfib', 'PAvita', 'PAvitb1','PAvitb2','PAvitb3','PAvitb5','PAvitb6',
#          'PAvitb9','PAvitb12', 'PAvitc', 'PAvitd', 'PAvite', 'PAca', 'PAcu', 
#          'PAfe', 'PAiode', 'PAmgkg', 'PAmn', 'PA1p', 'PAk', 'PAse', 'PAzn']
