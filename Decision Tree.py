# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 11:55:32 2017

Script for full tests, decision tree (pruned)

"""

import sklearn.model_selection as ms
import pandas as pd
from helpers import basicResults,dtclf_pruned,makeTimingCurve
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def DTpruningVSnodes(clf,alphas,trgX,trgY,dataset):
    '''Dump table of pruning alpha vs. # of internal nodes'''
    out = {}
    for a in alphas:
        clf.set_params(**{'DT__alpha':a})
        clf.fit(trgX,trgY)
        out[a]=clf.steps[-1][-1].numNodes()
        print(dataset,a)
    out = pd.Series(out)
    out.index.name='alpha'
    out.name = 'Number of Internal Nodes'
    out.to_csv('./output/DT_{}_nodecounts.csv'.format(dataset))
    
    return  

# Load Data       
spam = pd.read_hdf('spambase.hdf','spam')        
spamX = spam.drop('is_spam',1).copy().values
spamY = spam['is_spam'].copy().values


spam_trgX, spam_tstX, spam_trgY, spam_tstY = ms.train_test_split(spamX, spamY, test_size=0.3, random_state=0, stratify=spamY)     

# Search for good alphas
alphas = [-1,-1e-3,-(1e-3)*10**-0.5, -1e-2, -(1e-2)*10**-0.5,-1e-1,-(1e-1)*10**-0.5, 0, (1e-1)*10**-0.5,1e-1,(1e-2)*10**-0.5,1e-2,(1e-3)*10**-0.5,1e-3]
#alphas=[0]
pipeS_fs = Pipeline([('Scale',StandardScaler()),
                 ('Cull1',SelectFromModel(RandomForestClassifier(random_state=1),threshold='median')),
                 ('Cull2',SelectFromModel(RandomForestClassifier(random_state=2),threshold='median')),
                 ('Cull3',SelectFromModel(RandomForestClassifier(random_state=3),threshold='median')),
                 ('Cull4',SelectFromModel(RandomForestClassifier(random_state=4),threshold='median')),
                 ('DT',dtclf_pruned(random_state=55))])


pipeS = Pipeline([('Scale',StandardScaler()),                 
                 ('DT',dtclf_pruned(random_state=55))])


params = {'DT__criterion':['gini','entropy'],'DT__alpha':alphas,'DT__class_weight':['balanced']}

spam_clf_fs = basicResults(pipeS_fs,spam_trgX,spam_trgY,spam_tstX,spam_tstY,params,'DT','spam_fs')        
spam_clf = basicResults(pipeS,spam_trgX,spam_trgY,spam_tstX,spam_tstY,params,'DT','spam')        


#madelon_final_params = {'DT__alpha': -0.00031622776601683794, 'DT__class_weight': 'balanced', 'DT__criterion': 'entropy'}
#adult_final_params = {'class_weight': 'balanced', 'alpha': 0.0031622776601683794, 'criterion': 'entropy'}
spam_fs_final_params = spam_clf_fs.best_params_
spam_final_params = spam_clf.best_params_

pipeS_fs.set_params(**spam_fs_final_params)
makeTimingCurve(spamX,spamY,pipeS_fs,'DT','spam_fs')
pipeS.set_params(**spam_final_params)
makeTimingCurve(spamX,spamY,pipeS,'DT','spam')


DTpruningVSnodes(pipeS_fs,alphas,spam_trgX,spam_trgY,'spam_fs')
DTpruningVSnodes(pipeS,alphas,spam_trgX,spam_trgY,'spam')