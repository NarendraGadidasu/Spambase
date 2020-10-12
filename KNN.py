# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 15:42:58 2017

@author: JTay
"""

import numpy as np
import sklearn.model_selection as ms
from sklearn.neighbors import KNeighborsClassifier as knnC
import pandas as pd
from helpers import  basicResults,makeTimingCurve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel



spam = pd.read_hdf('spambase.hdf','spam')        
spamX = spam.drop('is_spam',1).copy().values
spamY = spam['is_spam'].copy().values

#madelon = pd.read_hdf('datasets.hdf','madelon')        
#madelonX = madelon.drop('Class',1).copy().values
#madelonY = madelon['Class'].copy().values



spam_trgX, spam_tstX, spam_trgY, spam_tstY = ms.train_test_split(spamX, spamY, test_size=0.3, random_state=0,stratify=spamY)     
#madelon_trgX, madelon_tstX, madelon_trgY, madelon_tstY = ms.train_test_split(madelonX, madelonY, test_size=0.3, random_state=0,stratify=madelonY)     


d = spamX.shape[1]
hiddens_spam = [(h,)*l for l in [1,2,3] for h in [d,d//2,d*2]]
alphas = [10**-x for x in np.arange(1,9.01,1/2)]
#d = madelonX.shape[1]
#hiddens_madelon = [(h,)*l for l in [1,2,3] for h in [d,d//2,d*2]]


pipeM = Pipeline([('Scale',StandardScaler()),
                 ('Cull1',SelectFromModel(RandomForestClassifier(),threshold='median')),
                 ('Cull2',SelectFromModel(RandomForestClassifier(),threshold='median')),
                 ('Cull3',SelectFromModel(RandomForestClassifier(),threshold='median')),
                 ('Cull4',SelectFromModel(RandomForestClassifier(),threshold='median')),
                 ('KNN',knnC())])  

pipeS = Pipeline([('Scale',StandardScaler()),                
                 ('KNN',knnC())])  



#params_madelon= {'KNN__metric':['manhattan','euclidean','chebyshev'],'KNN__n_neighbors':np.arange(1,51,3),'KNN__weights':['uniform','distance']}
params_spam = {'KNN__metric':['manhattan','euclidean','chebyshev'],'KNN__n_neighbors':np.arange(1,51,3),'KNN__weights':['uniform','distance']}

#madelon_clf = basicResults(pipeM,madelon_trgX,madelon_trgY,madelon_tstX,madelon_tstY,params_madelon,'KNN','madelon')        
spam_clf = basicResults(pipeS,spam_trgX,spam_trgY,spam_tstX,spam_tstY,params_spam,'KNN','spam')        


#madelon_final_params={'KNN__n_neighbors': 43, 'KNN__weights': 'uniform', 'KNN__p': 1}
#adult_final_params={'KNN__n_neighbors': 142, 'KNN__p': 1, 'KNN__weights': 'uniform'}
#madelon_final_params=madelon_clf.best_params_
spam_final_params=spam_clf.best_params_



#pipeM.set_params(**madelon_final_params)
#makeTimingCurve(madelonX,madelonY,pipeM,'KNN','madelon')
pipeS.set_params(**spam_final_params)
makeTimingCurve(spamX,spamY,pipeS, 'KNN','spam')