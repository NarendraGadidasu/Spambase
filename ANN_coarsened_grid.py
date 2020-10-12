
import numpy as np
from sklearn.neural_network import MLPClassifier
import sklearn.model_selection as ms
import pandas as pd
from helpers import  basicResults,makeTimingCurve,iterationLC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

spam = pd.read_hdf('spambase.hdf','spam')        
spamX = spam.drop('is_spam',1).copy().values
spamY = spam['is_spam'].copy().values

spam_trgX, spam_tstX, spam_trgY, spam_tstY = ms.train_test_split(spamX, spamY, test_size=0.3, random_state=0,stratify=spamY)     

pipeS = Pipeline([('Scale',StandardScaler()),
                 ('MLP',MLPClassifier(max_iter=2000,early_stopping=True,random_state=55))])

#pipeS_fs = Pipeline([('Scale',StandardScaler()),
#                 ('Cull1',SelectFromModel(RandomForestClassifier(random_state=1),threshold='median')),
#                 ('Cull2',SelectFromModel(RandomForestClassifier(random_state=2),threshold='median')),
#                 ('Cull3',SelectFromModel(RandomForestClassifier(random_state=3),threshold='median')),
#                 ('Cull4',SelectFromModel(RandomForestClassifier(random_state=4),threshold='median')),
#                 ('MLP',MLPClassifier(max_iter=2000,early_stopping=True,random_state=55))])

d = spamX.shape[1]
hiddens_spam = [(h,)*l for l in [3] for h in [d*2]]
alphas = [10**-x for x in np.arange(-1,8.01,1/2)]

params_spam = {'MLP__activation':['relu'],'MLP__alpha':alphas,'MLP__hidden_layer_sizes':hiddens_spam}

spam_clf = basicResults(pipeS,spam_trgX,spam_trgY,spam_tstX,spam_tstY,params_spam,'ANN','spam')  
#spam_clf_fs = basicResults(pipeS_fs,spam_trgX,spam_trgY,spam_tstX,spam_tstY,params_spam,'ANN','spam_fs')

spam_final_params = spam_clf.best_params_
spam_OF_params =spam_final_params.copy()
spam_OF_params['MLP__alpha'] = 0

#spam_fs_final_params = spam_clf_fs.best_params_
#spam_fs_OF_params =spam_fs_final_params.copy()
#spam_fs_OF_params['MLP__alpha'] = 0

pipeS.set_params(**spam_final_params)
pipeS.set_params(**{'MLP__early_stopping':False})                  
makeTimingCurve(spamX,spamY,pipeS,'ANN','spam')

#pipeS_fs.set_params(**spam_fs_final_params)
#pipeS_fs.set_params(**{'MLP__early_stopping':False})                  
#makeTimingCurve(spamX,spamY,pipeS_fs,'ANN','spam_fs')

pipeS.set_params(**spam_final_params)
pipeS.set_params(**{'MLP__early_stopping':False})                  
iterationLC(pipeS,spam_trgX,spam_trgY,spam_tstX,spam_tstY,{'MLP__max_iter':[2**x for x in range(12)]},'ANN','spam')                

#pipeS_fs.set_params(**spam_fs_final_params)
#pipeS_fs.set_params(**{'MLP__early_stopping':False})                  
#iterationLC(pipeS_fs,spam_trgX,spam_trgY,spam_tstX,spam_tstY,{'MLP__max_iter':[2**x for x in range(12)]+[2100,2200,2300,2400,2500,2600,2700,2800,2900,3000]},'ANN','spam_fs')

pipeS.set_params(**spam_OF_params)
pipeS.set_params(**{'MLP__early_stopping':False})               
iterationLC(pipeS,spam_trgX,spam_trgY,spam_tstX,spam_tstY,{'MLP__max_iter':[2**x for x in range(12)]},'ANN_OF','spam')

#pipeS_fs.set_params(**spam_fs_OF_params)
#pipeS_fs.set_params(**{'MLP__early_stopping':False})               
#iterationLC(pipeS_fs,spam_trgX,spam_trgY,spam_tstX,spam_tstY,{'MLP__max_iter':[2**x for x in range(12)]+[2100,2200,2300,2400,2500,2600,2700,2800,2900,3000]},'ANN_OF','spam_fs')