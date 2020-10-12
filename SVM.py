
# -*- coding: utf-8 -*-

import numpy as np
import sklearn.model_selection as ms
import pandas as pd
from helpers import  basicResults,makeTimingCurve,iterationLC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import euclidean_distances
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix

class primalSVM_RBF(BaseEstimator, ClassifierMixin):
    '''http://scikit-learn.org/stable/developers/contributing.html'''
    
    def __init__(self, alpha=1e-9,gamma_frac=0.1,n_iter=2000):
         self.alpha = alpha
         self.gamma_frac = gamma_frac
         self.n_iter = n_iter
         
    def fit(self, X, y):
         # Check that X and y have correct shape
         X, y = check_X_y(X, y)
         
         # Get the kernel matrix
         dist = euclidean_distances(X,squared=True)
         median = np.median(dist) 
         del dist
         gamma = median
         gamma *= self.gamma_frac
         self.gamma = 1/gamma
         kernels = rbf_kernel(X,None,self.gamma )
         
         self.X_ = X
         self.classes_ = unique_labels(y)
         self.kernels_ = kernels
         self.y_ = y
         self.clf = SGDClassifier(loss='hinge',penalty='l2',alpha=self.alpha,
                                  l1_ratio=0,fit_intercept=True,verbose=False,
                                  average=False,learning_rate='optimal',
                                  class_weight='balanced',n_iter=self.n_iter,
                                  random_state=55)         
         self.clf.fit(self.kernels_,self.y_)
         
         # Return the classifier
         return self

    def predict(self, X):
         # Check is fit had been called
         check_is_fitted(self, ['X_', 'y_','clf','kernels_'])
         # Input validation
         X = check_array(X)
         new_kernels = rbf_kernel(X,self.X_,self.gamma )
         pred = self.clf.predict(new_kernels)
         return pred
    

spam = pd.read_hdf('spambase.hdf','spam')        
spamX = spam.drop('is_spam',1).copy().values
spamY = spam['is_spam'].copy().values

spam_trgX, spam_tstX, spam_trgY, spam_tstY = ms.train_test_split(spamX, spamY, test_size=0.3, random_state=0,stratify=spamY)     


N_spam = spam_trgX.shape[0]

alphas = [10**-x for x in np.arange(1,9.01,1/2)]


#Linear SVM
pipeS = Pipeline([('Scale',StandardScaler()),                
                 ('SVM',SGDClassifier(loss='hinge',l1_ratio=0,penalty='l2',class_weight='balanced',random_state=55))])

params_spam = {'SVM__alpha':alphas,'SVM__n_iter':[int((1e6/N_spam)/.8)+1]}
                                                  
spam_clf = basicResults(pipeS,spam_trgX,spam_trgY,spam_tstX,spam_tstY,params_spam,'SVM_Lin','spam') 

y_score = spam_clf.decision_function(spam_tstX)

fpr, tpr, thresholds = roc_curve(spam_tstY, y_score)

import matplotlib.pyplot as plt

plt.figure()

plt.plot(fpr, tpr)

plt.xlabel('FPR')
plt.ylabel('TPR')

plt.title('ROC_Curve(Spambase)')

plt.savefig('./output/SVM_Lin_ROC_Curve.png')

plt.clf()

cm = pd.DataFrame(confusion_matrix(spam_tstY, spam_clf.predict(spam_tstX)))

cm.to_csv('./output/SVM_Lin_Confusion_matrix.csv')

       

spam_final_params =spam_clf.best_params_
spam_OF_params ={'SVM__n_iter': 55, 'SVM__alpha': 1e-16}

pipeS.set_params(**spam_final_params)
makeTimingCurve(spamX,spamY,pipeS,'SVM_Lin','spam')
     
pipeS.set_params(**spam_final_params)
iterationLC(pipeS,spam_trgX,spam_trgY,spam_tstX,spam_tstY,{'SVM__n_iter':np.arange(1,75,3)},'SVM_Lin','spam')                

pipeS.set_params(**spam_OF_params)
iterationLC(pipeS,spam_trgX,spam_trgY,spam_tstX,spam_tstY,{'SVM__n_iter':np.arange(1,200,5)},'SVM_LinOF','spam')






#RBF SVM
gamma_fracsS = np.arange(0.2,2.1,0.2)

#
pipeS_fs = Pipeline([('Scale',StandardScaler()),
                 ('Cull1',SelectFromModel(RandomForestClassifier(random_state=1),threshold='median')),
                 ('Cull2',SelectFromModel(RandomForestClassifier(random_state=2),threshold='median')),
                 ('Cull3',SelectFromModel(RandomForestClassifier(random_state=3),threshold='median')),
                 ('Cull4',SelectFromModel(RandomForestClassifier(random_state=4),threshold='median')),
                 ('SVM',primalSVM_RBF())])

pipeS = Pipeline([('Scale',StandardScaler()),
                 ('SVM',primalSVM_RBF())])


params_spam = {'SVM__alpha':alphas,'SVM__n_iter':[int((1e6/N_spam)/.8)+1],'SVM__gamma_frac':gamma_fracsS}

#                                                  
spam_clf_fs = basicResults(pipeS_fs,spam_trgX,spam_trgY,spam_tstX,spam_tstY,params_spam,'SVM_RBF','spam_fs')        
spam_clf = basicResults(pipeS,spam_trgX,spam_trgY,spam_tstX,spam_tstY,params_spam,'SVM_RBF','spam')        



spam_fs_final_params = spam_clf_fs.best_params_
spam_fs_OF_params = spam_fs_final_params.copy()
spam_fs_OF_params['SVM__alpha'] = 1e-16
spam_final_params = spam_clf.best_params_
spam_OF_params = spam_final_params.copy()
spam_OF_params['SVM__alpha'] = 1e-16

pipeS_fs.set_params(**spam_fs_final_params)                     
makeTimingCurve(spamX,spamY,pipeS_fs,'SVM_RBF','spam_fs')
pipeS.set_params(**spam_final_params)
makeTimingCurve(spamX,spamY,pipeS,'SVM_RBF','spam')


pipeS_fs.set_params(**spam_fs_final_params)
iterationLC(pipeS_fs,spam_trgX,spam_trgY,spam_tstX,spam_tstY,{'SVM__n_iter':[2**x for x in range(12)]},'SVM_RBF','spam_fs')        
pipeS.set_params(**spam_final_params)
iterationLC(pipeS,spam_trgX,spam_trgY,spam_tstX,spam_tstY,{'SVM__n_iter':np.arange(1,75,3)},'SVM_RBF','spam')                

pipeS.set_params(**spam_OF_params)
iterationLC(pipeS, spam_trgX,spam_trgY,spam_tstX,spam_tstY,{'SVM__n_iter':np.arange(1,75,3)},'SVM_RBF_OF','spam')                
pipeS_fs.set_params(**spam_fs_OF_params)
iterationLC(pipeS_fs,spam_trgX,spam_trgY,spam_tstX,spam_tstY,{'SVM__n_iter':np.arange(100,2600,100)},'SVM_RBF_OF','spam_fs')                
