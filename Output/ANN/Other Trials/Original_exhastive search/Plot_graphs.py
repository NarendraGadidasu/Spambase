# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 19:14:12 2019

@author: A103932
"""

import pandas as pd

import matplotlib.pyplot as plt

res = pd.read_csv('ANN_spam_reg.csv')

#model complexity curve

res = res[['mean_test_score','mean_train_score', 
         'param_MLP__activation', 'param_MLP__alpha', 'param_MLP__hidden_layer_sizes']]

res1 = res.loc[res['param_MLP__hidden_layer_sizes'] == '(114, 114, 114)', :]

res1 = res1.loc[res['param_MLP__activation'] == 'relu', :]

res1 = res1.drop(columns = ['param_MLP__hidden_layer_sizes', 'param_MLP__activation'])

res1 = res1.rename(columns = {'mean_train_score':'train_balanced_accuracy', 'mean_test_score':'CV_balanced_accuracy'})

res1.plot('param_MLP__alpha', ['CV_balanced_accuracy', 'train_balanced_accuracy'])

plt.xlabel('MLP_alpha')

plt.xlim(0.00001, 10)

plt.title('Activation : relu and Hidden Layers : (114, 114, 114)')

plt.savefig('ANN_Acc_wrt_Alpha.png')

plt.clf()

#learning curve wrt samples

lc_train = pd.read_csv('ANN_spam_LC_train.csv')

lc_test = pd.read_csv('ANN_spam_LC_test.csv')

lc_train.rename(columns = {'Unnamed: 0':'num_of_samples'}, inplace = True)

lc_test.rename(columns = {'Unnamed: 0':'num_of_samples'}, inplace = True)

lc_train['train_balanced_accuracy'] = (lc_train['0']+lc_train['1']+lc_train['2']+lc_train['3']+lc_train['4'])/5

lc_test['CV_balanced_accuracy'] = (lc_test['0']+lc_test['1']+lc_test['2']+lc_test['3']+lc_test['4'])/5

lc_train.drop(columns = ['0','1','2','3','4'], inplace = True)

lc_test.drop(columns = ['0','1','2','3','4'], inplace = True)

plt.figure()

fig, ax = plt.subplots()

ax.plot(lc_train['num_of_samples'], lc_train['train_balanced_accuracy'], label = 'train_balanced_accuracy')

ax.plot(lc_test['num_of_samples'], lc_test['CV_balanced_accuracy'], label = 'CV_balanced_accuracy')

plt.legend()

plt.title('Learning curve wrt Number of training samples')

plt.xlabel('Number of training samples')

plt.savefig('ANN_LC_wrt_Samples.png')

#Learning Curve wrt Iterations

lc_iter = pd.read_csv('ITER_base_ANN_spam.csv')

lc_iter = lc_iter[['mean_test_score','mean_train_score', 'param_MLP__max_iter']]

lc_iter = lc_iter.rename(columns = {'mean_train_score':'train_balanced_accuracy', 'mean_test_score':'CV_balanced_accuracy'})

lc_iter.plot('param_MLP__max_iter', ['CV_balanced_accuracy', 'train_balanced_accuracy'])

plt.xlabel('Number of Iterations')

plt.title('Learning curve wrt Number of iterations')

plt.savefig('ANN_LC_wrt_Iterations.png')

plt.clf()



