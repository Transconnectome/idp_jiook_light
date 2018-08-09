
# coding: utf-8

# In[1]:

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
import numpy as np
import pandas as pd
from pprint import pprint
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import StratifiedKFold
from scipy import stats
import numpy as np
from sklearn import linear_model, svm
import re
from sklearn.metrics import roc_curve, auc,f1_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
#s = "../braindata/data_1_mor_select_100.csv"
import os          
from sklearn.ensemble import ExtraTreesClassifier
import sys, os
sys.path.insert(0, '..')
from lib import models, graph, coarsening, utils
import tensorflow as tf
import time
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from scipy import stats
import matplotlib.pyplot as plt


print('finished this block')


# In[2]:


os.getcwd()
os.chdir('../data/adni')
print('finished this block')


# In[3]:


dd =pd.read_csv("combine_new_biomarker_correct.csv",header=0)
print('the original training data dimension is')
print(dd.shape)
import csv


# In[4]:


with open('combine_new_biomarker_correct.csv', 'r') as f:
    d_reader = csv.DictReader(f)
    #get fieldnames from DictReader object and store in list
    headers = d_reader.fieldnames
    
data=np.array(dd)
idx_IN_columns = np.append(np.array(range(1,6)),np.array([14,22]))
idx_IN_columns = np.append(idx_IN_columns,np.array(range(23,data.shape[1])))

X=data[:,idx_IN_columns]
X_biomarker=X[:,0:5]

y=data[:,9]  
y_dxbl=data[:,10]

ind_num_matrix=np.isnan(X_biomarker)
ind_num_vector=np.any(ind_num_matrix,axis=1)

X_no_nan=X[~ind_num_vector,:]
y_no_nan=y[~ind_num_vector]
y_dxbl_no_nan = y_dxbl[~ind_num_vector]

X=X_no_nan
y=y_no_nan
y_dxbl=y_dxbl_no_nan

MCI = (y_dxbl==2)
MCI_index =[ i for i in range(0, MCI.shape[0]) if MCI[i]]

X = stats.zscore(X)

base_labels= []


np.isnan(X).any()
X[np.isnan(X)] = np.median(X[~np.isnan(X)])

print("after the precoessing, the X.shape is ")
print(X.shape)
print("the y.shape is")
print(y.shape)
print("the number of MCI case is")
print(len(MCI_index))


# In[5]:


os.getcwd()
os.chdir('../../Chenxiao_Results_Output')


# In[6]:


test_number=23

C = max(y) + 1  # number of classes    

common = {}
common['dir_name']       = 'CX_GCN'
common['num_epochs']     = 35
common['batch_size']     = 5
common['eval_frequency'] = common['num_epochs']
common['brelu']          = 'b1relu'
common['pool']           = 'mpool1'
common['filter']         = 'chebyshev5'
# Architecture.
common['F']              = [64, 32]  # Number of graph convolutional filters.
common['K']              = [10, 10]  # Polynomial orders.
common['p']              = [4, 4]    # Pooling sizes.
common['M']              = [256, 512, 128, C]  # Output dimensionality of fully connected layers.s
# Optimization.
common['regularization'] = 1e-4
common['dropout']        = 0.5
common['learning_rate']  = 5e-4
common['decay_rate']     = 0.95
common['momentum']       = 0.9

model_perf = utils.model_perf()    

number_of_features = 100
Metric = 'euclidean'
K=10


textfile_name=common['dir_name']+str(test_number)+'.txt'
f = open(textfile_name,'w')
f.write('This is the Result Output for the %d th testing. \n' % test_number )
f.write('The following tis the super-parameter setting:\n')
f.write('num_epochs: %d \n' % common['num_epochs'])
f.write('batch_size: %d \n' % common['batch_size'])
f.write('regularization: %f  \n' % common['regularization'])
f.write('dropout: %f  \n' % common['dropout'])
f.write('learning_rate: %f  \n' % common['learning_rate'])
f.write('decay_rate: %f  \n' % common['decay_rate'])
f.write('number_of_features: %d: \n' % number_of_features)
f.write('Metric: %s \n' % Metric )
f.write('K: %d \n' % K )
f.write('Common[F]')
f.write(str(common['F']))
f.write('\nCommon[K]')
f.write(str(common['K']))
f.write('\nCommon[p]')
f.write(str(common['p']))
f.write('\nCommon[M]')
f.write(str(common['M']))


f.close()
print('finish this block')


# In[7]:


sep1 = '*' * 100
sep2 = '*' * 50
sep3 = '*' * 30

accr_run = []
f1s_run = []
accr_MCI_run = []
f1s_MCI_run = []

base_labels= []

for runs in range(10):        
    counter=0
    print("\n RUN: {} {} \n".format(runs, sep3))

    f = open(textfile_name,'a')
    f.write("\n RUN: {} {} \n".format(runs, sep3))
    f.close()
    
    strat_labels = []
    accr_CV = []
    f1s_CV = []
    
    test_labels_MCI_CV = []
    y_pred_MCI_CV = []
        
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=int(runs))
    for train_index, test_index in skf.split(X, y):
        counter = counter+1
        print("\n Fold: {} {} \n".format(counter, sep2))

        f = open(textfile_name,'a')
        f.write("\n Fold: {} {} \n".format(counter, sep2))
        f.close()

        train_data_origin, test_data_origin = X[train_index], X[test_index]
        train_labels, test_labels = y[train_index], y[test_index]

        strat_labels = np.append(strat_labels, test_labels)

        clf = ExtraTreesClassifier(n_estimators=250,random_state=0)
        clf = clf.fit(train_data_origin, train_labels)
        importances = clf.feature_importances_
        indices = np.argsort(importances)[::-1]        
            
        index=indices[0:number_of_features]
        train_data=train_data_origin[:, index]
        test_data=test_data_origin[:, index]                          
                               
        test_index_MCI = [test_index[i] for i in range(0, len(test_index)) if test_index[i] in MCI_index]
                                            
###################################################
#Chenxiao: generating permuational matrix
        dist, idx = graph.distance_scipy_spatial(train_data.transpose(), k=10, metric= Metric)
        A = graph.adjacency(dist, idx).astype(np.float32)
        assert A.shape == (train_data.shape[1], train_data.shape[1])
        print('d = |V| = {}, k|V| < |E| = {}'.format(train_data.shape[1], A.nnz))
        plt.spy(A, markersize=2, color='black')     
        graphs, perm = coarsening.coarsen(A, levels=6, self_connections=False)
        
        train_data = coarsening.perm_data(train_data, perm)
 #       val_data = coarsening.perm_data(test_data, perm)
        test_data = coarsening.perm_data(test_data, perm)

        L = [graph.laplacian(A, normalized=True) for A in graphs]
 #       graph.plot_spectrum(L)    
 ###################################################        
    
        name = 'CGCNN'
        params = common.copy()
        params['dir_name'] = textfile_name + 'run' +str(runs) + 'counter' + str(counter)
        params['decay_steps'] = len(train_labels) / common['batch_size']
        
        print('begin working!!!!!!!!')
        f = open(textfile_name, 'a')
        f.write("begin working!!!!!!!!\n")
        f.close()
                          
                          
        Models1=models.cgcnn(L, **params)
                
        model_perf.test(Models1, name, params, train_data, train_labels, test_data, test_labels, test_data, test_labels, test_index)
        scores, f1, y_labels, test_accuracy, y_pred = model_perf.show()   


        acc = np.sum(y_pred == test_labels) / test_labels.shape[0]                    
        f1 = f1_score(test_labels, y_pred)
                          
                          
        print('Accuracy: %f' % acc)
        print('F1 score: %f' % f1)                          
            
        accr_CV = np.append(accr_CV, acc)
        f1s_CV=np.append(f1s_CV, f1)    
                          
                          
        strat_labels=np.append(strat_labels, test_labels)    
                          
        #MCI_case            
        if (len(test_index_MCI)==0):
            print("There is no MCI in this shuffling test set, so we skip it")
        else:
            test_data_MCI_origin = X[test_index_MCI]
            test_data_MCI = test_data_MCI_origin[:, index]
            test_labels_MCI = y[test_index_MCI]
                    
            test_data_MCI = coarsening.perm_data(test_data_MCI, perm)

                          
            Models2=models.cgcnn(L, **params)
            
            name2 = 'CGCNN'
            
            model_perf.test(Models2, name2, params, train_data, train_labels, test_data_MCI, test_labels_MCI, test_data_MCI, test_labels_MCI, test_index)
            scores, f1, y_labels, test_accuracy, y_pred_MCI = model_perf.show()                   
            
            print(y_pred_MCI)
            print(test_labels_MCI)
                                    
            y_pred_MCI_CV = np.append(y_pred_MCI_CV, y_pred_MCI)
            test_labels_MCI_CV = np.append(test_labels_MCI_CV, test_labels_MCI)
        
####################################################  
                          
    base_labels=np.append(base_labels, strat_labels)  
    print("the mean accr_CV is")
    print(np.mean(accr_CV))
    print("the mean f1s_CV is")
    print(np.mean(f1s_CV))
    print("the accr_MCI_CV is")
    print(np.sum(test_labels_MCI_CV==y_pred_MCI_CV)/len(test_labels_MCI_CV))
    print("the f1_score_MCI is")
    print(f1_score(test_labels_MCI_CV, y_pred_MCI_CV))
    
    
    f = open(textfile_name,'a')

    f.write("\n the accr_MCI_CV is")
    a=np.sum(test_labels_MCI_CV==y_pred_MCI_CV)/len(test_labels_MCI_CV)
    f.write(str(a))
    f.write("\n the f1_score_MCI is")
    b=f1_score(test_labels_MCI_CV, y_pred_MCI_CV)
    f.write(str(b))
     

    f.close()    
                        
    accr_run = np.append(accr_run, np.mean(accr_CV))
    f1s_run = np.append(f1s_run, np.mean(f1s_CV))
    accr_MCI_run = np.append(accr_MCI_run, np.sum(test_labels_MCI_CV==y_pred_MCI_CV)/len(test_labels_MCI_CV))
    f1s_MCI_run = np.append(f1s_MCI_run, f1_score(test_labels_MCI_CV, y_pred_MCI_CV))                          

            
print("Runs Avg Accuracies: {}".format(np.mean(accr_run)))
print("Standard Deviation: {}".format(np.std(accr_run)))
print("Runs Avg F1: {}".format(np.mean(f1s_run)))
print("Standard Deviation: {}".format(np.std(f1s_run)))
print("Runs Avg Accuracies_MCI: {}".format(np.mean(accr_MCI_run)))
print("Standard Deviation: {}".format(np.std(accr_MCI_run)))    
print("Runs Avg Accuracies_MCI: {}".format(np.mean(f1s_MCI_run)))
print("Standard Deviation: {}".format(np.std(f1s_MCI_run)))     

f = open(textfile_name,'a')

f.write("\n Runs Avg Accuracies: {}".format(np.mean(accr_run)))
f.write("\n Standard Deviation: {}".format(np.std(accr_run)))    
f.write("\n Runs Avg F1s: {}".format(np.mean(f1s_run)))
f.write("\n Standard Deviation: {}".format(np.std(f1s_run))) 

f.write("\n Runs Avg Accuracies_MCI: {}".format(np.mean(accr_MCI_run)))
f.write("\n Standard Deviation: {}".format(np.std(accr_MCI_run)))    
f.write("\n Runs Avg F1s_MCI: {}".format(np.mean(f1s_MCI_run)))
f.write("\n Standard Deviation: {}".format(np.std(f1s_MCI_run)))   

f.close()

