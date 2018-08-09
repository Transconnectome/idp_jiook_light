
# coding: utf-8

# In[2]:


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
from sklearn import linear_model, svm
from sklearn.ensemble import ExtraTreesClassifier
#import xgboost as xgb

print('finished this block')


# In[3]:


os.getcwd()
os.chdir('../data/adni')
print('finished this block')


# In[4]:


dd =pd.read_csv("combine_new_biomarker_correct.csv",header=0)
print('the original training data dimension is')
print(dd.shape)
import csv


# In[29]:


with open('combine_new_biomarker_correct.csv', 'r') as f:
    d_reader = csv.DictReader(f)
    #get fieldnames from DictReader object and store in list
    headers = d_reader.fieldnames

os.getcwd()
os.chdir('../../Chenxiao_Results_Output')

data=np.array(dd)
idx_IN_columns = np.append(np.array(range(1,6)),np.array([14,22]))
idx_IN_columns = np.append(idx_IN_columns,np.array(range(23,data.shape[1])))

X=data[:,idx_IN_columns]
#X_biomarker=X[:,0:5]

y=data[:,11]
#y_dxbl=data[:,10]

#ind_num_matrix=np.isnan(X_biomarker)
#ind_num_vector=np.any(ind_num_matrix,axis=1)

#X_no_nan=X[~ind_num_vector,:]
#y_no_nan=y[~ind_num_vector]
#y_dxbl_no_nan = y_dxbl[~ind_num_vector]


#X=X_no_nan
#y=y_no_nan
#y_dxbl=y_dxbl_no_nan

#MCI = (y_dxbl==2)
#MCI_index =[ i for i in range(0, MCI.shape[0]) if MCI[i]]

#X = X[MCI_index, :]
#y = y[MCI_index]

X = X[~ind_num, :]
y = y[~ind_num]


X = stats.zscore(X)

np.isnan(X).any()
X[np.isnan(X)] = np.median(X[~np.isnan(X)])

print("after the precoessing, the X.shape is ")
print(X.shape)
print("the y.shape is")
print(y.shape)

print("the number of MCI case is")
print(len(MCI_index))

test_number=1

C_all=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0]
#C_all=[0.01, 0.1]
textfile_name='ADvsnormal_converter_LR_grid_search'+str(test_number)+'.txt'

sep1 = '*' * 100
sep2 = '*' * 50
sep3 = '*' * 30

# In[ ]:


print("Begin")
f = open(textfile_name, 'w')
f.write("begin!!\n")
f.close()

# In[30]:

for C_select in C_all:

    sep1 = '*' * 100
    sep2 = '*' * 50
    sep3 = '*' * 30
   
    print("beginning selection best number of features")
    n_features = [10,30,50,70,80,100,1000,2000]
    n_features = [10, 30, 50, 70, 100]
    logistic = linear_model.LogisticRegression(C=C_select)


    accr_feature = []
    f1s_feature = []
    accr_MCI_feature = []
    f1s_MCI_feature = []     

    for i in n_features:
        print("\n\n Number of Feature: {} {} \n".format(i, sep1))
      
        accr_run  = []
        f1s_run = []
        accr_MCI_run  = []
        f1s_MCI_run = []
        base_labels=[]
    
        for runs in range(10):
            counter=0
            print("\n RUN: {} {} \n".format(runs, sep3))
        
            accr_CV = []
            f1s_CV=[]

            test_labels_MCI_CV = []
            y_pred_MCI_CV = []
            
            strat_labels = []
        
            skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=int(runs))
            for train_index, test_index in skf.split(X, y):       
                print("\n Fold: {} {} \n".format(counter, sep2))        
                counter=counter+1
                        
                train_data_origin, test_data_origin=X[train_index], X[test_index]
                train_labels, test_labels = y[train_index], y[test_index]
                strat_labels = np.append(strat_labels, test_labels)

 #           print("Random forest for feature selection")
                clf = ExtraTreesClassifier(n_estimators=250,random_state=0)
                clf = clf.fit(train_data_origin, train_labels)
                importances = clf.feature_importances_
                indices = np.argsort(importances)[::-1]        
            
                index=indices[0:i]
                train_data=train_data_origin[:, index]
                test_data=test_data_origin[:, index]

                test_index_MCI = [test_index[i] for i in range(0, len(test_index)) if test_index[i] in MCI_index]
               
                                     
        #SVM
                log = logistic.fit(train_data, train_labels)
                acc = log.score(test_data, test_labels)
                
            #f1 calculation
                y_pred = logistic.predict(test_data)
                f1 = f1_score(test_labels, y_pred)
                     
                print('Accuracy: %f' % acc)
                print('F1 score: %f' % f1)
                    
                accr_CV = np.append(accr_CV, acc)
                f1s_CV=np.append(f1s_CV, f1)

        #MCI_case            
#                if (len(test_index_MCI)==0):
#                    print("There is no MCI in this shuffling test set, so we skip it")
#                else:
#                    test_data_MCI_origin = X[test_index_MCI]
#                    test_data_MCI = test_data_MCI_origin[:, index]      
#                    test_labels_MCI = y[test_index_MCI]
            
#                    y_pred_MCI = logistic.predict(test_data_MCI)
                
#                    print(y_pred_MCI)
#                    print(test_labels_MCI)
                                    
#                    y_pred_MCI_CV = np.append(y_pred_MCI_CV, y_pred_MCI)
#                    test_labels_MCI_CV = np.append(test_labels_MCI_CV, test_labels_MCI)
                

            base_labels=np.append(base_labels, strat_labels)        
            accr_run=np.append(accr_run, np.mean(accr_CV))
            f1s_run =np.append(f1s_run, np.mean(f1s_CV))
#            accr_MCI_run = np.append(accr_MCI_run, np.sum(test_labels_MCI_CV==y_pred_MCI_CV)/len(test_labels_MCI_CV))
#            f1s_MCI_run = np.append(f1s_MCI_run, f1_score(test_labels_MCI_CV, y_pred_MCI_CV))

            print("the mean accr_CV is")
            print(np.mean(accr_CV))
            print("the mean f1s_CV is")
            print(np.mean(f1s_CV))
#            print("the accr_MCI_CV is")
#            print(np.sum(test_labels_MCI_CV==y_pred_MCI_CV)/len(test_labels_MCI_CV))
#            print("the f1_score_MCI is")
#            print(f1_score(test_labels_MCI_CV, y_pred_MCI_CV))

                           
        accr_feature=np.append(accr_feature, np.mean(accr_run))
        f1s_feature=np.append(f1s_feature, np.mean(f1s_run))
#        accr_MCI_feature=np.append(accr_MCI_feature, np.mean(accr_MCI_run))
#        f1s_MCI_feature=np.append(f1s_MCI_feature, np.mean(f1s_MCI_run))


    f = open(textfile_name,'a')
    f.write('\nC=%f. \n' % C_select )
    f.write("Runs Max Accuracies: {}".format(np.max(accr_feature)))
    f.write(" Happens at:")
    f.write(str(np.argmax(accr_feature)))
    f.write("\nRuns Max F1s: {}".format(np.max(f1s_feature)))
    f.write(" Happens at:")
    f.write(str(np.argmax(f1s_feature)))
#    f.write("\nRuns Max MCI Accuracies: {}".format(np.max(accr_MCI_feature)))
#    f.write(" Happens at:")
#    f.write(str(np.argmax(accr_MCI_feature)))
#    f.write("\nRuns Max MCI F1s: {}".format(np.max(f1s_MCI_feature)))
#    f.write(" Happens at:")
#    f.write(str(np.argmax(f1s_MCI_feature)))   

    f.close()
    print("finished one loop")
            
