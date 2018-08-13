#!/opt/apps/gcc7_1/python3/3.6.1/bin/python3
# coding: utf-8

# In[1]:

from __future__ import print_function
import pickle
import os
import scipy.io
from scipy import stats
import scipy.stats as st

import pandas as pd
from numpy import *
from sklearn.svm import SVC
from sklearn import linear_model
import numpy as np

import matplotlib
matplotlib.use('Agg') # TO SET METPLOTLIB NOT TO USE XWINDOWS BACKEND
from matplotlib import pyplot as plt

from scipy import stats
import pandas as pd
import argparse
from sklearn.model_selection import GridSearchCV, cross_val_score,cross_val_predict,StratifiedKFold
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score,confusion_matrix
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc,f1_score

import sys, getopt
# K=sys.argv[1]

def main(argv):
	K = ''
	try:
		opts, args = getopt.getopt(argv,"k:h")
	except getopt.GetoptError:
		print("GetoptError")
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print("ADNI_Combine.py -k <kfold>")
			sys.exit()
		elif opt in ("-k"):
			K = int(arg)
			print("k for cv is "+str(K))
	return K

K = main(sys.argv[1:])

print("K fold is "+ str(K))
# K=10

##read the data and clean data
def data_fetch_clean(file,type):
    #os.getcwd()
    #os.chdir('../braindata')
    dd =pd.read_csv(file,header=0)
    print(dd.shape)
    import csv

    with open(file, 'r') as f:
        d_reader = csv.DictReader(f)

        #get fieldnames from DictReader object and store in list
        headers = d_reader.fieldnames
    data=np.array(dd)
    #print(data.shape)
    idx_IN_columns = np.array(range(18,data.shape[1]))
    #idx_IN_columns = np.append(arange(1,6),np.array(range(24,data.shape[1])))
   # idx_IN_columns = np.append(np.array([3]),np.array(range(11,data.shape[1])))

    print(idx_IN_columns)
    X=data[:,idx_IN_columns]
    #features=data[:,11:data.shape[1]]
    #features = features.transpose()
    X = stats.zscore(X)
    #print(features.shape)
    y=data[:,type]
#/ 6:AD-normal / 7:AD-MCI / 8:MCI-normal


    ind_num=np.isnan(y)
    # print(ind_num.shape)


    y_no_nan = y[~ind_num]

    X_no_nan = X[~ind_num,:]

           # print(y.shape)

    y=y_no_nan
    X=X_no_nan
    feature_num_all=[]
    lr_all_feature=[]
    svm_all_feature=[]
    lr_fls_feature=[]
    svm_fls_feature=[]
    base_labels= []

    np.isnan(X).any()

    X[np.isnan(X)] = np.median(X[~np.isnan(X)])
    return X,y


# In[2]:

from sklearn.decomposition import PCA


# In[3]:

def main_classifier(X,y,name,filename,params,pipe,path_to_save,key):
    all_TP = []
    all_TN = []
    all_FP = []
    all_FN = []

    all_acc = []
    all_sen = []
    all_spec = []
    all_auc = []
    all_f1s = []

    all_roc_label = []
    all_roc_pred = []
    all_roc_prob = []
    all_features=[]
    n_fold = K
    rs_list=[33994,31358,27381,8642,7012,42023,44642,44002,30706,12571]
    # rs_list = [33994,31358,27381]
#    rs_list = [33994]
    for rs in rs_list:
        print('********random seed:{}'.format(rs))

        inner_cv = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=rs)
        outer_cv = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=rs)

#         scaler = StandardScaler()
        scaler = RobustScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        avg_auc = []
        avg_acc = []
        avg_TP = []
        avg_TN = []
        avg_FP = []
        avg_FN = []
        avg_sen = []
        avg_spec = []
        avg_f1s = []

        roc_label = []
        roc_pred = []
        roc_prob = []
        features=[]
        for train_index, test_index in outer_cv.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            clf = GridSearchCV(estimator=pipe, param_grid=params, cv=inner_cv, scoring='accuracy',n_jobs=48)
            clf.fit(X_train, y_train)

            fs = clf.best_estimator_.named_steps['featureExtract']
            mask = fs.get_support(indices=True)
            print(mask.shape)
            features=np.append(features,mask[:30,])
            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)
          #  roc_pred = clf.predict_proba(X_test)

            acc = accuracy_score(y_test, y_pred)
            f1=f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob[:, 1])


            roc_label = np.append(roc_label, y_test)
            roc_pred = np.append(roc_pred, y_pred)
            roc_prob = np.append(roc_prob, y_prob[:, 1])


            conf_mat = confusion_matrix(y_test, y_pred)

            TP = conf_mat[0][0]
            FP = conf_mat[0][1]
            FN = conf_mat[1][0]
            TN = conf_mat[1][1]

            avg_TP = np.append(avg_TP, TP)
            avg_TN = np.append(avg_TN, TN)
            avg_FP = np.append(avg_FP, FP)
            avg_FN = np.append(avg_FN, FN)

            avg_acc = np.append(avg_acc, acc)
            avg_f1s=np.append(avg_f1s,f1)
            print(TP, FP, FN, TN)
            sen = TP / (TP + FN)
            spec = TN / (TN + FP)

            avg_sen = np.append(avg_sen, sen)
            avg_spec = np.append(avg_spec, spec)
            avg_auc = np.append(avg_auc, auc)

        all_TP = np.append(all_TP, avg_TP)
        all_TN = np.append(all_TN, avg_TN)
        all_FP = np.append(all_FP, avg_FP)
        all_FN = np.append(all_FN, avg_FN)

        all_acc = np.append(all_acc, avg_acc)
        all_sen = np.append(all_sen, avg_sen)
        all_spec = np.append(all_spec, avg_spec)
        all_auc = np.append(all_auc, avg_auc)
        all_f1s=np.append(all_f1s,avg_f1s)

        all_roc_label = np.append(all_roc_label, roc_label)
        all_roc_pred = np.append(all_roc_pred, roc_pred)
        all_roc_prob = np.append(all_roc_prob, roc_prob)
        all_features=np.append(all_features,features)
        all_acc= all_acc.reshape(-1,1)
        all_sen= all_sen.reshape(-1,1)
        all_spec= all_spec.reshape(-1,1)
        all_auc= all_auc.reshape(-1,1)
        all_f1s= all_f1s.reshape(-1,1)


        #FD=pd.DataFrame(np.hstack((all_acc,all_sen,all_spec,all_auc,all_f1s)),columns=["Accuracy","Sensitivity","Specificity","AUC","F1"])

#         print("Accuracy Avg: {}".format(np.mean(avg_acc)))
#         print("Accuracy Standard Deviation: {}".format(np.std(avg_acc)))
#         print("Sensitivity Avg: {}".format(np.mean(avg_sen)))
#         print("Sensitivity Standard Deviation: {}".format(np.std(avg_sen)))
#         print("Specificity Avg: {}".format(np.mean(avg_spec)))
#         print("Specificity Standard Deviation: {}".format(np.std(avg_spec)))

        #if dataset == 1:
    pickle.dump(all_roc_label, open(path_to_save+'/'+'roc_label_'+name+'.p', "wb"))
    pickle.dump(all_roc_pred, open(path_to_save+'/'+'roc_pred_'+name+'.p', "wb"))
    pickle.dump(all_roc_prob, open(path_to_save+'/'+'roc_prob_'+name+'.p', "wb"))
    pickle.dump(all_features, open(path_to_save+'/'+'features_30'+name+'.p', "wb"))

    acc_CI=st.t.interval(0.95, len(all_acc)-1, loc=np.nanmean(all_acc), scale=st.sem(all_acc))
    sen_CI=st.t.interval(0.95, len(all_sen)-1, loc=np.nanmean(all_sen), scale=st.sem(all_sen))
    spec_CI=st.t.interval(0.95, len(all_spec)-1, loc=np.nanmean(all_spec), scale=st.sem(all_spec))
    auc_CI=st.t.interval(0.95, len(all_auc)-1, loc=np.nanmean(all_auc), scale=st.sem(all_auc))
    import os

#     if not os.path.exists('../imgs3_idp/'+  todaystr+'/'+filename):
#         os.makedirs('../imgs3_idp/'+filename)
    txt_name=path_to_save+'/'+name +  '.txt'
    print("ACC={a},  95%CI={l}-{u}, sd={sd}".format(a=np.nanmean(all_acc),l=acc_CI[0], u=acc_CI[1], sd=np.nanstd(all_acc)),file=open(txt_name, "a"))
    print("AUC={a}, 95%CI={l}-{u},sd={sd}".format(a=np.nanmean(all_auc), l=auc_CI[0], u=auc_CI[1], sd=np.nanstd(all_auc)),file=open(txt_name, "a"))
    print("SENSITIVITY={a}, 95%CI={l}-{u},sd={sd}".format(a=np.nanmean(all_sen), l=sen_CI[0],u=sen_CI[1], sd=np.nanstd(all_sen)),file=open(txt_name, "a"))
    print("SPECIFICITY={a}, 95%CI={l}-{u},sd={sd}".format(a=np.nanmean(all_spec), l=spec_CI[0],u=spec_CI[1], sd=np.nanstd(all_spec)),file=open(txt_name, "a"))



    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    fpr, tpr, _ = roc_curve(all_roc_label, all_roc_prob)
    #auc = roc_auc_score(all_roc_label, all_roc_prob)

    plt.figure()

    plt.plot(fpr, tpr, lw=2, label=key+ ' '+'(AUC = %0.2f)' % np.mean(all_auc))
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(name)
    plt.legend(loc="lower right")
    #plt.savefig('10x_Combined_ROC.eps')
    roc_name=path_to_save+'/'+name +'.pdf'
    plt.savefig(roc_name)
    #plt.show()

#    return





# In[4]:

save_name=["AD vs SMC","AD vs MCI","MCI vs SMC"]


# In[5]:

filename='combine_new'
file=filename+'.csv'
cwd=os.getcwd()
os.chdir('../data/adni')

print(os.getcwd())

# In[6]:

import time, datetime, os

today = datetime.date.today()

todaystr = today.isoformat()
os.makedirs('../../imgs3_adni/'+ todaystr, exist_ok=True)
os.makedirs('../../imgs3_adni/'+ todaystr+'/'+filename+'_'+ str(K)+'fold', exist_ok=True)

# if not os.path.exists('../../imgs3_adni/'+todaystr):
#    # print('exist')
#     os.mkdir('../../imgs3_adni/'+ todaystr)
#     #os.mkdir('../../imgs3_adni/'+ todaystr+'/'+filename)
# if not os.path.exists('../imgs3_adni/'+todaystr+'/'+filename+'_3fold'):
#    # print('exist')
#     #os.mkdir('../../imgs3_adni/'+ todaystr)
#     os.mkdir('../../imgs3_adni/'+ todaystr+'/'+filename+'_3fold')


# In[ ]:

models1 = {

    'RandomForestClassifier': RandomForestClassifier(),
    'SVC': SVC(probability=True),
    'linear_model.LogisticRegression':linear_model.LogisticRegression()

}

params1 = {
            'RandomForestClassifier': [{ 'RandomForestClassifier__n_estimators': np.arange(10, 500, 50) },
                                       {'RandomForestClassifier__min_samples_leaf': np.arange(1, 51, 5)},
                                      ],
    'SVC': [
        {'SVC__kernel': ['linear'], 'SVC__C': [0.001,0.01,0.1,1, 10]},
    ],
    'linear_model.LogisticRegression':{'linear_model.LogisticRegression__C':[0.001, 0.01, 0.1, 1, 10]}
}

path_save='../../imgs3_adni/' + todaystr+'/'+filename+'_noPCA_'+ str(K)+'fold' + '/'
# C={}
# C['models']=models
# C['params']=params
# C['keys']=models.keys()


# In[ ]:

for key, value in models1.items():
    # pipe=Pipeline([('PCA',PCA()),
    #             ('featureExtract', SelectFromModel(ExtraTreesClassifier())),
    #             (key, models1[key])
    #         ])
    pipe=Pipeline([
	            ('featureExtract', SelectFromModel(ExtraTreesClassifier())),
	            (key, models1[key])
	        	])
    print(key)
    para=params1[key]
    path_to_save=path_save+key
    os.makedirs(path_to_save, exist_ok=True)
#     if not os.path.exists(path_to_save):
#         os.mkdir(path_to_save)
    MM=pd.DataFrame()
    SS=pd.DataFrame()
    for i in range(6,9):
            print(save_name[i-6])
            X,y=data_fetch_clean(file,i)

            y = y.reshape(-1)
#             F=SelectFromModel(ExtraTreesClassifier(),prefit=True)
#             X_feature=F.transform(X)
#             list(X_features)
            name=save_name[i-6]
            main_classifier(X,y,name,filename,para,pipe,path_to_save,key)
            #FD=main_classifier(X,y,name,filename,para,pipe,path_to_save,key)
            #M=FD.mean(axis=0)

            #S=FD.std(axis=0)
            ##S.rename(index={S.index[0]: name})

            #MM=MM.append(M,ignore_index=True)
            #SS=SS.append(S,ignore_index=True)
    #MM.index=save_name
    #SS.index=save_name
    #writer = pd.ExcelWriter(path_save+key+'_Performance.xlsx')
    #MM.to_excel(writer,'Mean')
    #SS.to_excel(writer,'SD')
    #writer.save()
