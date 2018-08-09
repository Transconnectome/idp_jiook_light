# Import modules
from __future__ import print_function
import pickle
import os
import scipy.io
from scipy import stats

import pandas as pd
from numpy import *

import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
import pandas as pd
import argparse
from sklearn.model_selection import GridSearchCV, cross_val_score,cross_val_predict,StratifiedKFold,KFold
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score,confusion_matrix, mean_squared_error,precision_score,recall_score,f1_score,hamming_loss
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
# ===========================================================
# Training settings
# ===========================================================

def argumentparser():
    parser = argparse.ArgumentParser(description='PyTorch Connectome CNN')
    # hyper-parameters
    parser.add_argument('--dataset', type=int, default=1, help='select a dataset (1:connectome, 2: connectome + morphometry)')
    parser.add_argument('--num_feat', type=int, default=12,
                        help='the number of features')
    parser.add_argument('--atlas', type=int, default=3,
                        help='atlas type: (1-164 atlas, 2-84 atls, 3-both')
    parser.add_argument('--task', type=int, default=1,
                        help='task type: (1-binary classification, 2-regression')
    parser.add_argument('--multilabel', type=int, default=0,
                        help='multilabel type: (0-single label, 1-multilable')
    return parser

def data_fetch_clean(atlas, num_features,task,multi):
    cwd = os.getcwd()
    os.chdir('../CNN_Alex')
    # test=scipy.io.loadmat('Connectome_group_aparc.a2009s+aseg_AD.mat')

    test = scipy.io.loadmat('Connectome_group_aparc.a2009s+aseg_AD.mat')
    AD_2009 = np.array(test['connectome'])
    # print(AD_2009.shape)

    test = scipy.io.loadmat('Connectome_group_aparc.a2009s+aseg_count.mat')
    count_2009 = np.array(test['connectome'])
    # print(count_2009.shape)

    test = scipy.io.loadmat('Connectome_group_aparc.a2009s+aseg_FA.mat')
    FA_2009 = np.array(test['connectome'])
    # print(FA_2009.shape)

    test = scipy.io.loadmat('Connectome_group_aparc.a2009s+aseg_length.mat')
    length_2009 = np.array(test['connectome'])
    # print(length_2009.shape)

    test = scipy.io.loadmat('Connectome_group_aparc.a2009s+aseg_MD.mat')
    MD_2009 = np.array(test['connectome'])
    # print(MD_2009.shape)

    test = scipy.io.loadmat('Connectome_group_aparc.a2009s+aseg_RD.mat')
    RD_2009 = np.array(test['connectome'])
    # print(RD_2009.shape)

    test = scipy.io.loadmat('Connectome_group_aparc+aseg_AD.mat')
    AD_aseg = np.array(test['connectome'])
    # print(AD_aseg.shape)

    test = scipy.io.loadmat('Connectome_group_aparc+aseg_count.mat')
    count_aseg = np.array(test['connectome'])
    # print(count_aseg.shape)

    test = scipy.io.loadmat('Connectome_group_aparc+aseg_FA.mat')
    FA_aseg = np.array(test['connectome'])
    # print(FA_aseg.shape)

    test = scipy.io.loadmat('Connectome_group_aparc+aseg_length.mat')
    length_aseg = np.array(test['connectome'])
    # print(length_aseg.shape)

    test = scipy.io.loadmat('Connectome_group_aparc+aseg_MD.mat')
    MD_aseg = np.array(test['connectome'])
    # print(MD_aseg.shape)

    test = scipy.io.loadmat('Connectome_group_aparc+aseg_RD.mat')
    RD_aseg = np.array(test['connectome'])
    # print(RD_aseg.shape)

    zero_AD = np.zeros([164, 164, 303])
    zero_count = np.zeros([164, 164, 303])
    zero_FA = np.zeros([164, 164, 303])
    zero_length = np.zeros([164, 164, 303])
    zero_MD = np.zeros([164, 164, 303])
    zero_RD = np.zeros([164, 164, 303])

    zero_AD[40:124, 40:124, :] = AD_aseg
    zero_count[40:124, 40:124, :] = count_aseg
    zero_FA[40:124, 40:124, :] = FA_aseg
    zero_length[40:124, 40:124, :] = length_aseg
    zero_MD[40:124, 40:124, :] = MD_aseg
    zero_RD[40:124, 40:124, :] = RD_aseg




    X = np.zeros([164, 164, 303, num_features])
    if num_features == 2:
        if atlas == 1:
            X[:, :, :, 0] = count_2009
            X[:, :, :, 1] = length_2009
        elif atlas == 2:
            X[:, :, :, 0] = zero_count
            X[:, :, :, 1] = zero_length
    else:
        X[:, :, :, 0] = AD_2009
        X[:, :, :, 1] = count_2009
        X[:, :, :, 2] = FA_2009
        X[:, :, :, 3] = length_2009
        X[:, :, :, 4] = MD_2009
        X[:, :, :, 5] = RD_2009
        X[:, :, :, 6] = zero_AD
        X[:, :, :, 7] = zero_count
        X[:, :, :, 8] = zero_FA
        X[:, :, :, 9] = zero_length
        X[:, :, :, 10] = zero_MD
        X[:, :, :, 11] = zero_RD

    # print(X.shape)
    X = X.transpose([2, 0, 1, 3])
    # print(X.shape)
    X_conn = np.reshape(X, (303, 164 * 164 * num_features))

    alldata = pd.read_csv('../data/Demographics/SER_MOR_136_v4.csv', header=0)
    print(alldata.shape)
    alldata = np.array(alldata)
    datasubjid = alldata[:, 0]
    list_subjs = pd.read_csv('../data/connectome/list_subject_303.csv', header=0)
    # list_subjs=list_subjs.apply(lambda x: x.str.slice(0,6))
    # print(list_subjs.shape)

    filtindex = np.isin(list_subjs, datasubjid)
    filtindex = filtindex.ravel()
    X_conn = X_conn[filtindex]
    # print(X_conn.shape)

    alldata = pd.read_csv('../data/Demographics/SER_MOR_136_v4.csv', header=0)
    filtindex = np.isin(datasubjid, list_subjs)
    filtindex = filtindex.ravel()
    Mor = alldata[filtindex]

    print(X_conn.shape)
    print(alldata.shape)
    print(filtindex.shape)

    demoInfo = pd.read_csv('../data/Demographics/SER_MOR_136_v4.csv', header=0)
    filtindex = np.isin(datasubjid, list_subjs)
    filtindex = filtindex.ravel()
    demoInfoDat = demoInfo[filtindex]

    #X_adFeat = np.array(demoInfoDat)[:,7:13]
    #print(X_adFeat.shape)

    cl_index = [7,8,9,10,11,12,13,14];
    import csv
    file = '../data/Demographics/SER_MOR_136_v4.csv'
    with open(file, 'r') as f:
        d_reader = csv.DictReader(f)
        headers = d_reader.fieldnames

    if atlas == 1:
        data = np.array(Mor.iloc[:, 388:1060], dtype=np.float32)
    elif atlas == 2:
        data = np.array(Mor.iloc[:, 44:388], dtype=np.float32)
    else:
        data = np.array(Mor.iloc[:, 44:1060], dtype=np.float32)
    print(Mor.shape)
    X = data

    if task == 1:
        if multi == 0:
            y = Mor.iloc[:, 1].values
        else:
            cov_list = [1,3,4]
            y = Mor.iloc[:, cov_list].values
    elif task == 2:
        y = Mor.iloc[:, 5].values
    y_temp = Mor.iloc[:, 1].values
    ind_num = pd.isnull(y_temp)
    y = y[~ind_num,:]
    y = y.astype(int)
   
    # cl_index=[2,7,8,25,1047,1048,1049];
    # free_index=
    w = np.array(Mor.iloc[~ind_num, cl_index])
    cll_index = [2, 4, 7, 8, 25];

    ww = np.array(Mor.iloc[~ind_num, cll_index])

    X = X[~ind_num, :]
    X_conn = X_conn[~ind_num, :]
    #X_adFeat = X_adFeat[~ind_num, :]
    X_conn = np.append(w, X_conn, axis=1)
    X_conn = stats.zscore(X_conn)
    print('Conn shape: {}'.format(X_conn.shape))

    # X=np.append(ww,X,axis=1)
    X = stats.zscore(X)
    #X_adFeat = stats.zscore(X_adFeat)
    print('mor shape: {}'.format(X.shape))
    #y = np.reshape(y, [-1, len(cov_list)])
    print('y shape:{}'.format(y.shape))

    np.isnan(X).any()
    X[np.isnan(X)] = np.median(X[~np.isnan(X)])
    # site = Mor.iloc[~ind_num, 4].values
    # HAMD = np.array(Mor.iloc[~ind_num, 7]) - np.array(Mor.iloc[~ind_num, 19])
    #
    # whole_index = np.append(cl_index, np.arange(30, len(headers), 1))
    # # headers[whole_index]
    # names = headers[30:]
    # # print(names)
    X_clinical = w
    # print(X_clinical)

    X_clinical[np.isnan(X_clinical)] = np.median(X_clinical[~np.isnan(X_clinical)])
    X_conn[np.isnan(X_conn)] = np.median(X_conn[~np.isnan(X_conn)])
    names = headers[30:]

    names.insert(0, 'sex')
    names.insert(1, 'site')
    names.insert(2, 'w0_17')
    names.insert(3, 'w0_24')
    names.insert(4, 'age')
    X_mor = X

    return X_mor, X_conn, X_clinical, y


def clf_randomforest(X,y,dataset,num_feat,atlas,multi):

    n_fold = 10

    if multi == 1:
        inner_cv = KFold(n_splits=n_fold, shuffle=False, random_state=234)
        outer_cv = KFold(n_splits=n_fold, shuffle=False, random_state=234)
    else:
        inner_cv = StratifiedKFold(n_splits=n_fold, shuffle=False, random_state=234)
        outer_cv = StratifiedKFold(n_splits=n_fold, shuffle=False, random_state=234)

    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    avg_acc = []
    avg_hloss = []
    avg_train_acc = []
    avg_pre = []
    avg_rec = []
    avg_f1 = []
    avg_TP = []
    avg_TN = []
    avg_FP = []
    avg_FN = []
    avg_sen = []
    avg_spec = []

    roc_label = []
    roc_pred = []
    roc_prob = []

    for train_index, test_index in outer_cv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # 'featureExtract__n_estimators': np.arange(10, 100, 10),
        params = {'randomforest__min_samples_leaf': np.arange(1, 51, 5),
                  'randomforest__n_estimators': np.arange(10, 100, 10)}
        # clf_m = RandomForestClassifier(random_state=0)

        pipe = Pipeline([
            ('featureExtract', SelectFromModel(ExtraTreesClassifier())),
            ('randomforest', RandomForestClassifier())
        ])

        clf = GridSearchCV(estimator=pipe, param_grid=params, cv=inner_cv, scoring='accuracy')
        clf.fit(X_train, y_train)

        fs = clf.best_estimator_.named_steps['featureExtract']
        mask = fs.get_support()
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)
      #  roc_pred = clf.predict_proba(X_test)

        if multi == 1:
            h_loss = hamming_loss(y_test, y_pred)
            pre_s = precision_score(y_test, y_pred, average='macro')
            rec_s = recall_score(y_test, y_pred, average='macro')
            f1_s = f1_score(y_test, y_pred, average='macro')

            avg_hloss = np.append(avg_hloss, h_loss)
            avg_pre = np.append(avg_pre, pre_s)
            avg_rec = np.append(avg_rec, rec_s)
            avg_f1 = np.append(avg_f1, f1_s)

            print('Precision:{}'.format(pre_s))
            print('Recall:{}'.format(rec_s))
            print('F1 score:{}'.format(f1_s))
            print('Hamming loss:{}'.format(h_loss))

        else:
            y_pred_train = clf.predict(X_train)
            acc = accuracy_score(y_test, y_pred)
            train_acc = accuracy_score(y_train,y_pred_train)
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
            avg_train_acc = np.append(avg_train_acc,train_acc)
            print(TP, FP, FN, TN)
            sen = TP / (TP + FN)
            spec = TN / (TN + FP)

            avg_sen = np.append(avg_sen, sen)
            avg_spec = np.append(avg_spec, spec)
            print('Accuracy:{},AUC:{}'.format(acc, auc))
            print('Train Accuracy:{}'.format(train_acc))
            print('ground truth:{}'.format(y_test))
            print('predicted:{}'.format(y_pred))
            print('Sensitivity:{},Specificity:{}'.format(sen, spec))

    if multi == 1:
        print('Avg. Precision:{},std:{}'.format(np.mean(avg_pre),np.std(avg_pre)))
        print('Avg. Recall:{},std:{}'.format(np.mean(avg_rec),np.std(avg_rec)))
        print('Avg. F1 score:{},std:{}'.format(np.mean(avg_f1),np.std(avg_f1)))
        print('Avg. Hamming loss:{},std:{}'.format(np.mean(avg_hloss),np.std(avg_hloss)))
    else:
        print("Accuracy Avg: {}".format(np.mean(avg_acc)))
        print("Train Accuracy Avg: {}".format(np.mean(avg_train_acc)))
        print("Accuracy Standard Deviation: {}".format(np.std(avg_acc)))
        print("Sensitivity Avg: {}".format(np.mean(avg_sen)))
        print("Sensitivity Standard Deviation: {}".format(np.std(avg_sen)))
        print("Specificity Avg: {}".format(np.mean(avg_spec)))
        print("Specificity Standard Deviation: {}".format(np.std(avg_spec)))

    if dataset == 1:
        pickle.dump(roc_label, open('roc_label_con_'+str(multi)+'-'+num_feat+'_'+atlas+'.p', "wb"))
        pickle.dump(roc_pred, open('roc_pred_con_'+str(multi)+'-'+num_feat+'_'+atlas+'.p', "wb"))
        pickle.dump(roc_prob, open('roc_prob_con_'+str(multi)+'-'+num_feat+'_'+atlas+'.p', "wb"))
    elif dataset == 2:
        pickle.dump(roc_label, open('roc_label_con_morph_'+str(multi)+'-'+num_feat+'_'+atlas+'.p', "wb"))
        pickle.dump(roc_pred, open('roc_pred_con_morph_'+str(multi)+'-'+num_feat+'_'+atlas+'.p', "wb"))
        pickle.dump(roc_prob, open('roc_prob_con_morph_'+str(multi)+'-'+num_feat+'_'+atlas+'.p', "wb"))
    elif dataset == 3:
        pickle.dump(roc_label, open('roc_label_morph_'+str(multi)+'-'+num_feat+'_'+atlas+'.p', "wb"))
        pickle.dump(roc_pred, open('roc_pred_morph_'+str(multi)+'-'+num_feat+'_'+atlas+'.p', "wb"))
        pickle.dump(roc_prob, open('roc_prob_morph_'+str(multi)+'-'+num_feat+'_'+atlas+'.p', "wb"))

def rg_randomforest(X,y,dataset,num_feat,atlas,):

    n_fold = 10

    inner_cv = KFold(n_splits=n_fold, shuffle=False, random_state=234)
    outer_cv = KFold(n_splits=n_fold, shuffle=False, random_state=234)

    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    avg_mse = []
    avg_TP = []
    avg_TN = []
    avg_FP = []
    avg_FN = []
    avg_sen = []
    avg_spec = []

    roc_label = []
    roc_pred = []
    roc_prob = []
    for train_index, test_index in outer_cv.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # 'featureExtract__n_estimators': np.arange(10, 100, 10),
        params = {'randomforest__min_samples_leaf': np.arange(1, 51, 5),
                  'randomforest__n_estimators': np.arange(10, 100, 10)}
        # clf_m = RandomForestClassifier(random_state=0)

        pipe = Pipeline([
            ('featureExtract', SelectFromModel(ExtraTreesClassifier())),
            ('randomforest', RandomForestRegressor())
        ])

        clf = GridSearchCV(estimator=pipe, param_grid=params, cv=inner_cv, scoring='neg_mean_squared_error')
        clf.fit(X_train, y_train)

        fs = clf.best_estimator_.named_steps['featureExtract']
        mask = fs.get_support()
        y_pred = clf.predict(X_test)
        #y_prob = clf.predict_proba(X_test)
      #  roc_pred = clf.predict_proba(X_test)

        mse = mean_squared_error(y_test, y_pred)
        #acc = accuracy_score(y_test, y_pred)
        #auc = roc_auc_score(y_test, y_prob[:, 1])

        roc_label = np.append(roc_label, y_test)
        roc_pred = np.append(roc_pred, y_pred)
        #roc_prob = np.append(roc_prob, y_prob[:, 1])


        # conf_mat = confusion_matrix(y_test, y_pred)
        #
        # TP = conf_mat[0][0]
        # FP = conf_mat[0][1]
        # FN = conf_mat[1][0]
        # TN = conf_mat[1][1]
        #
        # avg_TP = np.append(avg_TP, TP)
        # avg_TN = np.append(avg_TN, TN)
        # avg_FP = np.append(avg_FP, FP)
        # avg_FN = np.append(avg_FN, FN)
        #
        # avg_acc = np.append(avg_acc, acc)
        #
        # print(TP, FP, FN, TN)
        # sen = TP / (TP + FN)
        # spec = TN / (TN + FP)
        #
        # avg_sen = np.append(avg_sen, sen)
        # avg_spec = np.append(avg_spec, spec)
        avg_mse = np.append(avg_mse,mse)
        print('MSE:{}'.format(mse))
        #print('Sensitivity:{},Specificity:{}'.format(sen, spec))

    print("MSE Avg: {}".format(np.mean(avg_mse)))
    print("MSE Standard Deviation: {}".format(np.std(avg_mse)))
    # print("Sensitivity Avg: {}".format(np.mean(avg_sen)))
    # print("Sensitivity Standard Deviation: {}".format(np.std(avg_sen)))
    # print("Specificity Avg: {}".format(np.mean(avg_spec)))
    # print("Specificity Standard Deviation: {}".format(np.std(avg_spec)))

    if dataset == 1:
        pickle.dump(roc_label, open('roc_label_con_'+num_feat+'_'+atlas+'.p', "wb"))
        pickle.dump(roc_pred, open('roc_pred_con_'+num_feat+'_'+atlas+'.p', "wb"))

        pickle.dump(fs, open('roc_featEx_con_' + num_feat + '_' + atlas + '.p', "wb"))
        #pickle.dump( clf.feature_importances_, open('roc_featIm_con_' + num_feat + '_' + atlas + '.p', "wb"))


        #pickle.dump(roc_prob, open('roc_prob_con_'+num_feat+'_'+atlas+'.p', "wb"))
    elif dataset == 2:
        pickle.dump(roc_label, open('roc_label_con_morph_'+num_feat+'_'+atlas+'.p', "wb"))
        pickle.dump(roc_pred, open('roc_pred_con_morph_'+num_feat+'_'+atlas+'.p', "wb"))
        pickle.dump(fs, open('roc_featEx_con_morph_' + num_feat + '_' + atlas + '.p', "wb"))
        pickle.dump(clf.feature_importances_, open('roc_featIm_con_morph_' + num_feat + '_' + atlas + '.p', "wb"))

        #pickle.dump(roc_prob, open('roc_prob_con_morph_'+num_feat+'_'+atlas+'.p', "wb"))
    elif dataset == 3:
        pickle.dump(roc_label, open('roc_label_morph_'+num_feat+'_'+atlas+'.p', "wb"))
        pickle.dump(roc_pred, open('roc_pred_morph_'+num_feat+'_'+atlas+'.p', "wb"))

        pickle.dump(fs, open('roc_featEx_morph_' + num_feat + '_' + atlas + '.p', "wb"))
        pickle.dump(clf.feature_importances_, open('roc_featIm_morph_' + num_feat + '_' + atlas + '.p', "wb"))

        #pickle.dump(roc_prob, open('roc_prob_morph_'+num_feat+'_'+atlas+'.p', "wb"))


def main():
    parser = argumentparser()
    args = parser.parse_args()

    cwd=os.getcwd()
    os.chdir('../data')

    X_mor,X_conn,X_clinical,y=data_fetch_clean(args.atlas, args.num_feat,args.task,args.multilabel)
    y = y.reshape(-1)

    #5: ad-smi / 6:mci-smi / 7:adonly-smi / 8:ad-mci / 9:adonly-mci / 10:adonly - adwithsmallvv
    X_conn_mor=np.concatenate([X_conn,X_mor],axis=1)

    if args.task == 1:
        if args.dataset == 1:
            clf_randomforest(X_conn,y,args.dataset,str(args.num_feat),str(args.atlas),args.multilabel)
        elif args.dataset == 2:
            clf_randomforest(X_conn_mor, y, args.dataset,str(args.num_feat),str(args.atlas),args.multilabel)
        elif args.dataset == 3:
            clf_randomforest(X_mor, y, args.dataset, str(args.num_feat), str(args.atlas),args.multilabel)
    elif args.task == 2:
        if args.dataset == 1:
            rg_randomforest(X_conn,y,args.dataset,str(args.num_feat),str(args.atlas))
        elif args.dataset == 2:
            rg_randomforest(X_conn_mor, y, args.dataset,str(args.num_feat),str(args.atlas))

if __name__ == '__main__':
    main()
