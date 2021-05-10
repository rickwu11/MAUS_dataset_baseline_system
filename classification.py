#LOSO
from argparse import ArgumentParser
import os
import numpy as np
import scipy
import pandas as pd
import pickle
import math
from scipy import stats
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import random
random.seed(9)
from sklearn.utils import shuffle

from HRV_feature_extraction import feat_read_from_pkl, feat_normalization

def classification_LOSO(feat_inf_e_norm, feat_inf_p_norm, feat_pix_norm, label, obj_position):

    from warnings import filterwarnings
    filterwarnings('ignore')

    model1 = []
    model2 = []
    model3 = []

    Train_acc_inf_e = []
    Train_acc_inf_p = []
    Train_acc_pix = []

    Train_f1_inf_e = []
    Train_f1_inf_p = []
    Train_f1_pix = []

    ave_acc_inf_e = []
    ave_acc_inf_p = []
    ave_acc_pix = []
    ave_f1_inf_e = []
    ave_f1_inf_p = []
    ave_f1_pix = []
    cm1 = np.zeros((2,2))
    cm2 = np.zeros((2,2))
    cm3 = np.zeros((2,2))
    
#     cm1 = np.zeros((3,3))
#     cm2 = np.zeros((3,3))
#     cm3 = np.zeros((3,3))

    subject_num = len(obj_position)-1

    train_round = subject_num

    screen_num = 0

    for i in range(train_round):
        str_pos = obj_position[i]
        end_pos = obj_position[i+1]
        
        feat_inf_e_train = np.delete(feat_inf_e_norm,  np.s_[str_pos:end_pos] ,0)
        # print(feat_inf_e_train.shape)
        feat_inf_e_test = feat_inf_e_norm[str_pos:end_pos,:]
        # print(feat_inf_e_test.shape)
        
        feat_inf_p_train = np.delete(feat_inf_p_norm,  np.s_[str_pos:end_pos] ,0)
        feat_inf_p_test = feat_inf_p_norm[str_pos:end_pos,:]
        
        feat_pix_train = np.delete(feat_pix_norm,  np.s_[str_pos:end_pos] ,0)
        feat_pix_test = feat_pix_norm[str_pos:end_pos,:]
        
        label_train =  np.delete(label, np.s_[str_pos:end_pos])
        label_test = label[str_pos:end_pos]

        
        # infiniti ecg
        # xgb_model = Pipeline([('scale', StandardScaler()),
                   # ('SVC',LinearSVC(C = 20.001, class_weight = { 0:0.67, 1:0.33 }, random_state = 9))])
        xgb_model = LinearSVC(C= 20.001, class_weight = { 0:0.67, 1:0.33 }, random_state = 9, verbose = False)

        xgb_model.fit(feat_inf_e_train, label_train)

        y1_pred = xgb_model.predict(feat_inf_e_train)
        Train_acc_inf_e.append(accuracy_score(label_train, y1_pred))
        Train_f1_inf_e.append(f1_score(label_train, y1_pred, average='weighted'))
        # print('Train inf ecg  acc',accuracy_score(label_train, y1_pred))
        y1_pred = xgb_model.predict(feat_inf_e_test)
        acc_inf_e =  accuracy_score(label_test, y1_pred)
        cm1 += confusion_matrix(label_test, y1_pred)
        model1.append(xgb_model)

        # print('Test inf ecg acc',acc_inf_e)
        ave_acc_inf_e.append(acc_inf_e)
        
        inf_e_f1 = f1_score(label_test, y1_pred, average='weighted')
        # print('inf_f1',inf_e_f1)
        ave_f1_inf_e.append(inf_e_f1)
        
       
        
        # infiniti ppg
        # xgb_model2 = Pipeline([('scale', StandardScaler()),
                   # ('SVC',LinearSVC(C = 0.001, class_weight = { 0:0.67, 1:0.33 }, random_state = 9))])
        xgb_model2 = LinearSVC(C= 0.001, class_weight = { 0:0.67, 1:0.33 }, random_state = 9, verbose = False)
        xgb_model2.fit(feat_inf_p_train, label_train)
        
        y1_1_pred = xgb_model2.predict(feat_inf_p_train)
        Train_acc_inf_p.append(accuracy_score(label_train, y1_1_pred))
        Train_f1_inf_p.append(f1_score(label_train, y1_1_pred, average='weighted'))
        # print('Train inf ppg acc',accuracy_score(label_train, y1_1_pred))
        y1_1_pred = xgb_model2.predict(feat_inf_p_test)
        acc_inf_p =  accuracy_score(label_test, y1_1_pred)
        cm2 += confusion_matrix(label_test, y1_1_pred)
        model2.append(xgb_model2)

        # print('Test inf ppg acc',acc_inf_p)
        ave_acc_inf_p.append(acc_inf_p)
        
        inf_p_f1 = f1_score(label_test, y1_1_pred, average='macro')
        # print('inf_f1',inf_p_f1)
        ave_f1_inf_p.append(inf_p_f1)


        # pixart ppg
        xgb_model3 = LinearSVC(C= 0.01, class_weight = { 0:0.67, 1:0.33 }, random_state = 9, verbose = False)
    
        xgb_model3.fit(feat_pix_train, label_train)

        y2_pred = xgb_model3.predict(feat_pix_train)
        Train_acc_pix.append(accuracy_score(label_train, y2_pred))
        Train_f1_pix.append(f1_score(label_train, y2_pred, average='macro'))
        # print('Train pix acc',accuracy_score(label_train, y2_pred))
        y2_pred = xgb_model3.predict(feat_pix_test)
        acc_pix = accuracy_score(label_test, y2_pred)
        cm3 += confusion_matrix(label_test, y2_pred)
        model3.append(xgb_model3)

        # print('Test pix acc',acc_pix)
        ave_acc_pix.append(acc_pix)
        pix_f1 = f1_score(label_test, y2_pred, average='macro')
        # print('pix_f1',pix_f1)
        ave_f1_pix.append(pix_f1)

        # print()
     
    print()     
    print('Average infinity ecg acc: ', np.mean(ave_acc_inf_e))    
    print('Std infinity ecg acc: ', np.std(ave_acc_inf_e))
    print()  
    print('Average infinity ppg acc: ', np.mean(ave_acc_inf_p))  
    print('Std infinity ppg acc: ', np.std(ave_acc_inf_p))  
    print()  
    print('Average pixart ppg acc: ', np.mean(ave_acc_pix))
    print('Std pixart ppg acc: ', np.std(ave_acc_pix))
    print()  

    print('Average infinity ecg f1 : ',np.mean(ave_f1_inf_e))   
    print('Std infinity ecg f1: ', np.std(ave_f1_inf_e)) 
    print()  
    print('Average infinity ppg f1 : ', np.mean(ave_f1_inf_p))   
    print('Std infinity ppg f1: ', np.std(ave_f1_inf_p))  
    print() 
    print('Average pixart ppg f1 : ', np.mean(ave_f1_pix))
    print('Std pixart f1 acc: ', np.std(ave_f1_pix))
    print()  

#     plot_confusion_matrix(cm1, 'cm1', normalize=True)
#     plot_confusion_matrix(cm2, 'cm2', normalize=True)
#     plot_confusion_matrix(cm3, 'cm3', normalize=True)

    return Train_f1_inf_e, Train_f1_inf_p, Train_f1_pix, ave_f1_inf_e, ave_f1_inf_p, ave_f1_pix, model1,model2,model3

def classification_Mixed(feat_inf_e_norm, feat_inf_p_norm, feat_pix_norm, label, obj_position):

    from warnings import filterwarnings
    filterwarnings('ignore')


    ave_acc_inf_e = []
    ave_acc_inf_p = []
    ave_acc_pix = []
    ave_f1_inf_e = []
    ave_f1_inf_p = []
    ave_f1_pix = []

    for i in range(50):

        kf = KFold(n_splits=5, shuffle=True, random_state = i )
        kf.get_n_splits(feat_inf_e_norm)


        for train_index, test_index in kf.split(feat_inf_e_norm):
        #     print("TRAIN:", train_index, "TEST:", test_index)
            feat_inf_e_train, feat_inf_e_test = feat_inf_e_norm[train_index], feat_inf_e_norm[test_index]
            feat_inf_p_train, feat_inf_p_test = feat_inf_p_norm[train_index], feat_inf_p_norm[test_index]
            feat_pix_train, feat_pix_test = feat_pix_norm[train_index], feat_pix_norm[test_index]

            label_train, label_test = label[train_index], label[test_index]

            xgb_model = Pipeline([('scale', StandardScaler()),
    #                    ('SVC',LinearSVC(C = 0.001, class_weight = { 0:0.67, 1:0.33 }, random_state = 9))])
                        ('SVC',LinearSVC(C = 25.001, class_weight = { 0:0.67, 1:0.33 }, random_state = 9))])

            xgb_model.fit(feat_inf_e_train, label_train)


            xgb_model_2 = Pipeline([('scale', StandardScaler()),
                       ('SVC',LinearSVC(C = 0.001, class_weight = { 0:0.67, 1:0.33 }, random_state = 9))])
                        # ('SVC',LinearSVC(C = 25.001, class_weight = { 0:0.67, 1:0.33 }, random_state = 9))])

            xgb_model_2.fit(feat_inf_p_train, label_train)

            xgb_model_3 = Pipeline([('scale', StandardScaler()),
                       ('SVC',LinearSVC(C = 0.001, class_weight = { 0:0.67, 1:0.33 }, random_state = 9))])
                        # ('SVC',LinearSVC(C = 25.001, class_weight = { 0:0.67, 1:0.33 }, random_state = 9))])

            xgb_model_3.fit(feat_pix_train, label_train)

            y1_pred = xgb_model.predict(feat_inf_e_test)
            acc_inf_e =  accuracy_score(label_test, y1_pred)
            # print('Test inf ecg acc',acc_inf_e)
            ave_acc_inf_e.append(acc_inf_e)
            inf_e_f1 = f1_score(label_test, y1_pred, average='macro')
            # print('inf_f1',inf_e_f1)
            ave_f1_inf_e.append(inf_e_f1)

            y1_pred2 = xgb_model_2.predict(feat_inf_p_test)
            acc_inf_p =  accuracy_score(label_test, y1_pred2)
            # print('Test inf ecg acc',acc_inf_p)
            ave_acc_inf_p.append(acc_inf_p)
            inf_p_f1 = f1_score(label_test, y1_pred2, average='macro')
            # print('inf_f1',inf_p_f1)
            ave_f1_inf_p.append(inf_p_f1)

            y1_pred3 = xgb_model_3.predict(feat_pix_test)
            acc_pix =  accuracy_score(label_test, y1_pred3)
            # print('Test inf ecg acc',acc_pix)
            ave_acc_pix.append(acc_pix)
            pix_f1 = f1_score(label_test, y1_pred3, average='macro')
            # print('inf_f1',pix_f1)
            ave_f1_pix.append(pix_f1)
        

    print()     
    print('Average infinity ecg acc: ', np.mean(ave_acc_inf_e))    
    print('Std infinity ecg acc: ', np.std(ave_acc_inf_e))
    print()  
    print('Average infinity ppg acc: ', np.mean(ave_acc_inf_p))  
    print('Std infinity ppg acc: ', np.std(ave_acc_inf_p))  
    print()  
    print('Average pixart ppg acc: ', np.mean(ave_acc_pix))
    print('Std pixart ppg acc: ', np.std(ave_acc_pix))
    print()  

    print('Average infinity ecg f1 : ',np.mean(ave_f1_inf_e))   
    print('Std infinity ecg f1: ', np.std(ave_f1_inf_e)) 
    print()  
    print('Average infinity ppg f1 : ', np.mean(ave_f1_inf_p))   
    print('Std infinity ppg f1: ', np.std(ave_f1_inf_p))  
    print() 
    print('Average pixart ppg f1 : ', np.mean(ave_f1_pix))
    print('Std pixart f1 acc: ', np.std(ave_f1_pix))
    print()  


if __name__ == "__main__":

    #parse argument
    parser = ArgumentParser(
        description='Mental Workload N-backs Dataset -- classification')
    parser.add_argument('--data', type=str, default='./feature_data/')
    parser.add_argument('--mode', type=str, default='LOSO')
    args = parser.parse_args()

    feat_inf_e, feat_inf_p, feat_pix, label, obj_position = feat_read_from_pkl(args.data)
    feat_inf_e_norm, feat_inf_p_norm, feat_pix_norm = feat_normalization(feat_inf_e, feat_inf_p, feat_pix, label, obj_position)
    # print(feat_inf_e.shape)
    # print(feat_inf_e_norm.shape)
    label[label == 2] = 1
    label[label == 3] = 1  
    if args.mode == 'LOSO':
        classification_LOSO(feat_inf_e_norm[:,:], feat_inf_p_norm[:,:], feat_pix_norm[:,:], label, obj_position)
    else:
        classification_Mixed(feat_inf_e_norm[:,:], feat_inf_p_norm[:,:], feat_pix_norm[:,:], label, obj_position)

