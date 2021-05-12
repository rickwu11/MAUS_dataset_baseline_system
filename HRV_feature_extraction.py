"""
    Author          : Rick
    File Description: Data preprocessing and feature extraction. Save file to pickle files.
    Date            : 20191002
"""
from argparse import ArgumentParser
import os
import numpy as np
import scipy
import pandas as pd
import pickle
import math
import scipy
import matplotlib.pyplot as plt

from biosppy import storage
from biosppy.signals import ecg
from biosppy.signals import bvp

# from check_alignment import *
from src.HRV.feature_set import hrv_feature
from src.util.detect_peaks import detect_peaks
from src.util.signal_alignment import phase_align, chisqr_align

from scipy.ndimage.interpolation import shift
from warnings import filterwarnings
filterwarnings('ignore')




# Read extracted feature from pickle file

def feat_read_from_pkl(dir_name = './feature_data/'):
    with open(dir_name + 'feat_inf_ecg.pkl','rb') as f:
        feat_inf_e = pickle.load(f)
    with open(dir_name + 'feat_inf_ppg.pkl','rb') as f1:
        feat_inf_p = pickle.load(f1)
    with open(dir_name + 'feat_pix_ppg.pkl','rb') as f2:
        feat_pix = pickle.load(f2)
    with open(dir_name + 'label.pkl','rb') as f3:
        label = pickle.load(f3)
    with open(dir_name + 'obj_position.pkl','rb') as f4:
        obj_position = pickle.load(f4)

    feat_inf_e = np.asarray(feat_inf_e)
    feat_inf_p = np.asarray(feat_inf_p)
    feat_pix = np.asarray(feat_pix)
    label = np.asarray(label)
    obj_position = np.asarray(obj_position)
        
    # print(feat_inf_e.shape)
    # print(feat_pix.shape)
    # print(obj_position)

    return feat_inf_e, feat_inf_p, feat_pix, label, obj_position

# Normalize the feature by z score
def feat_normalization(feat_inf_e, feat_inf_p, feat_pix, label, obj_position):
    train_round = len(obj_position)-1
    feat_inf_e_norm = feat_inf_e.copy()
    feat_inf_p_norm = feat_inf_p.copy()
    feat_pix_norm = feat_pix.copy()
    for i in range(train_round):
        str_pos = obj_position[i]
        end_pos = obj_position[i+1]

        feat_inf_e_0 = feat_inf_p[str_pos:end_pos,:]
        feat_inf_p_0 = feat_inf_e[str_pos:end_pos,:]
        feat_pix_0 = feat_pix[str_pos:end_pos,:]
        
        feat_inf_e_0 = scipy.stats.mstats.zscore(feat_inf_e_0, axis = 0)
        feat_inf_p_0 = scipy.stats.mstats.zscore(feat_inf_p_0, axis = 0)
        feat_pix_0 = scipy.stats.mstats.zscore(feat_pix_0, axis = 0)
        
        feat_inf_e_norm[str_pos:end_pos,:] = feat_inf_e_0
        feat_inf_p_norm[str_pos:end_pos,:] = feat_inf_p_0
        feat_pix_norm[str_pos:end_pos,:] = feat_pix_0

    return feat_inf_e_norm, feat_inf_p_norm, feat_pix_norm


def feature_extraction(dir_name = "./MAUS/Data/IBI_sequence/"):

    cnt = 0
    feat_inf_e = []
    feat_inf_p = []
    feat_pix = []
    label = []
    obj_position = []
    sub_id = ["002", "003", "004", "005", "006", "008", "010", "011", "012", \
    "015", "016", "017", "018", "019", "021", "022", "023", "024", "025"]

    # for case_file in sorted(os.listdir(dir_name)):
    for case_file in sub_id:
        if case_file == "013" or case_file == "014" or case_file == "020":
            continue
        print(case_file)
        for trial in range(18):
            seg_idx = trial%3 + 1
            trial_idx = trial//3 + 1
            IBI = pd.read_csv(dir_name + case_file + "/trial_" + str(trial_idx) + "_" + str(seg_idx) + ".csv")
            RRI = np.asarray(IBI)[:,0]
            PPI_inf = np.asarray(IBI)[:,1]
            PPI = np.asarray(IBI)[:,2]

            ppi_std = np.std(RRI)

            if(cnt <= 4):
                pix_fs = 102.5
            else:
                pix_fs = 100

            SDNN, nn50, pnn50, rmssd, sdsd, tinn, tri_index, \
                    welch_TF, welch_LF, welch_HF, welch_LF_n, welch_HF_n, welch_LF_HF, \
                    lomb_TF, lomb_LF, lomb_HF, lomb_LF_n, lomb_HF_n, lomb_LF_HF, \
                    sd1, sd2, sd_ratio, sampen, dfa_alpha1, dfa_alpha2  = hrv_feature(RRI, sampling_rate = 256)


            feat_inf_e.append([SDNN, nn50, pnn50, rmssd, sdsd, tinn, tri_index, \
                    welch_TF, welch_LF, welch_HF, welch_LF_n, welch_HF_n, welch_LF_HF])

            SDNN, nn50, pnn50, rmssd, sdsd, tinn, tri_index, \
                    welch_TF, welch_LF, welch_HF, welch_LF_n, welch_HF_n, welch_LF_HF, \
                    lomb_TF, lomb_LF, lomb_HF, lomb_LF_n, lomb_HF_n, lomb_LF_HF, \
                    sd1, sd2, sd_ratio, sampen, dfa_alpha1, dfa_alpha2  = hrv_feature(PPI_inf, sampling_rate = 256)

            feat_inf_p.append([SDNN, nn50, pnn50, rmssd, sdsd, tinn, tri_index, \
                    welch_TF, welch_LF, welch_HF, welch_LF_n, welch_HF_n, welch_LF_HF])


            SDNN, nn50, pnn50, rmssd, sdsd, tinn, tri_index, \
                    welch_TF, welch_LF, welch_HF, welch_LF_n, welch_HF_n, welch_LF_HF, \
                    lomb_TF, lomb_LF, lomb_HF, lomb_LF_n, lomb_HF_n, lomb_LF_HF, \
                    sd1, sd2, sd_ratio, sampen, dfa_alpha1, dfa_alpha2  = hrv_feature(PPI, sampling_rate = pix_fs)

            feat_pix.append([SDNN, nn50, pnn50, rmssd, sdsd, tinn, tri_index, \
                    welch_TF, welch_LF, welch_HF, welch_LF_n, welch_HF_n, welch_LF_HF])

            if trial <= 2:
                label.append(0)  
            elif (trial > 2) and (trial <= 5):
                label.append(2) 
            elif (trial > 5) and (trial <= 8):
                label.append(3) 
            elif (trial > 8) and (trial <= 11):
                label.append(2) 
            elif (trial > 11) and (trial <= 14):
                label.append(3)
            elif (trial > 14) and (trial <= 17):
                label.append(0) 


        cnt = cnt + 1
        obj_position.append(18)

    obj_position = np.cumsum(obj_position)
    obj_position = np.insert(obj_position,0,0)
    feat_inf_e = np.asarray(feat_inf_e)
    feat_inf_p = np.asarray(feat_inf_p)
    feat_pix = np.asarray(feat_pix)
    label = np.asarray(label)
    print(obj_position)
    print('feat_dim',feat_inf_e.shape)

    dir_name = 'feature_data/'

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    with open( dir_name + 'feat_inf_ecg.pkl','wb') as f:
        pickle.dump(feat_inf_e, f)
    with open( dir_name + 'feat_inf_ppg.pkl','wb') as f:
        pickle.dump(feat_inf_p, f)
    with open( dir_name + 'feat_pix_ppg.pkl','wb') as f:
        pickle.dump(feat_pix, f)
    with open( dir_name + 'label.pkl','wb') as f:
        pickle.dump(label, f)
    with open( dir_name + 'obj_position.pkl','wb') as f:
        pickle.dump(obj_position, f)


if __name__ == "__main__":


    #parse argument
    parser = ArgumentParser(
        description='Mental Workload N-backs Dataset -- feature extraction')
    parser.add_argument('--data', help="IBI sequence datapath", type=str, default='./MAUS/Data/IBI_sequence/')
    args = parser.parse_args()

    feature_extraction(dir_name = args.data)
