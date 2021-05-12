import os
import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
from PyEMD import EMD, Visualisation
from argparse import ArgumentParser

from src.util.detect_peaks import detect_peaks
from src.util.filters import butterworth, QVR, median_filter,detrend ,EMD_baseline_removal  
from src.util.signal_alignment import phase_align, chisqr_align, RMSE_align
from src.util.PPI import RRI, PPI
from src.util.read_file import parse_files, parse_resting_files
from scipy.ndimage.interpolation import shift

from biosppy import storage
from biosppy.signals import ecg
from biosppy.signals import bvp

def check_alignment_3sig(ppi_1, ppi_2, ppi_3, loc_1, loc_2, loc_3, show=False):


    #s1 = phase_align(ppi_1, ppi_2, [10,50])
    s1 = RMSE_align(ppi_1[:60], ppi_2[:60], [10,30])
    if np.abs(s1) >= 9 :
        s1 = phase_align(ppi_1, ppi_2, [10,50])
    # print('Phase shift of 1&2 value to align is',s1)


    # make some diagnostic plots
    if s1>=0:
        ppi_1 =ppi_1[int(round(s1)):]

        loc_1 =loc_1[int(round(s1)):]
    else:
        ppi_2 = ppi_2[int(round(-s1)):]

        loc_2 = loc_2[int(round(-s1)):]

    #s2 = phase_align(ppi_2, ppi_3, [10,50])
    s2 = RMSE_align(ppi_2[:60], ppi_3[:60], [10,30])
    if np.abs(s2) >= 9 :
        s2 = phase_align(ppi_2, ppi_3, [10,50])
        if np.abs(s2) >= 9:
            s2 = 0
    # print('Phase shift of 2&3 value to align is',s2)

    if s2>=0:
        ppi_1 =ppi_1[int(round(s2)):]
        ppi_2 =ppi_2[int(round(s2)):]

        loc_1 =loc_1[int(round(s2)):]
        loc_2 =loc_2[int(round(s2)):]
    else:
        ppi_3 = ppi_3[int(round(-s2)):]

        loc_3 = loc_3[int(round(-s2)):]

    #Truncate to same length
    min_length = min([len(ppi_1), len(ppi_2), len(ppi_3)])
    ppi_1 = ppi_1[:min_length]
    ppi_2 = ppi_2[:min_length]
    ppi_3 = ppi_3[:min_length]

    min_length = min([len(loc_1), len(loc_2), len(loc_3)])
    loc_1 = loc_1[:min_length]
    loc_2 = loc_2[:min_length]
    loc_3 = loc_3[:min_length]


    if show == True:
        f = plt.figure(figsize=(140,7))
        plt.plot(ppi_1,label='RRI')
        plt.plot(ppi_2,label='PPI_inf')
        plt.plot(ppi_3,label='PPI_pix')
        plt.legend(loc='best')
        if not os.path.exists('images/'):
            os.makedirs('images/')
        plt.savefig('images/inf_pix_ppg_ppi_alignment.png', format='png', dpi=150)
        plt.show()
    
    return ppi_1, ppi_2, ppi_3, loc_1, loc_2, loc_3, s1, s2


def peak_detection(src_path = "./MAUS/Data/Raw_data/", dst_path='./MAUS/Data/', single_subject = False, subject_id = "002", extract_resting = False):

    # Parse files
    infiniti_data_e, infiniti_data_p, pixart_data = parse_files(src_path)
    infiniti_data_e_resting, infiniti_data_p_resting, pixart_data_resting = parse_resting_files(src_path)

    sub_id = ["002", "003", "004", "005", "006", "008", "010", "011", "012", "013", "014", \
    "015", "016", "017", "018", "019", "020", "021", "022", "023", "024", "025"]
    
    EMD_ratio = 0.6

    subject_idx = sub_id.index(subject_id)

    if single_subject:
        sub_str_pos = subject_idx
        sub_end_pos = subject_idx+1
    else:
        sub_str_pos = 0
        sub_end_pos = len(sub_id)

    for sub in range(sub_str_pos, sub_end_pos):
        print(np.asarray(infiniti_data_e_resting).shape)

        sub_name = sub_id[sub]

        if sub > 4:
            pix_fs = 102.5
        else:
            pix_fs = 100


        if(extract_resting):
            trial_num = 1
            # Resting
            ecg_inf = np.asarray(infiniti_data_e_resting[sub])
            ppg_inf = np.asarray(infiniti_data_p_resting[sub])
            ppg_pix = np.asarray(pixart_data_resting[sub])
        else:
            trial_num = 6
            # Trial: 0_back: 0,5; 2_back: 1,3; 3_back: 2,4
            if sub >= 1:
                sub_shift = sub + 1
            ecg_inf = np.asarray(infiniti_data_e[sub].values)
            ppg_inf = np.asarray(infiniti_data_p[sub].values)
            ppg_pix = np.asarray(pixart_data[sub].values)

        for trial in range(trial_num):


            if (extract_resting):
                ecg_inf_raw = ecg_inf
                ppg_inf_raw = ppg_inf
                ppg_pix_raw = ppg_pix

            else:
                ecg_inf_raw = ecg_inf[:,trial]
                ppg_inf_raw = ppg_inf[:,trial]
                ppg_pix_raw = ppg_pix[:,trial]


            for seg in range(3):
                # print("trial: ",trial, " seg: ", seg)    
                if seg == 0:
                    str_pos = 5
                    end_pos = 120
                elif seg == 1:
                    str_pos = 95
                    end_pos = 210
                elif seg == 2:
                    str_pos = 185
                    end_pos = 300
            
                # emd = EMD()
                # emd.emd(ecg_inf_raw[int(str_pos*256):int(end_pos*256)])
                # imfs_pix, res_pix = emd.get_imfs_and_residue()
                # sig_inf = np.sum(imfs_pix[:round(len(imfs_pix)*EMD_ratio):], axis = 0)
                sig_inf = ecg_inf_raw[int(str_pos*256):int(end_pos*256)]
                ppi_inf_ecg, peak_inf_ecg = RRI(sig_inf) 
                # f = plt.figure(figsize=(20,4))
                # plt.plot(sig_inf)
                # plt.scatter(peak_inf_ecg, sig_inf[peak_inf_ecg], c = 'r')
                # plt.show()
                mpd_thres = np.min(ppi_inf_ecg)//10 - 4
                # if np.min(ppi_inf_ecg) < 450:
                #     mpd_thres = 55
                mpd_thres_pix = mpd_thres
                mpd_thres_inf = int(mpd_thres*2.56)

                emd = EMD()
                emd.emd(ppg_pix_raw[int(str_pos*pix_fs):int(end_pos*pix_fs)])
                imfs_pix, res_pix = emd.get_imfs_and_residue()
                sig_pix = np.sum(imfs_pix[:round(len(imfs_pix)*EMD_ratio):], axis = 0)
                sig_pix = median_filter(sig_pix)
                
                ppi_inf_ppg, peak_inf_ppg = PPI(ppg_inf_raw[str_pos*256:end_pos*256], sourse='infiniti', manual_mpd = True ,  m_mpd=mpd_thres_pix, onset=False, show=False)
                ppi_pix_ppg, peak_pix_ppg = PPI(sig_pix, sourse='pixart', manual_mpd = True,  m_mpd=mpd_thres_pix, onset=True, show=False)
                
                p1, p2, p3, l1, l2, l3, s1, s2 = check_alignment_3sig(ppi_inf_ecg, ppi_inf_ppg, ppi_pix_ppg, peak_inf_ecg, peak_inf_ppg, peak_pix_ppg)
                
                
                # print("shift1: ", s1)
                # print("shift2: ", s2)
                # f = plt.figure(figsize=(20,5))
                # plt.plot(p1[:])
                # plt.plot(p2[:])
                # plt.plot(p3[:])
                
                dir_name = "IBI_seq/" + sub_name 
                if not os.path.exists(dst_path+dir_name):
                    os.makedirs(dst_path+dir_name)
                
                label = np.zeros((len(p3)))
                f0 = np.hstack((p1.reshape(-1, 1),p2.reshape(-1, 1),p3.reshape(-1, 1),label.reshape(-1, 1)))
                df_0 = pd.DataFrame(f0,columns=["RRI_inf","PPI_inf","PPI_pix","label"])
                
                
                pix_ppg_ppi = np.insert(p3,0,0)
                f1 = np.hstack((l3.reshape(-1, 1),pix_ppg_ppi.reshape(-1, 1)))
                df_1 = pd.DataFrame(f1,columns=["valley_location","PPI"])

                if(extract_resting):
                    df_0.to_csv(dst_path+dir_name + "/rest_"+ str(seg+1) + ".csv", index=False)
                    df_1.to_csv(dst_path+dir_name + "/rest_"+ str(seg+1) + "_peak.csv", index=False)
                else:
                    df_0.to_csv(dst_path+dir_name + "/trial_"+ str(trial+1) + "_" + str(seg+1) + ".csv", index=False)
                    df_1.to_csv(dst_path+dir_name + "/trial_"+ str(trial+1) + "_" + str(seg+1) + "_peak.csv", index=False)
            



if __name__ == "__main__":

	#parse argument
    parser = ArgumentParser(
        description="Mental Workload N-backs Dataset -- IBI extraction")
    parser.add_argument('--src_data', help="raw signal datapath", type=str, default="./MAUS/Data/Raw_data/")
    parser.add_argument('--dst_data', help="extract IBI sequence datapath", type=str, default="./MAUS/Data/")
    parser.add_argument('--single_sub', help="extract from single subject", type=bool, default=True)
    parser.add_argument('--sub_index', help="single subject ID", type=str, default="002")
    parser.add_argument('--rest_data', help="extract resting data", type=bool, default=False)
    args = parser.parse_args()
    peak_detection(src_path = args.src_data, dst_path=args.dst_data, single_subject = args.single_sub, subject_id = args.sub_index, extract_resting = args.rest_data)
