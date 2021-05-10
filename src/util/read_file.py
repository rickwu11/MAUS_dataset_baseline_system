"""
    Author          : Rick
    File Description: This file is use to parse INFINITI Data, PixArt Data
    Date            : 20191002
"""
import os
import pandas as pd
import numpy as np

def parse_files(data_path = './data/'):

    infiniti_e = None
    infiniti_p = None
    infiniti_data_e = []
    infiniti_data_p = []
    pixart = None
    pixart_data = []
    for case_file in sorted(os.listdir(data_path)):
        print(case_file)
        infiniti_part_e = pd.read_csv( data_path + case_file + '/inf_ecg.csv', 
                                header=0, 
                                index_col=False,
                                usecols=['Trial 1:0back', 'Trial 2:2back', 'Trial 3:3back','Trial 4:2back', 'Trial 5:3back', 'Trial 6:0back'],
                                error_bad_lines=False)
        
        infiniti_part_p = pd.read_csv( data_path + case_file + '/inf_ppg.csv', 
                                header=0, 
                                index_col=False,
                                usecols=['Trial 1:0back', 'Trial 2:2back', 'Trial 3:3back','Trial 4:2back', 'Trial 5:3back', 'Trial 6:0back'],
                                error_bad_lines=False)
        
        pixart_part = pd.read_csv( data_path + case_file + '/pixart.csv', 
                                header=0, 
                                index_col=False,
                                usecols=['Trial 1:0back', 'Trial 2:2back', 'Trial 3:3back','Trial 4:2back', 'Trial 5:3back', 'Trial 6:0back'],
                                error_bad_lines=False)
        
        infiniti_part_e.columns = ['one_zero', 'two_two', 'three_three', 'four_two', 'five_three', 'six_zero']
        infiniti_part_p.columns = ['one_zero', 'two_two', 'three_three', 'four_two', 'five_three', 'six_zero']
        pixart_part.columns = ['one_zero', 'two_two', 'three_three', 'four_two', 'five_three', 'six_zero']
        
        infiniti_e = infiniti_part_e.copy()
        infiniti_data_e.append(infiniti_e)
        
        infiniti_p = infiniti_part_p.copy()
        infiniti_data_p.append(infiniti_p)

        pixart = pixart_part.copy()
        pixart_data.append(pixart)

    return infiniti_data_e, infiniti_data_p, pixart_data

def parse_resting_files(data_path = './data/'):
    infiniti_e_rest = []
    infiniti_p_rest = []
    pix_rest = []
    for case_file in sorted(os.listdir(data_path)):
        print(case_file)
        if case_file == '003'  :
            continue
        infiniti_data_e_rest = pd.read_csv( data_path + case_file + '/inf_resting.csv')
        pixi_data_p_rest = pd.read_csv( data_path + case_file + '/pixart_resting.csv')
        #print(infiniti_data_e_rest['Resting_ECG'].values)
        infiniti_e = infiniti_data_e_rest.copy()
        infiniti_e_rest.append( (np.asarray(infiniti_e['Resting_ECG'])))
        infiniti_p_rest.append( (np.asarray(infiniti_e['Resting_PPG'])))
        pixart = pixi_data_p_rest.copy()
        #print(pixi_data_p_rest['Resting'].values)
        pix_rest.append((np.asarray(pixart['Resting'])))

    return infiniti_e_rest, infiniti_p_rest, pix_rest

# if __name__ == "__main__":
#     pass
