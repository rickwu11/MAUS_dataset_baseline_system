"""
    Author          : Rick
    File Description: Feature Extraction
    Date            : 20191002
"""

import pyhrv
import pyhrv.time_domain as td
import pyhrv.frequency_domain as fd
import pyhrv.nonlinear as nl

def hrv_feature(signal, sampling_rate):
    sig_len = len(signal)

    #Time domain
    time_results = td.time_domain(nni = signal, sampling_rate = sampling_rate, show = False)
    
    SDNN = time_results['sdnn']
    nn50 = time_results['nn50']
    pnn50 = time_results['pnn50']
    rmssd = time_results['rmssd']
    sdsd = time_results['sdsd']
    tinn =  time_results['tinn']
#     tinn_n = time_results['tinn_n'] ->fail
#     tinn_m, = time_results['tinn_m']
    tri_index = time_results['tri_index']
    
    # Frequency domain
    # results = fd.frequency_domain(ppi_inf[:80] , sampling_rate = 256)
    welch_result = fd.welch_psd(nni = signal, show = False )
    welch_TF = welch_result['fft_total']
    welch_VLF = welch_result['fft_abs'][0]
    welch_LF =  welch_result['fft_abs'][1]
    welch_HF =  welch_result['fft_abs'][2]
    
    welch_LF_n = welch_result['fft_norm'][0]
    welch_HF_n = welch_result['fft_norm'][1]
    
    welch_LF_HF =  welch_result['fft_ratio']
    
    lomb_result = fd.lomb_psd(nni = signal, show = False )
    lomb_TF = lomb_result['lomb_total']
    lomb_VLF = lomb_result['lomb_abs'][0]
    lomb_LF =  lomb_result['lomb_abs'][1]
    lomb_HF =  lomb_result['lomb_abs'][2]
    
    lomb_LF_n = lomb_result['lomb_norm'][0]
    lomb_HF_n = lomb_result['lomb_norm'][1]
    
    lomb_LF_HF =  lomb_result['lomb_ratio']
    
    # nonlinear
    nonlinear_results = nl.nonlinear(nni = signal, sampling_rate = sampling_rate, show = False )
    sd1 = nonlinear_results['sd1']
    sd2 = nonlinear_results['sd2']
    sd_ratio = nonlinear_results['sd_ratio']
    
    sampen = nonlinear_results['sampen']
    # dfa_alpha1 = nonlinear_results['dfa_alpha1']
    # dfa_alpha2 = nonlinear_results['dfa_alpha2']
    
    
    
    # return SDNN, nn50, pnn50, rmssd, sdsd, tinn, tri_index, \
    #             welch_TF, welch_VLF, welch_LF, welch_HF, welch_LF_n, welch_HF_n, welch_LF_HF, \
    #             lomb_TF, lomb_VLF, lomb_LF, lomb_HF, lomb_LF_n, lomb_HF_n, lomb_LF_HF, \
    #             sd1, sd2, sd_ratio, sampen, dfa_alpha1, dfa_alpha2
    return SDNN, nn50, pnn50, rmssd, sdsd, tinn, tri_index, \
                welch_TF, welch_VLF, welch_LF, welch_HF, welch_LF_n, welch_HF_n, welch_LF_HF, \
                lomb_TF, lomb_VLF, lomb_LF, lomb_HF, lomb_LF_n, lomb_HF_n, lomb_LF_HF, \
                sd1, sd2, sd_ratio, sampen

# if __name__ == "__main__":
#     pass        