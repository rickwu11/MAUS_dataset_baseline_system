# filters.py
# ==========
# Provide interfaces for signal processing filters

import numpy as np
from scipy.signal import lfilter,iirfilter,medfilt,butter
from scipy import interpolate
from .detect_peaks import *
from PyEMD import EMD, Visualisation

def butterworth(signal, f, fs, btype='low'):
    """ 
    Butterworth high-pass filter 

    Input
    =====
    signal      original signal
    f           frequency for high-pass or low-pass filters, or [f1, f2] for band-pass filters
    fs          sampling frequency
    btype       specify the type of filter
    
    Output
    ======
    signal      result of applying butterworth filter
    """

    nyq = float(fs) * 0.5
    f = f / nyq
    b, a = butter(4, f, btype=btype)
    signal = lfilter(b, a, signal)

    return signal

def QVR(signal, l=100):
    """ Quadratic Variation Reduction (QVR) """

    n = len(signal)
    
    D = np.zeros((n-1, n))
    for i in range(n-1):
        D[i][i] = 1
        D[i][i+1] = -1
    x = np.identity(n) + l * np.dot(D.transpose(), D)
    x = np.linalg.inv(x)
    x = np.dot(x, signal)
    
    return signal - x

# Smooth
def median_filter(signal,kernel_size=13):
    return medfilt(signal,kernel_size)

# Detrend
def detrend(signal,source="pixart"):
    #peak = find_ppg_peak(signal)
    #signal = median_filter(signal)
    if source == "mediatek":
        peak = detect_peaks(signal, mpd=120, edge='rising', valley=True, show=False)
    if source == "pixart":
        peak = detect_peaks(signal, mpd=60, edge='rising', valley=True, show=False)
    if source == "infiniti":
        peak = detect_peaks(signal, mpd=120, edge='rising', valley=True, show=False)
    #signal = detrend(signal,bp=peak)

    peak = np.append(0,peak)
    peak = np.append(peak,len(signal)-1)

    y = [signal[x] for x in peak]
    f = interpolate.interp1d(peak,y,kind='cubic')

    x = np.arange(0,len(signal)-1,1)
    trend = f(x)

    signal = [a-b for a,b in zip(signal,trend)]
    return signal

def EMD_baseline_removal(ppg):
    emd = EMD()
    emd.emd(ppg)
    imfs_pix, res_pix = emd.get_imfs_and_residue()
    sig_pix = np.sum(imfs_pix[:round(len(imfs_pix)*0.5),:], axis = 0)
    sig_pix = median_filter(sig_pix)
    return sig_pix
