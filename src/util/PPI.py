"""
    Author          : Rick
    File Description: Extract RRI PPI
    Date            : 20191203
"""

import numpy as np
from .detect_peaks import detect_peaks
from .filters import butterworth, QVR, median_filter,detrend 
from biosppy.signals import ecg


def RRI(ecg_sig, fs=256, show=False):
    ecg_out = ecg.ecg(signal=ecg_sig, sampling_rate=fs, show=show)
    peaks = ecg_out[2]
    rri = np.diff(peaks)
    rri = rri*1000/fs

    return rri, peaks



def PPI(ppg, sourse='pixart', manual_mpd = False,  m_mpd=60, onset=True, show=False, subject=5):
    ppg = median_filter(ppg)
    if onset == False:
        valley = False
        edge = 'falling'
    else:
        valley = True
        edge = 'rising'

    if sourse == 'pixart':
        if subject <= 4:
            fs = 102.5
        else:
            fs = 100
        
        if manual_mpd == False:
            mpd = 55
        else:
            mpd = m_mpd

        peaks = detect_peaks(ppg, mpd=mpd, edge=edge, valley=valley, show=show)
        ppi = np.diff(peaks)
        ppi = ppi*1000/fs
    else :
        fs = 256

        if manual_mpd == False:
            mpd = 130
        else:
            mpd = m_mpd*2.56
        peaks = detect_peaks(ppg, mpd=mpd, edge=edge, valley=valley, show=show)
        ppi = np.diff(peaks)
        ppi = ppi*1000/fs

    return ppi, peaks
