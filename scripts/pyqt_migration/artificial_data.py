import numpy as np
import ipdb


def spectrum(channels=8192, peak_loc=5000, snr=5):
    signal = np.random.random(channels)
    signal[peak_loc] = snr
    return signal



def radiovision(grid=[4,4], peak_val=2):
    """grid=dim of the array 4x4 default
    """
    data = np.random.random(grid)
    peak_loc = np.random.randint(3,size=2)
    data[peak_loc[0], peak_loc[1]] = peak_val
    return data 

def radiovision2(peak_loc=[2,2]):
    data = np.random.random([4,4])
    data[peak_loc[0], peak_loc[1]] = 2
    return data
