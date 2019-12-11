"""
Filters for EEG data preprocessing.
Includes bandpass, highpass and lowpass
"""

from scipy.signal import butter, sosfilt


def highpass(x, fs, fc, order=4):
    """
    Applies highpass filter

    Parameters:
     - x:     numpy.array, input signal, sampled at fs
     - fs:    float, sampling frequency
     - fc:    float, cutoff frequency
     - order: filter order

    Returns: numpy.array
    """
    nyq = 0.5 * fs
    norm_fc = fc / nyq
    sos = butter(order, norm_fc, btype='highpass', output='sos')
    return sosfilt(sos, x)


def bandpass(x, fs, fc_low, fc_high, order=5):
    """
    Applies highpass filter

    Parameters:
     - x:       numpy.array, input signal, sampled at fs
     - fs:      float, sampling frequency
     - fc_low:  float, low cutoff frequency
     - fc_high: float, high cutoff frequency
     - order:   filter order

    Returns: numpy.array
    """
    nyq = 0.5 * fs
    norm_fc_low = fc_low / nyq
    norm_fc_high = fc_high / nyq
    sos = butter(order, [norm_fc_low, norm_fc_high], btype='bandpass', output='sos')
    return sosfilt(sos, x)
