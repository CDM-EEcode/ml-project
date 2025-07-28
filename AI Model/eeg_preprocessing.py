import numpy as np
import scipy.signal as signal
from global_params import ELECTRODES_USED, FREQ_LSL, LINE_NOISE_FREQ

# === EEG Preprocessing ===

def preprocess_eeg_window(eeg_window, clab, fs=500):
    """
    Preprocess a single window of EEG data.
    Input:
        eeg_window: [samples, channels] raw EEG data
        clab: list of channel names matching eeg_window
        fs: sampling rate (default 500Hz)
    Output:
        preprocessed EEG window: shape [30, 500, 1]
    """
    try:
        indices = [clab.index(chan) for chan in ELECTRODES_USED]
    except ValueError as e:
        print(f"[ERROR] Missing channel: {e}")
        return None

    eeg_data = eeg_window[:, indices].T  # [channels, time]

    # Bandpass filter: 1–50 Hz
    b, a = signal.butter(5, [1 / (fs / 2), 50 / (fs / 2)], btype='band')
    eeg_data = signal.filtfilt(b, a, eeg_data)

    # Notch filter at 60 Hz (line noise)
    notch_freq = LINE_NOISE_FREQ
    b_notch, a_notch = signal.iirnotch(notch_freq / (fs / 2), Q=30)
    eeg_data = signal.filtfilt(b_notch, a_notch, eeg_data)

    # Normalize each channel (z-score)
    eeg_data = (eeg_data - np.mean(eeg_data, axis=1, keepdims=True)) / np.std(eeg_data, axis=1, keepdims=True)

    # Reshape to [channels, samples, 1] for CNN
    eeg_data = np.expand_dims(eeg_data, axis=-1)
    return eeg_data
