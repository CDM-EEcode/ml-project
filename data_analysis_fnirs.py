import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import butter, filtfilt

# Load file
data = pd.read_csv('ml-project/clean_fnirs_data.csv')  # update with your filename

# Separate time and signals
time = data.iloc[:, 0].values  # time in seconds
signals = data.iloc[:, 1:].values  # HbO and HbR columns

n_channels = signals.shape[1] // 2  # Assuming paired HbO/HbR per channel

# BANDPASS FILTER (0.01–0.2 Hz)
# This is a common range for fNIRS signals to remove noise and slow drifts
def bandpass_filter(signal, fs=10.0, low=0.01, high=0.2):
    nyq = 0.5 * fs
    b, a = butter(4, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, signal, axis=0)

fs = 10  # sampling rate in Hz
signals_filtered = bandpass_filter(signals, fs)

#PLOT HbO AND HbR PER CHANNEL
plt.figure(figsize=(12, 6))
for ch in range(n_channels):
    plt.plot(time, signals_filtered[:, ch], label=f'HbO Ch{ch+1}')
    plt.plot(time, signals_filtered[:, ch + n_channels], '--', label=f'HbR Ch{ch+1}')
plt.title('Hemodynamic Responses (Filtered)')
plt.xlabel('Time (s)')
plt.ylabel('Concentration Change')
plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
plt.tight_layout()
plt.grid(True)
plt.show()

# FEATURE EXTRACTION (per channel)
# Here we can extract features like mean, peak, etc. for each channel
window_start = 10  # seconds
window_end = 20

idx_start = int(window_start * fs)
idx_end = int(window_end * fs)

# Extract features for each channel in the specified time window
features = []
for ch in range(n_channels):
    hbo = signals_filtered[idx_start:idx_end, ch]
    hbr = signals_filtered[idx_start:idx_end, ch + n_channels]

    # skip empty channels
    if hbo.size == 0 or hbr.size == 0:
        print(f"Skipping channel {ch + 1} due to empty data.")
        continue

    feature_vector = {
        'Ch': ch + 1,
        'HbO_mean': np.mean(hbo),
        'HbO_peak': np.max(hbo),
        'HbR_mean': np.mean(hbr),
        'HbR_peak': np.min(hbr),
    }
    features.append(feature_vector)

# Convert to DataFrame
features_df = pd.DataFrame(features)
print("\nExtracted Features:")
print(features_df)

# SAVE FEATURES FOR ML
features_df.to_csv('fnirs_features.csv', index=False)
