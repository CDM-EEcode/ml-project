import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# === Step 1: Load Data ===
csv_path = "clean_fnirs_data.csv"
data = pd.read_csv(csv_path)

time = data['time'].values
fs = 10  # sampling rate (Hz)
n_channels = (data.shape[1] - 1) // 2

# === Step 2: Bandpass Filter ===
def bandpass_filter(signal, fs=10.0, low=0.01, high=0.2):
    nyq = 0.5 * fs
    b, a = butter(4, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, signal, axis=0)

signals = data.iloc[:, 1:].values
signals_filtered = bandpass_filter(signals, fs)

# Create filtered DataFrame
filtered_data = pd.DataFrame(np.column_stack((time, signals_filtered)), columns=data.columns)

# === Step 3: Side-by-Side Plot ===
fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

# -- Plot raw data
for ch in range(n_channels):
    axs[0].plot(time, data.iloc[:, 1 + ch], label=f'HbO Ch{ch+1}')
    axs[0].plot(time, data.iloc[:, 1 + ch + n_channels], '--', label=f'HbR Ch{ch+1}')
axs[0].set_title("Raw Hemodynamic Signals")
axs[0].set_xlabel("Time (s)")
axs[0].set_ylabel("Concentration Change")
axs[0].legend()
axs[0].grid(True)

# -- Plot filtered data
for ch in range(n_channels):
    axs[1].plot(time, filtered_data.iloc[:, 1 + ch], label=f'HbO Ch{ch+1}')
    axs[1].plot(time, filtered_data.iloc[:, 1 + ch + n_channels], '--', label=f'HbR Ch{ch+1}')
axs[1].set_title("Filtered Hemodynamic Signals (0.01–0.2 Hz)")
axs[1].set_xlabel("Time (s)")
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()

# === Step 4: Feature Extraction from Filtered Data ===
features = []

for ch in range(n_channels):
    hbo = filtered_data.iloc[:, 1 + ch].values
    hbr = filtered_data.iloc[:, 1 + ch + n_channels].values

    if hbo.size == 0 or hbr.size == 0:
        print(f"Skipping Channel {ch + 1}: empty HbO or HbR column.")
        continue

    feature_vector = {
        'Channel': ch + 1,
        'HbO_mean': np.mean(hbo),
        'HbO_std': np.std(hbo),
        'HbO_peak': np.max(hbo),
        'HbO_time_to_peak': time[np.argmax(hbo)],
        'HbO_auc': np.trapz(hbo, dx=1/fs),

        'HbR_mean': np.mean(hbr),
        'HbR_std': np.std(hbr),
        'HbR_peak': np.min(hbr),
        'HbR_time_to_peak': time[np.argmin(hbr)],
        'HbR_auc': np.trapz(hbr, dx=1/fs),

        'HbO_HbR_ratio': np.mean(hbo / (hbr + 1e-6))
    }

    features.append(feature_vector)

features_df = pd.DataFrame(features)

# === Step 5: Print & Save Feature Table ===
print("\n📋 Extracted Features Table (from Filtered Data):")
print(features_df.to_string(index=False))

features_df.to_csv("fnirs_extracted_features_filtered.csv", index=False)
print("\n✅ Features saved to fnirs_extracted_features_filtered.csv")
