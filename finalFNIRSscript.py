import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === Step 1: Load Data ===
csv_path = "/home/intern_acct/ml-project/clean_fnirs_data.csv"  # Make sure this file is in the same folder
data = pd.read_csv(csv_path)

time = data['time'].values
fs = 10  # sampling rate (Hz)
n_channels = (data.shape[1] - 1) // 2  # assumes paired HbO/HbR columns

# === Step 2: Plot HbO and HbR Signals ===
plt.figure(figsize=(12, 6))
for ch in range(n_channels):
    hbo = data.iloc[:, 1 + ch].values
    hbr = data.iloc[:, 1 + ch + n_channels].values

    # Plot HbO and HbR for each channel
    plt.plot(time, hbo, label=f'HbO Ch{ch+1}')
    plt.plot(time, hbr, '--', label=f'HbR Ch{ch+1}')
plt.title("Hemodynamic Response (HbO & HbR)")
plt.xlabel("Time (s)")
plt.ylabel("Concentration Change")
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.show()

# === Step 3: Extract Features ===
features = []

for ch in range(n_channels):
    hbo = data.iloc[:, 1 + ch].values
    hbr = data.iloc[:, 1 + ch + n_channels].values

    # Skip empty channels
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

        'HbO_HbR_ratio': np.mean(hbo / (hbr + 1e-6))  # small constant to avoid div-by-zero
    }

    features.append(feature_vector)

features_df = pd.DataFrame(features)

# === Step 4: Print Feature Table ===
print("\n📋 Extracted Features Table:")
print(features_df.to_string(index=False))  # Full clean table in terminal

# === Step 5: Save to CSV (optional) ===
features_df.to_csv("fnirs_extracted_features.csv", index=False)
print("\n✅ Features saved to fnirs_extracted_features.csv")
