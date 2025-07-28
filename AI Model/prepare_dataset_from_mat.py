import scipy.io as sio
import numpy as np
from fnirs_RawvsFiltered import preprocess_fnirs_window
from eeg_preprocessing import preprocess_eeg_window
from collections import defaultdict

# === Load EEG and fNIRS .mat files ===
cnt_eeg = sio.loadmat('cnt_nback.mat')['cnt_nback']
mrk = sio.loadmat('mrk_nback.mat')['mrk_nback']
cnt_fnirs = sio.loadmat('cnt_nback.mat')['cnt_nback']  # same file, diff field

# === EEG Data ===
eeg_data = cnt_eeg['x'][0, 0]        # shape: [samples, channels]
eeg_fs = int(cnt_eeg['fs'][0, 0][0, 0])
eeg_clab = [str(c[0]) for c in cnt_eeg['clab'][0, 0][0]]

# === fNIRS Data (HbO) ===
fnirs_hbo = cnt_fnirs['oxy'][0, 0]  # struct
fnirs_data = fnirs_hbo['x']         # shape: [samples, 36]
fnirs_fs = int(fnirs_hbo['fs'][0, 0])

# === Marker info ===
marker_pos = mrk['pos'][0, 0][0]       # sample indices of events
marker_code = mrk['y'][0, 0]           # one-hot [n_classes x n_events]
marker_classes = [s[0] for s in mrk['className'][0, 0][0]]

# === Marker code to label map ===
label_map = {
    'S 16': 0,   # 0-back
    'S 48': 1, 'S 64': 1,  # 2-back
    'S 80': 2, 'S 96': 2   # 3-back
}

# === Parameters ===
window_sec = 5
samples_per_window_eeg = eeg_fs * window_sec
samples_per_window_fnirs = fnirs_fs * window_sec

X_eeg, X_fnirs, y = [], [], []

print("[INFO] Starting window extraction and preprocessing...")

# === For each marker ===
for i, pos in enumerate(marker_pos):
    class_idx = np.argmax(marker_code[:, i])
    class_name = marker_classes[class_idx]

    if class_name not in label_map:
        continue  # skip unused markers

    label = label_map[class_name]

    # === EEG window ===
    eeg_start = pos
    eeg_end = eeg_start + samples_per_window_eeg
    if eeg_end > eeg_data.shape[0]:
        continue

    eeg_window = eeg_data[eeg_start:eeg_end, :]
    eeg_processed = preprocess_eeg_window(eeg_window, eeg_clab)  # [30, 500, 1]
    if eeg_processed is None:
        continue

    # === fNIRS window ===
    fnirs_start = int(pos / eeg_fs * fnirs_fs)
    fnirs_end = fnirs_start + samples_per_window_fnirs
    if fnirs_end > fnirs_data.shape[0]:
        continue

    fnirs_window = fnirs_data[fnirs_start:fnirs_end, :].T  # [channels, time]
    fnirs_features = preprocess_fnirs_window(fnirs_window)  # [36, 11]

    X_eeg.append(eeg_processed)
    X_fnirs.append(fnirs_features)
    y.append(label)

    print(f"Processed marker {class_name} at sample {pos} → label {label}")

# === Save arrays ===
X_eeg = np.array(X_eeg)
X_fnirs = np.array(X_fnirs)
y = np.array(y)

np.save('X_eeg.npy', X_eeg)
np.save('X_fnirs.npy', X_fnirs)
np.save('y.npy', y)

print("[DONE] Saved X_eeg.npy, X_fnirs.npy, y.npy")
