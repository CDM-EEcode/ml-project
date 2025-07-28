import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
from fnirs_RawvsFiltered import bandpass_filter, fs as FNIRS_FS
from tensorflow.keras.models import load_model
import time
import datetime
import csv

# === Simulation Setup ===
EEG_FS = 100  # Hz
EEG_CHANNELS = 32
EEG_WINDOW_SEC = 5
FNIRS_CHANNELS = 36
FNIRS_FEATURES_PER_CH = 11  # from fnirs_RawvsFiltered.py
SLIDE_STEP_SEC = 1

# === Load pretrained model ===
model = load_model("MultimodalEEGNetWorkload.h5")  # ensure model is saved with this name

# === Buffers ===
eeg_buffer = deque(maxlen=EEG_FS * EEG_WINDOW_SEC)
fnirs_buffer = deque(maxlen=FNIRS_FS * EEG_WINDOW_SEC)

# === CSV Logging ===
csv_file = open("predictions.csv", mode="w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["timestamp", "prediction"])

# === Real-time Plot ===
pred_history = deque(maxlen=60)
time_history = deque(maxlen=60)
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot([], [], label="Workload")
ax.set_ylim([-0.1, 2.1])
ax.set_yticks([0, 1, 2])
ax.set_yticklabels(["Easy", "Medium", "Hard"])
ax.set_title("Real-Time Cognitive Workload")
ax.set_xlabel("Time")
ax.set_ylabel("Prediction")
ax.legend()

# === Label Mapping ===
label_map = {0: "easy", 1: "medium", 2: "hard"}

# === Feature Extraction ===
def extract_fnirs_features(window):
    window = np.array(window)
    filtered = bandpass_filter(window, fs=FNIRS_FS)
    n_channels = filtered.shape[1] // 2
    time_axis = np.linspace(0, window.shape[0] / FNIRS_FS, window.shape[0])
    features = []

    for ch in range(n_channels):
        hbo = filtered[:, ch]
        hbr = filtered[:, ch + n_channels]
        feat = [
            np.mean(hbo), np.std(hbo), np.max(hbo), time_axis[np.argmax(hbo)], np.trapz(hbo),
            np.mean(hbr), np.std(hbr), np.min(hbr), time_axis[np.argmin(hbr)], np.trapz(hbr),
            np.mean(hbo / (hbr + 1e-6))
        ]
        features.append(feat)

    return np.array(features)

# === Main Inference Loop ===
print("\n🧠 Starting Real-Time Cognitive Workload Inference...\n")

while True:
    # === Simulated Live Input ===
    eeg_sample = np.random.randn(EEG_CHANNELS)  # Simulated 1 EEG sample
    fnirs_sample = np.random.randn(FNIRS_CHANNELS * 2)  # HbO + HbR

    eeg_buffer.append(eeg_sample)
    fnirs_buffer.append(fnirs_sample)

    # Wait for full 5s window
    if len(eeg_buffer) < EEG_FS * EEG_WINDOW_SEC:
        time.sleep(1 / EEG_FS)
        continue

    if len(fnirs_buffer) < FNIRS_FS * EEG_WINDOW_SEC:
        time.sleep(1 / FNIRS_FS)
        continue

    # === Preprocess EEG ===
    eeg_window = np.array(eeg_buffer).T  # [32, 500]
    eeg_input = np.expand_dims(eeg_window, axis=-1)  # [32, 500, 1]

    # === Preprocess fNIRS ===
    fnirs_window = np.array(fnirs_buffer)
    fnirs_features = extract_fnirs_features(fnirs_window)
    fnirs_input = np.expand_dims(fnirs_features, axis=0)  # [1, 36, 11]

    # === Add batch dim to EEG ===
    eeg_input = np.expand_dims(eeg_input, axis=0)  # [1, 32, 500, 1]

    # === Predict ===
    prediction = model.predict([eeg_input, fnirs_input], verbose=0)
    pred_label = np.argmax(prediction)
    pred_str = label_map[pred_label]

    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] Prediction: {pred_str}")
    csv_writer.writerow([timestamp, pred_str])
    csv_file.flush()

    # === Update plot ===
    pred_history.append(pred_label)
    time_history.append(timestamp)
    line.set_xdata(np.arange(len(pred_history)))
    line.set_ydata(pred_history)
    ax.set_xticks(np.arange(len(time_history))[::max(1, len(time_history)//10)])
    ax.set_xticklabels(list(time_history)[::max(1, len(time_history)//10)], rotation=45)
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()

    time.sleep(SLIDE_STEP_SEC)  # Slide window every second
