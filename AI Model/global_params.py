# === Parameters for Multimodal EEG-fNIRS Project (TU Berlin Dataset) ===

# Scaling EEG microvolt data (if needed)
SCALE_MICROVOLTS = 1e-3

# Sampling rates
FREQ_LSL = 500         # EEG sample rate (Hz)
FREQ_MODEL = 100        # (Optional) Target sample rate for model input
LINE_NOISE_FREQ = 60    # Notch filter frequency (Hz)

# === Custom EEG Electrode Layout Used (30 channels from BrainVision) ===
ELECTRODES_USED = [
    'Fp1', 'AFF5h', 'AFz', 'F1', 'FC5', 'FC1', 'T7', 'C3', 'Cz', 'CP5',
    'CP1', 'P7', 'P3', 'Pz', 'POz', 'O1', 'Fp2', 'AFF6h', 'F2', 'FC2',
    'FC6', 'C4', 'T8', 'CP2', 'CP6', 'P4', 'P8', 'O2', 'HEOG', 'VEOG'
]

# === fNIRS Parameters ===
FNIRS_SAMPLING_RATE = 10  # Hz (as per your NIRx data)
FNIRS_FEATURE_DIM = 6     # Mean, Std, Slope, Skewness, Kurtosis, AUC

# === Dataset Configuration Notes ===
# EEG Source: TU Berlin n-back task (BrainVision .mat)
# fNIRS Source: NIRx device (36 channels, raw input, bandpass + feature extraction)
# AI Model: Custom-trained CNN (EEGNet backbone) with early fusion for workload decoding
#           No pretrained EEG-only models used (all training is from scratch)

