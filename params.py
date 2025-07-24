SCALE_MICROVOLTS = 1e-3

FREQ_LSL = 500
FREQ_MODEL = 100
LINE_NOISE_FREQ = 60

ELECTRODES_MAP8_FCz = ['F3', 'C3', 'P3', 'FCz', 'Pz', 'F4', 'C4', 'P4']
ELECTRODES_MAP8 = ['F3', 'C3', 'P3', 'Cz', 'Pz', 'F4', 'C4', 'P4']
ELECTRODES_MAP16 = ['F7', 'F3', 'T7', 'C3', 'P7', 'P3', 'O1', 'Cz', 'Pz', 'F8', 'F4', 'T8', 'C4', 'P8', 'P4', 'O2']

ELECTRODES_MAP32_FCz = [
    "P8",   "T8",   "CP6",  "FC6",  "F8",   "F4",   "C4",   "P4",  
    "AF4",  "Fp2",  "Fp1",  "AF3",  "Fz",   "FC2",  "FCz",   "CP2",  
    "PO3",  "O1",   "Oz",   "O2",   "PO4",  "Pz",   "CP1",  "FC1", 
    "P3",   "C3",   "F3",   "F7",   "FC5",  "CP5" , "T7",   "P7",
]

NIC2_ELECTRODES = [
    "P8",   "T8",   "CP6",  "FC6",  "F8",   "F4",   "C4",   "P4",  
    "AF4",  "Fp2",  "Fp1",  "AF3",  "Fz",   "FC2",  "Cz",   "CP2",  
    "PO3",  "O1",   "Oz",   "O2",   "PO4",  "Pz",   "CP1",  "FC1", 
    "P3",   "C3",   "F3",   "F7",   "FC5",  "CP5" , "T7",   "P7",
]

NIC2_ELECTRODES_8_REMAP_FCz = {  # Electrode keys in list won't match actual head location, mapped here
    "F3": "Fz",  #12
    "C3": "FC2", #13
    "P3": "Fp2", #9
    "FCz": "Cz", #14
    "Pz": "AF4", #8
    "F4": "Fp1", #10
    "C4": "AF3", #11
    "P4": "CP2", #15
} 

ELECTRODE_IDX_8    = [NIC2_ELECTRODES.index(NIC2_ELECTRODES_8_REMAP_FCz[eeg_c]) for eeg_c in ELECTRODES_MAP8_FCz]  # Remap NIC2 electrode data to actual locations (Using 2nd cable on ELECTRODES_MAP8_FCz locations)
ELECTRODE_IDX_32 = [ELECTRODES_MAP32_FCz.index(eeg_c) for eeg_c in ELECTRODES_MAP8_FCz]  # Get only ELECTRODES_MAP8_FCz channels from NIC2 32-channel data

DATA_LOC_SRC = "/mnt/c/Users/nwymb/Documents/data"
DATA_LOCS = {  # Different Locations    # "compatable" meaning if this model is loaded, we can use this and the other compatable datasets together, with idx adjustments
    "StarStim_8": {'compatable': ["StarStim_32",]},  # i.e. if we use the 8 channel model, we can also use the 32 channel data by reindexing and subsampling.
    "StarStim_32": {'compatable': []},              #      We can't use 8 channel with 32 channel model however
}

DATA_LOC = "/mnt/c/Users/nwymb/Documents/data/XDF"

PRETRAINED_MODELS = {
    'nback': {
        2: "/mnt/c/Users/nwymb/PycharmProjects/Quick/models/20250403/8-channel_NBack-0-2_LOO-sub2_Train-1-16_epochs-50_D-7_F1-19.keras",
        3: "/mnt/c/Users/nwymb/PycharmProjects/Quick/models/20240711/eegnet_8chan_NBACK_3class_FCz.keras",
    },
    'matbii': {
        #2: "/mnt/c/Users/nwymb/PycharmProjects/Quick/models/20240711/eegnet_8chan_MATBII_2class_FCz.keras",
        2: "/mnt/c/Users/nwymb/PycharmProjects/Quick/models/20250707/8-channel_MATBII-0-2_LOO-sub10_epochs-30_D-7_F1-19.keras",
        3: "/mnt/c/Users/nwymb/PycharmProjects/Quick/models/20240711/eegnet_8chan_MATBII_3class_FCz.keras",
    }
}

class EegCap:
    """
    Defines attributes of each EEG Cap setup. Allows use of any cap into our software
    """
    def __init__(self, all_channel_names, desired_channel_names, dataset_path):
        self.all_channel_names = all_channel_names
        self.channels = desired_channel_names
        self.num_channels = len(self.channels)
        self.channel_idx = [all_channel_names.index(eeg_c) for eeg_c in desired_channel_names]
        self.model = ""  # String of path to, or variable of, relevant EEG Classifier model (number of and location of channels within model)


#StarStim_32 = EegCap()
