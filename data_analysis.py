import mne, os, pyxdf, argparse
import pandas as pd
import numpy as np
from params import ELECTRODES_MAP32_FCz, FREQ_LSL, FREQ_MODEL, LINE_NOISE_FREQ
import matplotlib.pyplot as plt

LSL_STREAM_NAMES = {
    'eeg': ["MSI_LSL_OUT-EEG", "Alienware_LSL_EEG_OUT-EEG", "HPLaptopLSLOutletStream-EEG"],
    'markers': ["OpenMATB", "PsychoPy_LSL"],
    'quality': ["MSI_LSL_OUT-Quality", "Alienware_LSL_EEG_OUT-Quality", "HPLaptopLSLOutletStream-Quality"],
}

DATA_DIR = "/mnt/c/Users/nwymb/Documents/data"  # Change to where you copy the data (one directory above 'XDF')
TS_AND_ELECTRODES = ['timestamps']+ELECTRODES_MAP32_FCz

info = mne.create_info(ELECTRODES_MAP32_FCz, FREQ_LSL, ['eeg']*len(ELECTRODES_MAP32_FCz))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--easy_file", "-f",
        type=str, nargs="?", default='P001_S001_R001',
        help="file to load (participant, session, run). Will load equivalent xdf"
    )

    parser.add_argument(
        "--plot_type", "-pt",
        type=str, nargs="?", default='psd', choices=['psd', 'timeseries'],
        help="file to load (participant, session, run). Will load equivalent xdf"
    )

    parser.add_argument(
        "--plot_timeseries", "-ts",
        type=bool, nargs="?", default=False, const=True,
        help="Plot Epoched Time Series Data"
    )

    parser.add_argument(
        "--plot_psd_topomap", "-psd",
        type=bool, nargs="?", default=False, const=True,
        help="Plot PSD topomap"
    )

    parser.add_argument(
        "--plot_quality", "-q",
        type=bool, nargs="?", default=False, const=True,
        help="Plot QI Data"
    )

    parser.add_argument(
        "--task", "-t",
        type=str, default="MATBII", choices=["NBACK", "MATBII"],
        help="Task to get data from"
    )

    parser.add_argument(
        "--use_prep_pipeline", "-prep",
        type=bool, nargs="?", default=False, const=True,
        help="Use Prep Pipeline to filter and interpolate EEG"
    )

    return parser.parse_args()

def plot_easy(file):

    with open(file,'r') as f:
        easy_data = np.array([[int(v) for v in line.rstrip().split('\t')] for line in f.readlines()])
    e_eeg = easy_data[:,:32]*1e-9  # EEG to ML
    e_ts = np.expand_dims(easy_data[:,-1]*1e-3, axis=1)
    e_np = pd.DataFrame(np.hstack((e_ts, e_eeg)), columns=TS_AND_ELECTRODES)

    return e_np

    print('Easy File Average Value:')
    print(e_np.mean(axis=0))
    e = mne.io.RawArray(e_np.to_numpy()[:,1:].T, info)
    e.plot(title='EASY File', n_channels=32, block=False)


def plot_quality(data, title=""):
    colors = plt.cm.get_cmap('tab20', 20)
    f, ax = plt.subplots(1, figsize=(12,10))

    print(data.shape)
    for e in range(32):

        i = np.where(data[:, e] > 0)[0]
        print(f"Electrode {e} ({ELECTRODES_MAP32_FCz[e]})")
        print(data[i, e])
        ax.plot(np.arange(len(i)), data[i, e], color=colors(e % 20), label=ELECTRODES_MAP32_FCz[e])  # +i)

    ax.set_title(title)
    ax.legend()
    plt.show()


def get_xdf_data(file):

    x_eeg = None
    x_ts = None
    x_markers = None
    x_quality = None

    xdf_data = pyxdf.load_xdf(file)[0]
    for _, data in enumerate(xdf_data):
        data_name = data['info']['name'][0]
        print(f"Stream-Name: {data_name}")
        if data_name in LSL_STREAM_NAMES['eeg']: # PYXDF_EEG_NAME:
            x_eeg = data['time_series']*1e-6
            x_ts = np.expand_dims(data['time_stamps'],axis=1)
        elif data_name in LSL_STREAM_NAMES['markers']:
            x_markers = data['time_series']

        elif data_name in LSL_STREAM_NAMES['quality']:
            x_quality = data['time_series']

    x_np = pd.DataFrame(np.hstack((x_ts, x_eeg)), columns=TS_AND_ELECTRODES)

    return x_np, x_markers, x_quality


def interpolate_channels(raw):
    from pyprep.prep_pipeline import PrepPipeline
    from pyprep.find_noisy_channels import NoisyChannels

    noisy_chans = NoisyChannels(raw).get_bads()
    prepped_array = PrepPipeline(
        raw,
        {'ref_chs': 'eeg', 'reref_chs': 'eeg', 'line_freqs': np.arange(LINE_NOISE_FREQ, FREQ_LSL / 2, LINE_NOISE_FREQ)},
        raw.get_montage()
    )
    prepped_array.fit()
    updated_array = mne.io.RawArray(prepped_array.EEG_clean, mne.create_info(raw.ch_names, raw.info['sfreq'], ['eeg']*len(raw.ch_names)))
    updated_array.set_montage(raw.get_montage())
    updated_array.info['bads'] = noisy_chans
    print(f"Interpolating channels: {noisy_chans}")
    interpolated_array = updated_array.copy().interpolate_bads()
    interpolated_array.plot(title="PREP Interpolated and Filtered", n_channels=32)

    return interpolated_array

def main(args):

    plot_type = args.plot_type

    plot_timeseries = args.plot_timeseries

    easy_files = args.easy_file
    easy_files = easy_files.split(' ')

    xdf_base_path = f"{DATA_DIR}/XDF/{args.task}/Starstim32_FCz"

    # Generate list of files based on input args
    xdf_file_paths = []
    for easy_file in easy_files:
        data = easy_file.split('.')[0].split('_')
        if len(data) == 1:  # Plot all participant data
            sub = data[0]
            sub_path = f"{xdf_base_path}/sub-{sub}"
            ses_paths = [f"{sub_path}/ses-{ses}/eeg" for ses in os.listdir(sub_path)]
            # runs_paths = [f"{ses}/run"]
            for ses in ses_paths:
                runs = [f"{ses}/{run}" for run in os.listdir(ses)]
                xdf_file_paths += runs

        elif len(data) == 2:
            sub, ses = data[0], data[1]
            ses_path = f"{xdf_base_path}/sub-{sub}/ses-{ses}/eeg"
            # for ses in ses_paths:
            runs = [f"{ses_path}/{run}" for run in os.listdir(ses_path)]
            xdf_file_paths += runs

        elif len(data) == 3:
            sub, ses, run = data[0], data[1], data[2]
            ses_path = f"{xdf_base_path}/sub-{sub}/ses-{ses}/eeg"
            runs = [f"{ses_path}/{r}" for r in os.listdir(ses_path) if f'run-{run[1:]}' in r][0]
            xdf_file_paths.append(runs)

        else:
            print(f"Error: Could not parse argument: {easy_file}")

    for xdf_file_path in xdf_file_paths:
        # for easy_file in easy_files:

        # sub, ses, run = easy_file.split('.')[0].split('_')
        # print(sub, ses, run)
        #
        # easy_file_full = f"{DATA_DIR}/EasyData/{args.task}/{easy_file}"
        # #xdf_path = f"{DATA_DIR}/XDF/MATBII/Starstim32_FCz/sub-{sub}/ses-{ses}/eeg"
        # xdf_path = f"{DATA_DIR}/XDF/{args.task}/Starstim32_FCz/sub-{sub}/ses-{ses}/eeg"
        # xdf_file = [f for f in os.listdir(xdf_path) if f'run-{run[1:]}' in f][0]
        # task = xdf_file.split("_")[2]
        #
        # xdf_file_full = f"{xdf_path}/{xdf_file}"

        xdf_filename = xdf_file_path.split('/')[-1]
        # e = plot_easy(easy_file_full)
        x, markers, quality = get_xdf_data(xdf_file_path)  # xdf_file_full

        data_dict = {
            #    'EASY': e,
            'XDF': x,
        }

        num_plots = len(data_dict.keys())
        figs = {}

        for count, (key, data) in enumerate(data_dict.items()):

            print(f'{key} File Average Value:')
            print(data.mean(axis=0))

            mne_data = mne.io.RawArray(data.to_numpy()[:, 1:].T, info)                  # Data that will be filtered with out functions
            mne_data.set_montage(mne.channels.make_standard_montage('standard_1020'))
            raw = mne_data.copy()                                                       # Raw Data for reference

            if args.use_prep_pipeline:
                prep_data = interpolate_channels(mne_data)                              # Data with advanced filtering and interpolation

            raw.plot(title=f'Raw Data', n_channels=32)
            mne_data.resample(FREQ_MODEL)
            # for freq in [60]: #[50, 60, 100]:  # Resample at nyquist of 50 should make this additional filtering pointless
            #    mne_data.notch_filter(freqs=freq, method='iir', trans_bandwidth=1.0, verbose=True)
            mne_data.filter(l_freq=0.1, h_freq=None, method='fir', verbose=True)        # Highpass at 0.1 Hz
            #return mne_data, raw

            # TODO: Add 'psd' and 'epochs' arguments to plot either/or/both
            if args.plot_timeseries:
                mne_data.plot(title=f'{key} File (HP @ 0.1Hz, Resample to 100Hz)', n_channels=32, block=True if count + 1 == num_plots else False)
            if args.plot_psd_topomap:
                fig = mne_data.compute_psd().plot_topomap(show=False)
                fig2 = prep_data.compute_psd().plot_topomap(show=False)
                fig.suptitle(xdf_filename)
                plt.show(block=True)
            if args.plot_quality:
                plot_quality(quality, title=xdf_filename)

            plt.show()

if __name__ == '__main__':
    args = get_args()
    main(args)
