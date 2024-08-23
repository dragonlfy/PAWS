from scipy import io
import os
import numpy as np

def load_eeg_data(file_list):
    raw_eeg = []
    for file in file_list:
        raw_eeg.append(io.loadmat(file))
    return raw_eeg

def process_experiment_data(raw_eeg, sub, prefix, len_window, label, raw_X, raw_y):
    while raw_eeg and sub < 12: # only using the first 12 subjects to avoid memory exceed
    # exp 1
        element = raw_eeg.pop(0)
        for i in range(1, 16):
            data = element[prefix[sub] + "_eeg" + str(i)]
            # print(data.shape)
            n_windows = data.shape[1] // len_window
            # print(n_windows)
            reshaped_X = np.reshape(data[:, :n_windows * len_window], (62, len_window, n_windows))
            raw_X.append(reshaped_X)
            raw_y.append(np.array([label['label'][0][i-1] for j in range(n_windows)]))
        del element


        # exp 2 
        element = raw_eeg.pop(0)
        for i in range(1, 16):
            data = element[prefix[sub] + "_eeg" + str(i)]
            # print(data.shape)
            n_windows = data.shape[1] // len_window
            # print(n_windows)
            reshaped_X = np.reshape(data[:, :n_windows * len_window], (62, len_window, n_windows))
            raw_X.append(reshaped_X)
            raw_y.append(np.array([label['label'][0][i-1] for j in range(n_windows)]))
        del element

        # exp 1
        element = raw_eeg.pop(0)
        for i in range(1, 16):
            data = element[prefix[sub] + "_eeg" + str(i)]
            # print(data.shape)
            n_windows = data.shape[1] // len_window
            # print(n_windows)
            reshaped_X = np.reshape(data[:, :n_windows * len_window], (62, len_window, n_windows))
            raw_X.append(reshaped_X)
            raw_y.append(np.array([label['label'][0][i-1] for j in range(n_windows)]))
        del element

        sub += 1

        