# ------------------------ #
# Load dataset for emulator
# ------------------------ #

import os
import sys
import json
import h5py
import numpy as np
from sklearn.model_selection import train_test_split

# get flder of this file
FILE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_dataset(source_bin, lens_bin):
    '''
    Load cosmology and signal from dataset.hdf5
    '''
    with h5py.File('emu/data/dataset.hdf5') as f:
        X = f["cosmology"][:]
        Y = f["g_plus/lens_"+str(lens_bin)+"_source_"+str(source_bin)][:]
        # print shape of loaded arrays
        print(f"Loaded X shape: {X.shape}")
        print(f"Loaded Y shape: {Y.shape}")
    return X, Y


def split_dataset(X, Y):
    '''
    Split into training, validate sets
    '''
    X_train, X_validate, Y_train, Y_validate = train_test_split(X, Y, test_size=0.2, random_state=42)
    return X_train, X_validate, Y_train, Y_validate


def compute_and_save_mean_std(X_train, Y_train):
    '''
    Get mean and standard deviation of training set for standardisation and save to metadata.js
    '''
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0)
    Y_mean = np.mean(Y_train, axis=0)
    Y_std = np.std(Y_train, axis=0)
    
    # save mean and std to metadata.js
    with open('emu/data/metadata.json', 'r') as f:
        metadata = json.load(f)
    metadata['X_mean'] = X_mean.tolist()
    metadata['X_std'] = X_std.tolist()
    metadata['Y_mean'] = Y_mean.tolist()
    metadata['Y_std'] = Y_std.tolist()
    with open('emu/data/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)

    return X_mean, X_std, Y_mean, Y_std

def standardise_data(data, mean_train, std_train):
    '''
    Standardise multidimensional array of cosmological and baryonic parameters 
    using mean and standard deviation of each parameter from training set.

    Parameters
    ----------
    data : array 
        Array containing parameter values, shape (n_sim, n_pars)
    mean_train : array
        Mean of each parameter from training set, shape (n_pars,)
    std_train : array
        Standard deviation of each parameter from training set, shape (n_pars,)
    '''
    norm_data = (data - mean_train) / std_train
    norm_data[np.isnan(norm_data)] = 0
    return norm_data


def de_standardise_data(norm_data, mean_train, std_train):
    '''
    De-standardise multidimensional array of cosmological and baryonic parameters to original scale.
    '''
    data = norm_data * std_train + mean_train
    return data


def process_data(source_bin, lens_bin):
    '''
    Load dataset, split into training, validation and test sets, and standardise the data.
    For a single bin pair (source_bin, lens_bin).
    '''
    X, Y = load_dataset(source_bin, lens_bin) 

    X_train, X_validate, Y_train, Y_validate = split_dataset(X, Y)
    X_mean, X_std, Y_mean, Y_std = compute_and_save_mean_std(X_train, Y_train)

    X_train = standardise_data(X_train, X_mean, X_std)
    X_validate = standardise_data(X_validate, X_mean, X_std)
    Y_train = standardise_data(Y_train, Y_mean, Y_std)
    Y_validate = standardise_data(Y_validate, Y_mean, Y_std)

    return X_train, X_validate, Y_train, Y_validate

def pca():
    ## FINISH ME!
    return