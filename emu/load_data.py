# ------------------------ #
# Load dataset for emulator
# ------------------------ #

import h5py
from sklearn.model_selection import train_test_split

def load_dataset():
    # load cosmology and signal from dataset.hdf5
    with h5py.File('dataset.hdf5') as f:
        X = f["cosmology"][:]
        Y = f["signal"][:]
    return X, Y

def split_dataset(X, Y):
    # split into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    return X_train, X_test, Y_train, Y_test

def normalise_parameters():
    ## FINISH ME!
    return

def normalise_signal():
    ## FINISH ME!
    return

def denormalise_parameters():
    ## FINISH ME!
    return

def denormalise_signal():
    ## FINISH ME!
    return

def pca():
    ## FINISH ME!
    return