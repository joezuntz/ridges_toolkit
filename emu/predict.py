# ------------------------------------------ #
# Make a prediction given model and cosmology
# ------------------------------------------ #

import keras
import json
import numpy as np
import os 
import sys

import load_data

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PARAMETER_ORDER = [
    'Omega_m',
    'bary_Mc',
    'bary_nu',
    'sigma8',
    'w0',
    'n_s',
    'Omega_b',
    'H0',
]


def make_prediction(params_dict, lens_bin, source_bin, model_path=None):
    """
    Use emulator to predict tangencial shear signal given cosmology and lens/source bins.
    
    Parameters:
    ----------
    params_dict : dictionary
        Dictionary of cosmological parameters values to use as emulator 
        input, shape (n_samples, n_params).
    lens_bin : int
        Lens bin number (1-6).
    source_bin : int
        Source bin number (1-6).
    model_path : str, optional
        Path to the trained model file. If None, defaults to 'lens{lens_bin}_source{source_bin}.keras'.
    """
    
    # convert params_dict to array in the order of PARAMETER_ORDER
    cosmology = np.array([params_dict[par] for par in PARAMETER_ORDER]).T

    # if cosmology.shape=(8,), reshape to (1,8)
    if cosmology.ndim == 1:
        cosmology = cosmology.reshape(1, -1)

    # get mean and std from metadata
    with open('emu/data/metadata.json', 'r') as f:
        metadata = json.load(f)
    X_mean = np.array(metadata[f'X_mean_l{lens_bin}s{source_bin}'])
    X_std = np.array(metadata[f'X_std_l{lens_bin}s{source_bin}'])
    Y_mean = np.array(metadata[f'Y_mean_l{lens_bin}s{source_bin}'])
    Y_std = np.array(metadata[f'Y_std_l{lens_bin}s{source_bin}'])
    
    cosmology = load_data.standardise_data(cosmology, X_mean, X_std)

    # load trained model.keras from file 
    if model_path==None:
        model_path = f'emu/models/lens{lens_bin}_source{source_bin}.keras'
    model = keras.models.load_model(model_path, compile=False)

    # make prediction 
    predict = model.predict(cosmology, verbose=2)
    predict = load_data.de_standardise_data(predict, Y_mean, Y_std)
    return predict


def fractional_error(predicted_signal, true_signal):
    # fractional error
    frac_error = (predicted_signal - true_signal) / true_signal
    return frac_error


def get_shot_noise(sim_idx, sigma_e = 0.26, counts_file='emu/data/counts.txt'):
    f = np.genfromtxt(counts_file, unpack=False)
    nsim, l, s, counts = f[:,0], f[:,1], f[:,2], f[:,3:]
    
    with open('emu/data/metadata.json', 'r') as f:
        metadata = json.load(f)
    
    npairs = len(metadata['bin_pairs'])
    # We're analysing for one simulation only (idx=0 in previous cells)
    # and we know that each simulation number repeats as many times as are the allowed bin pais
    # we can use the counts with indicies between idx and idx+n_pairs (as first index,
    # the second one refers to the radius)
    gal_per_sqarcmin = counts[sim_idx:sim_idx+npairs,:]

    return sigma_e**2 / gal_per_sqarcmin
