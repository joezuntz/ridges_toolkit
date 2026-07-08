# ------------------------------------------ #
# Make a prediction given model and cosmology
# ------------------------------------------ #

import keras
import json
import numpy as np

import load_data


def make_prediction(model_path, cosmology, lens_bin, source_bin):

    # get mean and std from metadata
    with open('emu/data/metadata.json', 'r') as f:
        metadata = json.load(f)
    X_mean = np.array(metadata[f'X_mean_l{lens_bin}s{source_bin}'])
    X_std = np.array(metadata[f'X_std_l{lens_bin}s{source_bin}'])
    Y_mean = np.array(metadata[f'Y_mean_l{lens_bin}s{source_bin}'])
    Y_std = np.array(metadata[f'Y_std_l{lens_bin}s{source_bin}'])
    
    cosmology = load_data.standardise_data(cosmology, X_mean, X_std)

    # load trained model.keras from file 
    model = keras.models.load_model(model_path)

    # make prediction 
    predict = model.predict(cosmology, verbose=2)
    predict = load_data.de_standardise_data(predict, Y_mean, Y_std)
    return predict


def fractional_error(predicted_signal, true_signal):
    # fractional error
    frac_error = (predicted_signal - true_signal) / true_signal
    return frac_error

