# ------------------------------------------ #
# Make a prediction given model and cosmology
# ------------------------------------------ #

import keras
import json
import numpy as np

import load_data


def make_prediction(model_path, cosmology):

    # get mean and std from metadata
    with open('emu/data/metadata.json', 'r') as f:
        metadata = json.load(f)
    X_mean = np.array(metadata['X_mean'])
    X_std = np.array(metadata['X_std'])
    Y_mean = np.array(metadata['Y_mean'])
    Y_std = np.array(metadata['Y_std'])
    print('ststs:', Y_mean.shape, Y_std.shape)
    print(cosmology.shape)
    
    cosmology = load_data.standardise_data(cosmology, X_mean, X_std)

    # load trained model.keras from file 
    print('MODEL:', model_path)
    model = keras.models.load_model(model_path)

    # make prediction 
    predict = model.predict(cosmology, verbose=2)
    predict = load_data.de_standardise_data(predict, Y_mean, Y_std)
    return predict


def fractional_error(predicted_signal, true_signal):
    # fractional error
    frac_error = (predicted_signal - true_signal) / true_signal
    return frac_error

