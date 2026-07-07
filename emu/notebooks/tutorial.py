import sys
import os
import json
import keras

# change directory to parent directory of this file
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import predict
import load_data
import plot_prediction


source_bin = 3
lens_bin = 0

# load test dataset
X_test, Y_test = load_data.load_dataset(source_bin, lens_bin)

# get radial bins, mean and stdev from metadata
with open('emu/data/metadata.json', 'r') as f:
    metadata = json.load(f)

X_mean = metadata['X_mean']
X_std = metadata['X_std']
Y_mean = metadata['Y_mean']
Y_std = metadata['Y_std']
sep_bin_center = metadata['sep_bin_center']

model_path = 'emu/models/source'+str(source_bin)+'_lens'+str(lens_bin)+'.keras'

# get prediction
prediction = predict.make_prediction(model_path, X_test)

frac_error = predict.fractional_error(prediction, Y_test)

idx1 = 0
idx2 = 50

plot_prediction.plot_prediction_frac_error(xarr=sep_bin_center,
                                           predicted_signal=prediction,
                                           true_signal=Y_test,
                                           fractional_err=frac_error,
                                           idx1=idx1,
                                           idx2=idx2,
                                           xlabel=r'$\theta$',
                                           ylabel=r'$\gamma_+$',
                                           legend=['test1', 'test2'],
                                           title=f'Test - Lens bin {lens_bin}, source bin {source_bin}')