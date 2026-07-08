import sys
import os
import json
import math
import matplotlib.pyplot as plt

# change directory to parent directory of this file
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import predict
import load_data
import plot_prediction


# get radial bins, mean and stdev from metadata
with open('emu/data/metadata.json', 'r') as f:
    metadata = json.load(f)

sep_bin_center = metadata['sep_bin_center']

idx1 = 0
idx2 = 50

# big plot with multiple subplots, each one having a different bin pair (lens, source)
all_pairs = metadata["bin_pairs"]
ncols = 3
nrows = math.ceil(len(all_pairs) / ncols)
fig = plt.figure(figsize=(6 * ncols, 5 * nrows))
outer_grid = fig.add_gridspec(nrows, ncols, hspace=0.35, wspace=0.25)

for panel_idx, (l, s) in enumerate(all_pairs):
    row = panel_idx // ncols
    col = panel_idx % ncols
    inner_grid = outer_grid[row, col].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
    ax0 = fig.add_subplot(inner_grid[0])
    ax1 = fig.add_subplot(inner_grid[1], sharex=ax0)

    model_path = 'emu/models/lens' + str(l) + '_source' + str(s) + '.keras'
    X_test, Y_test = load_data.load_dataset(l, s)
    prediction = predict.make_prediction(model_path, X_test, l, s)
    frac_error = predict.fractional_error(prediction, Y_test)

    plot_prediction.plot_prediction_frac_error(
        xarr=sep_bin_center,
        predicted_signal=prediction * 1e3,
        true_signal=Y_test * 1e3,
        fractional_err=frac_error,
        idx1=idx1,
        idx2=idx2,
        xlabel=r'$\theta$',
        ylabel=r'$\gamma_+\, \times10^3$',
        legend=['test1', 'test2'],
        title=f'Test - Lens bin {l}, source bin {s}',
        ax0=ax0,
        ax1=ax1,
        save=True,
        savename='all_bins.pdf'
    )

# hide any unused panels if the grid is larger than the number of bin pairs
for panel_idx in range(len(all_pairs), nrows * ncols):
    row = panel_idx // ncols
    col = panel_idx % ncols
    fig.add_subplot(outer_grid[row, col]).axis('off')

fig.tight_layout()
# plt.show()


# Example single plot
# plot_prediction.plot_prediction_frac_error(xarr=sep_bin_center,
#                                            predicted_signal=prediction*1e3,
#                                            true_signal=Y_test*1e3,
#                                            fractional_err=frac_error,
#                                            idx1=idx1,
#                                            idx2=idx2,
#                                            xlabel=r'$\theta$',
#                                            ylabel=r'$\gamma_+\, \times10^3$',
#                                            legend=['test1', 'test2'],
#                                            title=f'Test - Lens bin {lens_bin}, source bin {source_bin}',
#                                            save=True,
#                                            savename=f'test_l{lens_bin}s{source_bin}.png')