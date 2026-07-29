# --------------------------- #
# Prepare dataset for emulator
# --------------------------- #

import os
from pathlib import Path
import h5py
import json
import tqdm
import numpy as np


def write_dataset(file_path, cosmology, g_plus_data, g_cross_data):
    with h5py.File(file_path, 'w') as f:
        f.create_dataset('cosmology', data=cosmology)
        f.create_dataset('sep_bin_center', data=radius)
        for (l, s) in shear_lens_source_pairs_to_do:
            f.create_dataset(f'g_plus/lens_{l}_source_{s}', data=g_plus_data[:, l, s, :])
            f.create_dataset(f'g_cross/lens_{l}_source_{s}', data=g_cross_data[:, l, s, :])


# import fiducial dataset
metainfo_file = 'emu/data/CosmoGridV1_metainfo.h5'
metainfo = h5py.File(metainfo_file, 'r')

fiducials_dataset = metainfo['parameters/grid']
n_simulations = len(fiducials_dataset)
# read only cosmological + baryonic parameters
param_names = np.array(fiducials_dataset.dtype.names)[:14]

# select the true underlying parameters only
# other parameters are calculated from these, no needed for training
underlying_varying_params = ['bary_Mc', 'bary_nu', 'H0', 'Ob', 'Om', 'ns', 's8', 'w0']
# get indicies corresponding to these parameters in param_names
param_indices = [np.where(param_names == p)[0][0] for p in underlying_varying_params]


shear_folder = 'v2-shear/shear/'
nsims = 2500
radial_bins = 20
source_bins = 4
lens_bins = 4
# save some simulations for testing the emulator
n_tests = 50

# Pairs where the source is behind the lens, as determined
# from a signal-to-noise plot. Save them into metadata.json
shear_lens_source_pairs_to_do = [
    # (lens, source)
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 1),
    (1, 2),
    (1, 3),
    (2, 2),
    (2, 3),
    (3, 3),
]
with open('emu/data/metadata.json', 'r') as f:
    metadata = json.load(f)
metadata["bin_pairs"] = shear_lens_source_pairs_to_do
with open('emu/data/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
        
# create empty arrays to store signal and parameters
param_values_arr = np.zeros((nsims, len(param_names)))
g_plus_arr = np.zeros((nsims,
                       lens_bins,
                       source_bins,
                       radial_bins))
g_cross_arr = np.zeros((nsims,
                        lens_bins,
                        source_bins,
                        radial_bins))

# remove dataset.hdf5 file if already exists
if os.path.exists('emu/data/dataset.hdf5'):
    os.remove('emu/data/dataset.hdf5')

loop_count = 0
radius_saved = False

sim_folders = sorted(Path(shear_folder).glob("cosmo_*"))
for sim_folder in tqdm.tqdm(sim_folders, total=min(len(sim_folders), nsims), desc='Building dataset'):
    # extract sim_id from folder name (sim ids not continuous)
    sim_id_str = sim_folder.name.replace('cosmo_', '')
    sim_id = int(sim_id_str) - 1
    
    if loop_count < nsims:
        # store fiducial parameters
        param_values_arr[loop_count, :] = np.array([fiducials_dataset[loop_count][name] 
                                                    for name in param_names])

        for s in range(source_bins):
            for l in range(lens_bins):
                # check if bin pair is in the list of pairs to do
                if (l, s) not in shear_lens_source_pairs_to_do:
                    continue
                
                shear_file = 'perm_0000_shear_lens' + str(l) + '_source' + str(s) + '.txt'
                radius, _, g_plus, g_cross, _, _ = np.genfromtxt(sim_folder/shear_file,
                                                                 unpack=True)
                
                # store shear signal g+ and gx
                g_plus_arr[loop_count, l, s, :] = g_plus
                g_cross_arr[loop_count, l, s, :] = g_cross
        loop_count += 1
        # update values if key already exists in metadata.js metadata = {} only once
        if not radius_saved:
            with open('emu/data/metadata.json', 'r') as f:
                metadata = json.load(f)
            metadata['sep_bin_center'] = radius.tolist()
            radius_saved = True


# shuffle the dataset to randomise the order of simulations
indices = np.arange(nsims)
np.random.shuffle(indices)
# save only parameters with varying values, and the corresponding shear signals
param_values_arr = param_values_arr[indices]
param_values_arr = param_values_arr[:, param_indices]
g_plus_arr = g_plus_arr[indices]
g_cross_arr = g_cross_arr[indices]

write_dataset('emu/data/test_dataset.hdf5', param_values_arr[:n_tests], g_plus_arr[:n_tests], g_cross_arr[:n_tests])
write_dataset('emu/data/dataset.hdf5', param_values_arr[n_tests:], g_plus_arr[n_tests:], g_cross_arr[n_tests:])