# --------------------------- #
# Prepare dataset for emulator
# --------------------------- #

import os
from pathlib import Path
import h5py
import json
import tqdm
import numpy as np

# import fiducial dataset
metainfo_file = 'emu/CosmoGridV1_metainfo.h5'
metainfo = h5py.File(metainfo_file, 'r')

fiducials_dataset = metainfo['parameters/grid']
n_simulations = len(fiducials_dataset)
# read only cosmological + baryonic parameters
param_names = np.array(fiducials_dataset.dtype.names)[:14]

shear_folder = 'v1-shear/shear/'
nsims = 2500
radial_bins = 20
source_bins = 4
lens_bins = 4

# Pairs where the source is behind the lens, as determined
# from a signal-to-noise plot
shear_lens_source_pairs_to_do = [
    # (lens, source)
    # (0, 1),
    # (0, 2),
    (0, 3),
    # (1, 1),
    # (1, 2),
    # (1, 3),
    # (2, 2),
    # (2, 3),
    # (3, 3),
]

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
if os.path.exists('emu/dataset.hdf5'):
    os.remove('emu/dataset.hdf5')

loop_count = 0
radius_saved = False

sim_folders = sorted(Path(shear_folder).glob("cosmo_*"))
for sim_folder in tqdm.tqdm(sim_folders, total=min(len(sim_folders), nsims), desc='Building dataset'):
# for sim_folder in sorted(Path(shear_folder).glob("cosmo_*")):
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
            ### FINISH ME!
            radius_saved = True


# save arrays to dataset.hdf5
with h5py.File('emu/dataset.hdf5', 'w') as f:
    f.create_dataset('cosmology', data=param_values_arr)
    f.create_dataset('sep_bin_center', data=radius)
    # split bins into lens-source pairs and store g+ and gx signals
    for l in range(lens_bins):
        for s in range(source_bins):
            if (l, s) not in shear_lens_source_pairs_to_do:
                continue
            f.create_dataset(f'g_plus/lens_{l}_source_{s}', data=g_plus_arr[:, l, s, :])
            f.create_dataset(f'g_cross/lens_{l}_source_{s}', data=g_cross_arr[:, l, s, :])