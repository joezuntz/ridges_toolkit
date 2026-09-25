import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import sys
import time
import numpy as np
from datetime import timedelta
from nautilus import Prior, Sampler
from scipy.stats import norm
import multiprocessing
import yaml

import likelihood

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


free_parameter_names = [
    'Omega_m',
    'sigma8',
    'w0',
    'ns',
    'Omega_b',
    'H0',
]

prior_bounds = {
    'Omega_m': [0.15, 0.45],
    'sigma8':  [0.5, 1.3],
    'w0':      [-1.25, -0.75],
    'ns':      [0.93, 1.],
    'Omega_b': [0.04, 0.05],
    'H0':      [65, 75.],
}


Like = likelihood.Likelihood("emu/config_files/config_data.yaml")

fid = Like.fiducials
ll_fid = Like.compute_diag_likelihood(fid)

print(" *********** ")
print("This test checks how much does the chi2 change for each individual parameter \n" \
    "and the value of the logL calculated at the fiducials (if correct: logL=0). \n" \
    "If chi2 is near zero across most parameters individually, the emulator/data/covariance \n" \
    "has little information about them")
print(" *********** ")

print("fiducial logL:", ll_fid)  # should be 0 (up to floating point precision)

for name in free_parameter_names:
    lo, hi = prior_bounds[name]
    for value in (lo, hi):
        trial = fid.copy()
        trial[name] = value
        delta_chi2 = -2 * (Like.compute_diag_likelihood(trial) - ll_fid)
        print(name, value, "Delta chi2 =", delta_chi2)