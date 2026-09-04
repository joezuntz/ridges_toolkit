import numpy as np
import likelihood
import os 
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

param_file = 'emu/config_files/config_data.yaml'

Like = likelihood.Likelihood(param_file)

test_data = {
    'Omega_m': 0.31,
    'bary_Mc': 13.82,
    'bary_nu': 0.,
    'sigma8': 0.83,
    'w0': -1.,
    'n_s': 0.97,
    'Omega_b': 0.05,
    'H0': 68.0
}

print('Likelihood=', Like.test_one_likelihood_iteration(test_data))