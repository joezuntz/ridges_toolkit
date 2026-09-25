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

Like = None

def get_likelihood():
    global Like
    if Like is None:
        Like = likelihood.Likelihood("emu/config_files/config_data.yaml")
    return Like


def get_log_likelihood(params):
    like = get_likelihood()
    param_dic = dict(like.fixed_params)
    param_dic.update(params)

    loglike = like.compute_diag_likelihood(param_dic)
    return loglike if np.isfinite(loglike) else -np.inf


def initialize_prior(config_file):
    with open(config_file, 'r') as f:
        params_dict = yaml.safe_load(f)

    prior = Prior()
    for par_i in params_dict:

        if params_dict[par_i]['type']=='G':
            param_mean = params_dict[par_i]['mean']
            param_std = params_dict[par_i]['std']
            prior.add_parameter(par_i,
                                dist=norm(loc = param_mean, scale = param_std))
            
        elif params_dict[par_i]['type']=='U':
            lower_lim = params_dict[par_i]['low']
            upper_lim = params_dict[par_i]['upp']
            prior.add_parameter(par_i,
                                dist=(lower_lim, upper_lim))

    return prior


# def get_log_likelihood(params):
#     param_dic = params | Like.fixed_params

#     loglike = Like.compute_diag_likelihood(param_dic)

#     if not np.isfinite(loglike):
#         return -np.inf

#     return loglike


def get_header(params_dict):
    # write parameter names (free)
    header = f'# Omega_m sigma8 w0 ns Omega_b H0  log_w  log_l'
    return header


def main():  
    prior = initialize_prior("emu/config_files/config_data.yaml")

    filename = 'test_x'
    header = get_header("emu/config_files/config_data.yaml")

    n_processes = 4  # start conservatively; do not use every logical CPU
    context = multiprocessing.get_context("spawn")

    with context.Pool(processes=n_processes) as pool:
        sampler = Sampler(
            prior,
            get_log_likelihood,
            filepath=f"emu/chains/hdf5/{filename}.hdf5",
            resume=False,
            n_live=1000,
            pool=pool,
        )
        start = time.time()
        sampler.run(verbose=True, discard_exploration=True, n_eff=5000)
    

    
    log_z = sampler.evidence()
    points, log_w, log_l = sampler.posterior()
    finish = time.time()
    chain_time = timedelta(seconds = finish-start)

    np.savetxt(f"emu/chains/{filename}.txt",
               np.c_[points, log_w, log_l],
               header=header,
               footer=f'log_Z = {log_z}; chain_time = {chain_time} (--> {chain_time} hh:mm:ss)')


if __name__ == "__main__":
    try:
        main()
    finally:
        # Ensure all pools are properly closed
        multiprocessing.active_children()
