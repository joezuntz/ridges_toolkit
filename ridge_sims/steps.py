import sys
import os
import numpy as np
from .samples import load_sample_information_advanced,load_sample_information, load_mask
from .ridge_sims import (
    shell_configuration,
    generate_shell_cl,
    generate_lognormal_gls,
    get_parameter_objects,
    simulate_catalogs,
)
from .config import Config
from .tools import flush
import pickle
import multiprocessing




def step1(config):
    """
    Compute the C_ell density values for each matter shell.
    There are many shells, so this is slow.
    """

    if os.path.exists(config.shell_cl_file):
        print(f"Found shell_cl file {config.shell_cl_file}, skipping step 1")
        return

    _, cosmo = get_parameter_objects(config.h, config.Omega_m, config.Omega_b, config.sigma8)
    windows = shell_configuration(cosmo, config.zmax, config.dx)
    return generate_shell_cl(
        windows,
        config.h,
        config.Omega_m,
        config.Omega_b,
        config.sigma8,
        config.shell_cl_file,
        config.lmax,
    )


def step2(config):
    """
    Generate the log-normal g_ell quantities
    """
    if os.path.exists(config.g_ell_file):
        print(f"Found g_ell file {config.g_ell_file}, skipping step 2")
        return

    with open(config.shell_cl_file, "rb") as f:
        shell_cl = np.load(f)

    if config.nprocess > 1:
        pool = multiprocessing.Pool(config.nprocess)
    else:   
        pool = None

    generate_lognormal_gls(shell_cl, config.g_ell_file, config.nside, config.lmax, pool=pool)

    if config.nprocess > 1:
        pool.close()

    print("Step 2 complete")


def step3(config, sigma_e_default=1e-3):
    """
    Generate catalog files
    """
    rng = np.random.default_rng(seed=config.seed)

    # Load the number density information
    sample = load_sample_information_advanced(
        config.lens_type,
        config.combined,
        lsst=config.lsst,
        lsst10_nz=getattr(config, "lsst10_nz", False),  # NEW option
    )
    # if we want no noise in sims
    if not config.include_shape_noise:
        print("Shape noise disabled: setting sigma_e[:] = 1e-3")
        sample.sigma_e[:] = np.full_like(sample.sigma_e, sigma_e_default)

    # Load the results of the previous step
    with open(config.g_ell_file, "rb") as f:
        gls = pickle.load(f)

    # Load the overall geographic mask, referred to as a "vis" mask in GLASS.
    mask = load_mask(config.nside, lsst=config.lsst)

    # Get the cosmology definition object
    _, cosmo = get_parameter_objects(config.h, config.Omega_m, config.Omega_b, config.sigma8)

    # Simulate the catalogs
    simulate_catalogs(gls, rng, cosmo, sample, mask, config.nside, config.source_cat_file, config.lens_cat_file, config.zmax, config.dx)
