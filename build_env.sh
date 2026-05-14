#!/usr/bin/env bash

set -e

# Lets us use mamba activate in the script 
# whether or not it is already active in the shell
eval "$(mamba shell hook --shell bash)"

set -x

LOCATION_FLAG="-n"
LOCATION="ridges"
while getopts "d:" opt; do
    case $opt in
        d) LOCATION="$OPTARG"; LOCATION_FLAG="-p" ;;
    esac
done


mamba create ${LOCATION_FLAG} ${LOCATION} -c conda-forge camb=1.6.5 fitsio=1.3.0 h5py=3.16.0 healpix=2025.1 healpy=1.19.0 ipython=9.10.1 matplotlib-base=3.10.8 mpi4py=4.1.1 namaster=2.7 networkx=3.6.1 numba=0.65.0 numpy=2.4.3 pandas=3.0.2 pyccl=3.3.3 python=3.11.15 scikit-learn=1.8.0 scipy=1.17.1 statsmodels=0.14.6 tqdm=4.67.3


mamba activate ${LOCATION}
pip install git+https://github.com/joezuntz/glass

# We need to force this to have no dependencies because we are using my fork
# of glass which the pypi version doesn't know about.
pip install --no-deps glass.ext.camb==2023.6
