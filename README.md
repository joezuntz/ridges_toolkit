# Ridges Toolkit

The toolkit here is for simulating catalogs and then analyzing ridge lensing within them.

## Installation

Run the script `./build_env.sh` to generate a conda-forge environment with the required contents.

By default it will create a named environment called `ridges` in your standard conda location, but it can also create one
in a local directory if you use the flag `-d ./env`

The `mamba` commant needs to be in your environment for this to work.

Then activate it your environment  with `conda activate ridges`  or `conda activate ./env`.
Or `mamba activate` will work too.

<!-- mamba create  -c conda-forge -p ./env camb=1.6.5 fitsio=1.3.0 h5py=3.16.0 healpix=2025.1 healpy=1.19.0 ipython=9.10.1 matplotlib-base=3.10.8 mpi4py=4.1.1 namaster=2.7 networkx=3.6.1 numba=0.65.0 numpy=2.4.3 pandas=3.0.2 pyccl=3.3.3 python=3.11.15 scikit-learn=1.8.0 scipy=1.17.1 statsmodels=0.14.6 tqdm=4.67.3 -->

## Usage

The main script is `ridges.py` and the configuration info it runs on is in `config.yml`.

## Parallelism

Different stages of the code work in different ways in parallel.

### Simulations

You may not need to use the simulation code if we work entirely with CosmoGrid sims here, but I'll explain what happens here for completeness.

- step1 of the simulations is the slowest, calcuating the matter C_ell values in narrow shells. This uses camb so is OpenMP parallel. You control its parallelism by setting the environment variable OMP_NUM_THREADS to the number of processors you want to use.
- step2 uses multiprocessing. You control it using the configuration setting "nprocess"
- step3 is not set up to run in parallel as it is fast.

So when running the simulation step on 32 processes you would do something like:

```
export OMP_NUM_THREADS
python ridges.py config.yml simulate
```

and in the `simulate` part of the config set `config.nprocess=32`.


### Analysis steps


The analysis steps (finding ridges, segmenting them, calculating shear, and plotting)
are all MPI parallel. Run them with

Run it with 

```
export OMP_NUM_THREADS=1
mpirun -n 32 python ridges.py config.yml dredge
```

(or replace `dredge` with `segment`, `plot`, or `shear`).
