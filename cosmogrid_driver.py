import ridge_analysis
import glob
import os
import traceback
from timeit import default_timer

base_catalog_dir = "/pscratch/sd/z/zuntz/ridges/v1"
base_output_dir = "/pscratch/sd/z/zuntz/ridges/v1-analysis1"

dredge_config = dict(
    # not sure if this is enough.
    # will do a test run now with this to check speed
    num_ridge_points = 1000000,
    shift_180 = True,
    tree_nside = 128,
    bandwidth = 6.0, # arcmin
    convergence = 0.03,
)

def locate_ridges(cosmogrid_run, cosmogrid_index, permutation, comm, ridge_dir):
    catalog_dir = os.path.join(base_catalog_dir, cosmogrid_run)
    bins = [0, 1, 2, 3]
    nside = 1024
    seed = 91648 + cosmogrid_index + permutation

    # Loop through lens bins making one config and ridge catalog for each one.
    for b in bins:
        config = {
            "lens_catalog_file": f"{catalog_dir}/perm_{permutation:04d}_lens_catalog_{nside}_{b}.hdf5",
            "ridge_point_file": f"{ridge_dir}/perm_{permutation:04d}_ridge_points_{b}.hdf5",
            # this is now quick enough that we don't bother checkpointing
            "checkpoint_dir": None,
            "seed": seed,
        }
        if comm is None or comm.rank == 0:
            print(f"Running dredge {config['lens_catalog_file']} -> {config['ridge_point_file']}")

        config = ridge_analysis.DredgeConfig(**config, **dredge_config)
        ridge_analysis.locate_ridge_points(config, comm=comm)


def main():
    from mpi4py.MPI import COMM_WORLD as comm
    cosmo_dirs = sorted(glob.glob("cosmo_*", root_dir=base_catalog_dir))
    permutation = 0

    for i, cosmo_dir in enumerate(cosmo_dirs):
        t0 = default_timer()
        # check if this cosmology actually has a completed set of catalogs
        cat_marker_file = os.path.join(base_catalog_dir, cosmo_dir, f"complete.{permutation}")
        if not os.path.exists(cat_marker_file):
            continue

        # Make the directory where the outputs will go
        output_dir = os.path.join(base_output_dir, cosmo_dir)
        os.makedirs(output_dir, exist_ok=True)

        # check if this one is already complete.
        # skip if so.
        output_marker_file = os.path.join(output_dir, f"complete.{permutation}")
        if os.path.exists(output_marker_file):
            continue

        # if not, run dredge
        try:
            locate_ridges(cosmo_dir, i, permutation, comm, output_dir)
            # make the marker file indicating we have finished
            open(output_marker_file, "w").close()
        except Exception:
            print(traceback.format_exc())
        t1 = default_timer()
        if comm.rank == 0:
            print(f"Run on {cosmo_dir} took {(t1-t0)/60:.2f} minutes")




if __name__ == "__main__":
    main()
