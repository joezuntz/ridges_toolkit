import sys
# sys.path.append("/Users/jzuntz/src/ridges/ridges2")
import ridge_analysis
import mpi4py.MPI
comm = mpi4py.MPI.COMM_WORLD
# maglim_file = "/ceph/cephfs/jzuntz/ridges/des-data/DESY3_maglim_redmagic_v0.5.1.h5"
maglim_file = "./DESY3_maglim_trimmed.hdf5"
output_file = "./DESY3_maglim_ridges.hdf5"
# comm = None

config = ridge_analysis.DredgeConfig(
    lens_catalog_file=maglim_file,
    lens_catalog_group="/",
    ridge_point_file=output_file,
    checkpoint_dir="./des-y3-checkpoints",
    num_ridge_points=1_000_000,
    convergence=0.01,
    seed=4796,
    shift_180=True,
)


ridge_analysis.locate_ridge_points(config, comm)
