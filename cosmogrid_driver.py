import ridge_analysis
import glob
import os
import traceback
from timeit import default_timer
import argparse

base_dir = "/pscratch/sd/z/zuntz/ridges/v1"
base_catalog_dir = os.path.join(base_dir, "catalogs")
base_ridge_point_dir = os.path.join(base_dir, "ridges")
base_ridge_segment_dir = os.path.join(base_dir, "segments")
base_shear_dir = os.path.join(base_dir, "shear")

MAP_NSIDE = 1024
ADD_NOISE = False

num_groups = 1
group = 0



#Pairs where the source is behind the lens, as determined
# from a signal-to-noise plot
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

dredge_config = dict(
    # not sure if this is enough.
    # will do a test run now with this to check speed
    num_ridge_points = 1000000,
    shift_180 = True,
    tree_nside = 128,
    bandwidth = 6.0, # arcmin
    convergence = 0.03,
)

segmentation_config = dict(
    # we probably want to investigate this
    density_percentile=15.0,
    mst_neighbours=10,
    do_spline=False,
    n_spline_points=100,
)

shear_config = dict(
    flip_g1=False,
    flip_g2=False,
    num_bins=20,
    min_distance_arcmin=1.0,
    max_distance_arcmin=60.0,
    nside_coverage=128,
)
if ADD_NOISE:
    shear_config['add_sigma_e'] = 0.26

class AnalysisStep:
    input_base = ""
    output_base = ""
    uses_comm_internally = True
    
    def __init__(self, permutations=(0,)):
        self.permutations = permutations

    def run(self, task_index, input_dir, output_dir, comm):
        pass

    def main(self, comm):
        cosmo_dirs = sorted(glob.glob("cosmo_*", root_dir=self.input_base))
        # split the whole collection of cosmo_dirs
        # among the different processes
        cosmo_dirs = cosmo_dirs[group::num_groups]
        for perm in self.permutations:
            for i, cosmo_dir in enumerate(cosmo_dirs):
                # Skip cosmologies that are the responsibility
                # of another process
                if not self.uses_comm_internally:
                    if i % comm.size != comm.rank:
                        continue
                # Skip cosmologies where either the output already
                # exists or the input does not exist

                input_dir = os.path.join(self.input_base, cosmo_dir)
                output_dir = os.path.join(self.output_base, cosmo_dir)

                # Check we can do this task and it's not done already
                marker_file = self.marker_file_check(input_dir, output_dir, perm)
                if marker_file is None:
                    continue

                # Run the actual main task
                start_time = default_timer()
                self.run(i, input_dir, output_dir, perm, comm)

                time_taken = default_timer() - start_time
                print(f"{self.__class__.__name__} took {time_taken:.1f} seconds")

                # mark this one as done
                open(marker_file, "w").close()


    def marker_file_check(self, input_dir, output_dir, permutation):
        # check for a marker saying the input data
        # is complete so we can do this task
        input_marker_file = os.path.join(input_dir, f"complete.{permutation}")
        if not os.path.exists(input_marker_file):
            return None
        
        # check for a file marker saying the output is
        # done already. Skip the task if so.
        os.makedirs(output_dir, exist_ok=True)
        output_marker_file = os.path.join(output_dir, f"complete.{permutation}")
        if os.path.exists(output_marker_file):
            return None

        return output_marker_file



class RidgesStep(AnalysisStep):
    input_base = base_catalog_dir
    output_base = base_ridge_point_dir
    uses_comm_internally = True

    def run(self, task_index, input_dir, output_dir, permutation, comm):
        bins = [0, 1, 2, 3]
        nside = MAP_NSIDE
        seed = 91648 + task_index + permutation

        # Loop through lens bins making one config and ridge catalog for each one.
        for b in bins:
            config = {
                "lens_catalog_file": f"{input_dir}/perm_{permutation:04d}_lens_catalog_{nside}_{b}.hdf5",
                "ridge_point_file": f"{output_dir}/perm_{permutation:04d}_ridge_points_{b}.hdf5",
                "checkpoint_dir": None,  # quick enough that we don't bother checkpointing
                "seed": seed,
            }
            if comm is None or comm.rank == 0:
                print(f"Running dredge {config['lens_catalog_file']} -> {config['ridge_point_file']}")

            config = ridge_analysis.DredgeConfig(**config, **dredge_config)
            ridge_analysis.locate_ridge_points(config, comm=comm)


class SegmentationStep(AnalysisStep):
    input_base = base_ridge_point_dir
    output_base = base_ridge_segment_dir
    uses_comm_internally = False # this task is just embarassingly parallel

    def run(self, task_index, input_dir, output_dir, permutation, comm):
        bins = [0, 1, 2, 3]
        print("NOTE: USING DENSITY PERCENTILE: ", segmentation_config["density_percentile"])
        print(f"Rank {comm.rank} doing {input_dir}->{output_dir}")
        for b in bins:
            config = {
                "ridge_point_file": f"{input_dir}/perm_{permutation:04d}_ridge_points_{b}.hdf5",
                "ridge_file": f"{output_dir}/perm_{permutation:04d}_ridges_{b}.hdf5",
            }
            config = ridge_analysis.SegmentationConfig(**config, **segmentation_config)
            # we pass in comm=None here because when we are not doing
            # any spline fitting this task is serial and we just want to do it
            # at the embarassingly parallel level
            ridge_analysis.segment_ridges(config, comm=None)
            


class ShearStep(AnalysisStep):
    input_base = base_ridge_segment_dir
    output_base = base_shear_dir
    catalog_base = base_catalog_dir
    uses_comm_internally = True

    def run(self, task_index, input_dir, output_dir, permutation, comm):
        cosmo_dir = input_dir.removeprefix(self.input_base).lstrip(os.path.sep)

        cat_dir = os.path.join(self.catalog_base, cosmo_dir)
        nside = MAP_NSIDE
        base_seed = 7876

        if ADD_NOISE and ((comm is None) or (comm.rank == 0)):
            print("ADDING NOISE!")

        for (l, s) in shear_lens_source_pairs_to_do:
            config = {
                "ridge_file": f"{input_dir}/perm_{permutation:04d}_ridges_{l}.hdf5",
                "source_catalog_file": f"{cat_dir}/perm_{permutation:04d}_source_catalog_{nside}_{s}.hdf5",
                "output_shear_file": f"{output_dir}/perm_{permutation:04d}_shear_lens{l}_source{s}.txt",
                "seed": [base_seed, task_index, permutation, l, s],
            }
            if comm is None or comm.rank == 0:
                print(f"Running shear {cosmo_dir} lens bin {l} source bin {s}")
            config = ridge_analysis.ShearConfig(**config, **shear_config)
            ridge_analysis.measure_ridge_shear(config, comm=comm)
                


def main(action):
    from mpi4py.MPI import COMM_WORLD as comm
    if action == "ridges":
        step = RidgesStep()
    elif action == "segment":
        step = SegmentationStep()
    elif action == "shear":
        step = ShearStep()
    else:
        raise ValueError("Unknown action " + action)
    
    step.main(comm)



parser = argparse.ArgumentParser()
parser.add_argument("action", type=str, default="ridges", help="Action to perform: ridges, segment, shear")
parser.add_argument("--group", type=int, default=0, help="Index if doing multiple runs")
parser.add_argument("--num-groups", type=int, default=1, help="Num groups if doing multiple node runs")

if __name__ == "__main__":
    args = parser.parse_args()
    group = args.group
    num_groups = args.num_groups
    main(args.action)
