import ridge_analysis
import ridge_analysis.plots
import yaml


def simulate(global_config, simulation_config):
    """
    Run the GLASS-based simulation code to generate the lens and source catalogs.

    Generates output catalogs in the catalog_dir in the global configuration.

    Currently this ignores the lens_bins and source_bins in the global configuration,
    and always generates a single source and lens_catalog. There are options we could
    expose in the simulation config to change this if needed.

    Parameters
    ----------
    global_config: dict
        The global configuration dictionary, read from the YAML file. This is used to
        get the catalog_dir where the outputs will be saved.

    simulation_config: dict
        The simulation configuration dictionary, read from the YAML file. This is used to
        set the parameters for the simulation, such as nside, dx, cosmological parameters, etc.    
    """
    # Build the configuration object
    import ridge_sims
    sim_dir = global_config["catalog_dir"]
    config = ridge_sims.Config(sim_dir=sim_dir, **simulation_config)
    config.save()

    # Run the three steps of the simulation. The first is by far the slowest.
    # If the output already exists then these will skip the computation,
    # so this is safe to run multiple times.
    ridge_sims.step1(config)
    ridge_sims.step2(config)
    ridge_sims.step3(config)


def locate_ridges(global_config, dredge_config):
    """
    Locate ridge points from the lens catalogs.

    Finds catalogs in the catalog_dir in the global configuration
    and outputs to the ridge_dir.

    Makes one ridge point file per lens_bin.

    This is internally parallelized with MPI, so can be run with mpirun.

    Parameters
    ----------
    global_config: dict
        The global configuration dictionary, read from the YAML file. This is used to
        get the directories and lists of bins.

    dredge_config: dict
        The configuration for the ridge point location step, read from the YAML file. This is used
        to set parameters for the ridge point location, such as the number of ridge points to find, 
        the random seed, etc.    
    """
    import mpi4py.MPI as MPI
    catalog_dir = global_config["catalog_dir"]
    ridge_dir = global_config["ridge_dir"]
    bins = global_config["lens_bins"]
    comm = MPI.COMM_WORLD

    # Loop through lens bins making one config and ridge catalog for each one.
    for b in bins:
        config = {
            "lens_catalog_file": f"{catalog_dir}/lens_catalog_{b}.hdf5",
            "ridge_point_file": f"{ridge_dir}/ridge_points_{b}.hdf5",
            "checkpoint_dir": f"{ridge_dir}/checkpoints_{b}",
        }
    
        config = ridge_analysis.DredgeConfig(**config, **dredge_config)
        ridge_analysis.locate_ridge_points(config, comm=comm)


def segment_ridges(global_config, segmentation_config):
    """
    Segment ridge points into separate ridges.

    Finds ridge point catalogs in the ridge_dir and outputs ridge segment catalogs
    in the segment_dir, both specified in the global configuration.

    The segmentation is split among processes by lens bin, but this
    is not really necessary since the segmentation is very fast.
    You can run this on a single process without MPI if you want.

    Parameters
    ----------
    global_config: dict
        The global configuration dictionary, read from the YAML file. This is used to
        get the directories and lists of bins.

    segmentation_config: dict
        The configuration for the ridge segmentation step, read from the YAML file. This is used to
        set parameters for the ridge segmentation, such as the minimum number of points in a ridge, 
        the maximum distance between points, etc.
    """
    ridge_dir = global_config["ridge_dir"]
    segment_dir = global_config["segment_dir"]
    bins = global_config["lens_bins"]

    import mpi4py.MPI as MPI
    comm = MPI.COMM_WORLD

    for b in bins:
        
        config = {
            "ridge_point_file": f"{ridge_dir}/ridge_points_{b}.hdf5",
            "ridge_file": f"{segment_dir}/ridges_{b}.hdf5",
        }
    
        config = ridge_analysis.SegmentationConfig(**config, **segmentation_config)
        ridge_analysis.segment_ridges(config, comm=comm)

def plot(global_config, plot_config):
    """
    Plot the ridges on top of the density map.
    TODO: plot more things.

    You can run this under MPI if you want but only
    the rank 0 process will do anything.

    Parameters
    ----------
    global_config: dict
        The global configuration dictionary, read from the YAML file. This is used to
        get the directories and lists of bins.

    plot_config: dict
        The configuration for the plotting step, read from the YAML file. This is used to
        set parameters for the plotting, such as the nside for the density map, the smoothing scale.    
    """
    catalog_dir = global_config["catalog_dir"]
    ridge_dir = global_config["ridge_dir"]
    plot_dir = global_config["plot_dir"]
    bins = global_config["lens_bins"]
    nside = plot_config["nside"]
    smoothing_degrees = plot_config["smoothing_degrees"]

    import mpi4py.MPI as MPI
    comm = MPI.COMM_WORLD

    for i, b in enumerate(bins):
        if i % comm.size != comm.rank:
            continue

        dredge_config = ridge_analysis.DredgeConfig(
            lens_catalog_file=f"{catalog_dir}/lens_catalog_{b}.hdf5",
            ridge_point_file=f"{ridge_dir}/ridge_points_{b}.hdf5",
            checkpoint_dir=f"{ridge_dir}/checkpoints_{b}",
        )

        segmentation_config = ridge_analysis.SegmentationConfig(
            ridge_point_file=f"{ridge_dir}/ridge_points_{b}.hdf5",
            ridge_file=f"{plot_dir}/ridges_{b}.hdf5",
        )

        plot_filename = f"{plot_dir}/ridge_plot_{b}.png"
        ridge_analysis.plots.plot_segments_on_density(plot_filename, dredge_config, segmentation_config, nside, smoothing_degrees)



def shear(global_config, shear_config):
    """
    Compute the shear around the ridges.
    
    The shear catalogs are found in the catalog_dir and the ridge catalogs
    are found in the ridge_dir, both specified in the global configuration.

    The outputs are put in shear_dir.  There are n_lens * n_source output files,
    and all will be run. This is internally parallelized with MPI, and it 
    gives a large speed-up.

    Parameters
    ----------
    global_config: dict
        The global configuration dictionary, read from the YAML file. This is used to
        get the directories and lists of bins.

    shear_config: dict
        The configuration for the shear step, read from the YAML file. This is used to
        set parameters for the shear computation, such as the binning.

    """
    catalog_dir = global_config["catalog_dir"]
    ridge_dir = global_config["ridge_dir"]
    shear_dir = global_config["shear_dir"]
    lens_bins = global_config["lens_bins"]
    source_bins = global_config["source_bins"]

    import mpi4py.MPI as MPI
    comm = MPI.COMM_WORLD

    for b in lens_bins:
        for s in source_bins:
            config = {
                "output_shear_file": f"{shear_dir}/shear_lens{b}_source{s}.txt",
                "source_catalog_file": f"{catalog_dir}/source_catalog_{s}.hdf5",
                "ridge_file": f"{ridge_dir}/ridges_{b}.hdf5",
            }
        
            config = ridge_analysis.ShearConfig(**config, **shear_config)
            ridge_analysis.measure_ridge_shear(config, comm=comm)


def main(config_file, action):
    # read the configuration file
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # decide which step to do based on the command-line argument
    if action == "simulate":
        simulate(config["global"], config["simulate"])
    elif action == "dredge":
        locate_ridges(config["global"], config["dredge"])
    elif action == "segment":
        segment_ridges(config["global"], config["segment"])
    elif action == "plot":
        plot(config["global"], config["plot"])
    elif action == "shear":
        shear(config["global"], config["shear"])
    else:
        raise ValueError(f"Unknown action: {action}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the ridge analysis pipeline.")
    parser.add_argument("config_file", type=str, help="Path to the configuration file")
    parser.add_argument("action", type=str, help="Action to perform: simulate, ridges, segment, plot, shear")
    args = parser.parse_args()
    main(args.config_file, args.action)