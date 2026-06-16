import os
import sys
from timeit import default_timer as timer
from .bandwidth import estimate_bandwidth
import numpy as np
from .checkpoints import checkpoint, load_ridge_state
from .tree import make_tree, cut_points_with_tree
from .scms import ridge_update, mesh_generation


def find_filaments(
    coordinates,
    bandwidth=np.radians(0.5),
    convergence=np.radians(0.005),
    max_unconverged_fraction=0.005,
    max_iterations=5000,
    num_ridge_points=None,
    tree_nside=128,
    mesh_threshold=4.0,
    checkpoint_dir=None,
    min_checkpoint_gap=30,
    resume=False,
    seed=None,
    tree_file=None,
    comm=None,
    weights=None,
):
    """Estimate density rigdges for a user-provided dataset of coordinates.

    This function uses an augmented version of the subspace-constrained mean
    shift algorithm to return density ridges for a set of user-provided
    coordinates. Apart from the haversine distance to compute a more accurate
    version of a common optimal kernel bandwidth calculation in criminology,
    the code also features thresholding to avoid ridges in sparsely populated
    areas. While only the coordinate set is a required input, the user can
    override the number of nearest neighbors used to calculate the bandwidth
    and the bandwidth itself, as well as the convergence threshold used to
    assess when to terminate and the percentage indicating which top-level of
    filament points in high-density regions should be returned. If the latter
    is not chose, all filament points are returned in the output instead.

    Parameters:
    -----------
    coordinates : array-like
        The set of latitudes and longitudes as an (npoint, 2) array of floats.
        In radians.

    bandwidth : float
        The bandwidth used for weighting points in the ridge update.
        In radians

    convergence : float
        The convergence threshold for the inter-iteration update difference.
        In radians.

    max_unconverged_fraction : float, defaults to 0.001
        The fraction of points that are allowed to remain unconverged
        before the iterations stop. This is used to avoid running for too long
        when the points are very close to converging, but not quite there,
        or are oscillating around target points.

    max_iterations : int, defaults to 5000
        The maximum number of iterations to run before stopping.

    num_ridge_points : int, defaults to None
        The number of mesh points to be used to generate ridges.

    tree_nside : int, defaults to 128
        The nside parameter of the tree-like structure used in finding
        nearby points.

    distance_metric: string, defaults to 'haversine'
        The distance function to be used, can be 'haversine' or 'euclidean'.

    mesh_threshold: float, defaults to 4.0
        Throw away initial mesh point more than this many bandwidths from any coordinate

    checkpoint_dir: str, defaults to None
        If provided, the directory where the intermediate results will be saved.
        This allows for resuming the computation from a previous state.

    min_checkpoint_gap: int, defaults to 30
        The minimum time in seconds between checkpoints. This prevents
        excessive checkpointing during the iterations, which can things down,
        especially towards the end.

    resume: bool, defaults to False
        If True, the function will attempt to resume from a previous run
        in the checkpoint_dir

    seed: int, defaults to None
        If provided, sets the random seed for reproducibility.
        Required if using checkpointing as otherwise the resumed
        run will not match the original run.

    tree_file: str, defaults to None
        If provided, the file where the BallTree will be saved or loaded from.
        If the file exists, it will be loaded; otherwise, a new tree will be created.
        This is mainly useful when testing.

    comm: mpi4py.MPI.Comm, defaults to None
        If provided, the MPI communicator to use for parallel processing.
        If None, the function will run in single-process mode.

    weights: None or array-like, defaults to None
        If provided, an array of weights for the input coordinates. This can be used to give
        more importance to certain points in the ridge estimation. The array should have the
        same length as the number of input coordinates.

    Returns:
    --------
    ridges : array
        The coordinates for the estimated density ridges of the data.

    initial_density : array
        The initial density of the mesh points before iterations.

    final_density : array
        The final density of the ridge points after iterations.

    Attributes:
    -----------
    None
    """
    # Set a mesh size if none is provided by the user
    if num_ridge_points is None:
        num_ridge_points = int(np.min([1e5, np.max([5e4, len(coordinates)])]))

    parameter_check(**locals())

    # parallelization information
    is_root = comm is None or comm.rank == 0
    rank = 0 if comm is None else comm.rank
    comm_size = 1 if comm is None else comm.size

    # make checkpointing directory if it does not exist
    if checkpoint_dir is not None and is_root:
        os.makedirs(checkpoint_dir, exist_ok=True)

    # Only the root process makes the tree, so that
    # every process is using the exact same one.
    if is_root:
        # Make the ball tree to speed up finding nearby points
        print("Making tree.")
        tree = make_tree(coordinates, tree_nside)
    else:
        tree = None

    if comm is not None:
        # These commands send a copy of the tree
        # to all the processes
        tree = comm.bcast(tree, root=0)

    my_num_ridge_points = num_ridge_points // comm_size
    # Create an evenly-spaced mesh in for the provided coordinates
    seed_for_rank = None if seed is None else (seed, rank)
    ridges = mesh_generation(coordinates, my_num_ridge_points, seed_for_rank)
# remove any ridges that are more than mesh_threshold bandwidths from any point
# This cut runs independently on each rank (after broadcasting the tree).
    if is_root:
        print(f"Cutting initial mesh to points within {mesh_threshold} bandwidths of a galaxy")
    ridges = cut_points_with_tree(ridges, tree, bandwidth, threshold=mesh_threshold)
    if is_root:
        print(f"Finished cutting. {ridges.shape[0]} mesh points remain.")


    # Record the initial density of all the points to allow us to do cuts later
    initial_density = estimate_local_density(tree, ridges, bandwidth, weights)

    # Loop over the number of prescripted iterations
    iteration_number = 0

    # We keep track of which points need to be updated.
    # Initially, all points are set to be updated.
    points_to_update = np.ones(ridges.shape[0], dtype=bool)

    # Checkpoint 0 - initial random state of the points
    checkpoint(checkpoint_dir, iteration_number, ridges, points_to_update, comm)
    time_of_last_checkpoint = timer()

    # Optionally restart from an incomplete run
    if resume and checkpoint_dir is not None:
        state = load_ridge_state(checkpoint_dir, comm)
        if state is not None:
            ridges, points_to_update, iteration_number = state

    index = np.arange(ridges.shape[0])
    n_to_update = points_to_update.sum()
    print(f"[Proc {rank}]: iteration {iteration_number}  {n_to_update} points to iterate.")
    fail_mask = np.zeros(ridges.shape[0], dtype=bool)

    # Wait until almost all points have converged. Towards the end the
    # points will be updating very quickly as there are so few of them.
    while (iteration_number < max_iterations) and points_to_update.sum() > np.ceil(
        points_to_update.size * max_unconverged_fraction
    ):
        # pull out the set of points that we want to update
        ridges_subset = ridges[points_to_update]
        subset_index = index[points_to_update]

        # Update the points in the mesh. Record the timing
        t = timer()
        updates, subset_fail_mask = ridge_update(ridges_subset, coordinates, bandwidth, tree)
        time_taken = timer() - t

        # Copy the update from this set of ridges to the full set
        ridges[subset_index] = ridges_subset

        # Some points will get zero updates because they have no points in the bandwidth.
        # This can happen when the previous update moved them to a masked
        # region. These points will automatically be removed from the updating
        # process because they didn't move, but we also want to not
        # return them at the end of the process, so mark them here.
        fail_mask[subset_index] = subset_fail_mask

        # Find which points have converged, and mark in the list of
        # the full set of ridges that they have done so.
        update_average = np.mean(updates)
        newly_converged_points = updates < convergence
        points_to_update[subset_index[newly_converged_points]] = False
        n_to_update = points_to_update.sum()

        # Report convergence progress.
        iteration_number += 1
        print(
            f"[Proc {rank}]: iteration {iteration_number}  update change: {update_average:e} took {time_taken:.2f} seconds. {n_to_update} points left to converge."
        )
        # It's worth running flush here as on some MPI systems this can actually force the write.  Not all though.
        sys.stdout.flush()

        # The checkpointing is done independently for each MPI
        # process. This means you need to use the same number of
        # processes when resuming a run as you did when starting it,
        # but it also avoids requiring any synchronization.
        if checkpoint_dir is not None:
            if timer() - time_of_last_checkpoint > min_checkpoint_gap:
                time_of_last_checkpoint = timer()
                # Save the current state of the ridges and points to update
                checkpoint(checkpoint_dir, iteration_number, ridges, points_to_update, comm)

    nfail = fail_mask.sum()
    print(f"Rank {rank} finshed and removing {nfail}/{fail_mask.size} points that lost neighbours")
    ridges = ridges[~fail_mask]

    # We also record the density at the end of the iterations - we may want to cut
    # to high density ridges only.
    final_density = estimate_local_density(tree, ridges, bandwidth, weights)

    if comm is not None:
        ridges = comm.gather(ridges)
        initial_density = comm.gather(initial_density)
        final_density = comm.gather(final_density)

        if is_root:
            ridges = np.concatenate(ridges)
            initial_density = np.concatenate(initial_density)
            final_density = np.concatenate(final_density)

    # Return the iteratively updated mesh as the density ridges
    if is_root:
        print("\nDone!")

    return ridges, initial_density, final_density


def estimate_local_density(tree, coordinates, bandwidth, weights=None):
    density = np.zeros(len(coordinates))
    for i, (theta, phi) in enumerate(coordinates):
        theta = np.pi/2 - theta
        points, _ =tree.query_radius(theta, phi, bandwidth)
        if weights is None:
            density[i] = len(points)
        else:
            density[i] = weights[points].sum()
    return density


def parameter_check(**p):
    """Check the main function inputs for unsuitable formats or values.

    This function checks all of the user-provided main function inputs for
    their suitability to be used by the code. This is done right at the
    top of the main function to catch input errors early and before any
    time is spent on time-consuming computations. Each faulty input is
    identified, and a customized error message is printed for the user
    to inform about the correct inputs before the code is terminated.

    Parameters:
    -----------
    p : dict
        A dictionary of the user-provided parameters for the main function.
    """

    coordinates = p["coordinates"]
    bandwidth = p["bandwidth"]
    convergence = p["convergence"]
    num_ridge_points = p["num_ridge_points"]
    tree_nside = p["tree_nside"]
    mesh_threshold = p["mesh_threshold"]
    checkpoint_dir = p["checkpoint_dir"]
    resume = p["resume"]
    seed = p["seed"]
    tree_file = p["tree_file"]
    comm = p["comm"]

    # Check whether two-dimensional coordinates are provided
    if not (isinstance(coordinates, np.ndarray) and coordinates.ndim == 2 and coordinates.shape[1] == 2):
        raise ValueError("ERROR: coordinates must be a 2-column numpy.ndarray with second dimension 2 (dec, ra)")

    if np.any(coordinates > 2 * np.pi) or np.any(coordinates < -2 * np.pi):
        raise ValueError("ERROR: coordinates must be in radians. ")

    if np.any(np.isnan(coordinates)) or np.any(np.isinf(coordinates)):
        raise ValueError("ERROR: coordinates must not contain NaN or Inf values.")

    # Check whether tree_nside is a positive integer
    if not (isinstance(tree_nside, int) and tree_nside > 0):
        raise ValueError("ERROR: tree_nside must be a positive integer")

    # Check whether bandwidth is a positive float
    if not (isinstance(bandwidth, float) and bandwidth > 0):
        raise ValueError("ERROR: bandwidth must be a float > 0")

    # Check whether convergence is a positive float
    if not (isinstance(convergence, float) and convergence > 0):
        raise ValueError("ERROR: convergence must be a positive integer or float")

    # Check whether num_ridge_points is a positive integer
    if not (isinstance(num_ridge_points, int) and num_ridge_points > 0):
        raise ValueError("ERROR: num_ridge_points must be a positive integer")

    # Check whether mesh_threshold is a positive float or int
    if not (isinstance(mesh_threshold, (float, int)) and mesh_threshold > 0):
        raise ValueError("ERROR: mesh_threshold must be a positive float or int")

    # Check whether checkpoint_dir is a string or None
    if checkpoint_dir is not None and not isinstance(checkpoint_dir, str):
        raise ValueError("ERROR: checkpoint_dir must be a string or None")

    # Check whether resume is a boolean
    if not isinstance(resume, bool):
        raise ValueError("ERROR: resume must be a boolean")

    # Check whether seed is an integer or None
    if seed is not None and not isinstance(seed, int):
        raise ValueError("ERROR: seed must be an integer or None")

    # Check whether tree_file is a string or None
    if tree_file is not None and not isinstance(tree_file, str):
        raise ValueError("ERROR: tree_file must be a string or None")

    # Check whether comm is None or an MPI communicator
    if comm is not None and not hasattr(comm, "rank"):
        raise ValueError("ERROR: comm must be None or an MPI communicator (e.g., from mpi4py)")
