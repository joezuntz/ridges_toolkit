import os
import numpy as np
import re

def load_ridge_state(checkpoint_dir, comm):
    """
    Load the last saved ridge state from the specified directory.

    This function looks for the most recent ridge state file in the given
    directory and loads it. It assumes that the files are named in a way
    that allows sorting by iteration number.

    Parameters:
    -----------
    checkpoint_dir : str
        The directory where ridge state files are stored.

    Returns:
    --------
    ridges : numpy.ndarray
        The loaded ridge points from the most recent file.
    """
    rank = 0 if comm is None else comm.rank

    # All files in the checkpoint dir. Below we will filter them
    files = os.listdir(checkpoint_dir)

    # filter to find files that match the pattern 'ridges_<number>_<rank>.npz'
    # i.e. for this rank find the most recent checkpoint file
    checkpoint_iteration = -1
    for f in files:
        match = re.match(fr'ridges_(\d+)_{rank}\.npz', f)
        if match:
            checkpoint_iteration = max(checkpoint_iteration, int(match.group(1)))

    if checkpoint_iteration > 0:
        # If any checkpoints were found, load the most recent one
        filename = f"{checkpoint_dir}/ridges_{checkpoint_iteration}_{rank}.npz"
        print(f"Loading ridge state from {filename} for rank {rank}.")
        ridges_data = np.load(filename)
        state = ridges_data['ridges'], ridges_data['points_to_update'], checkpoint_iteration + 1

    else:
        print(f"Warning: No ridge state files found in {checkpoint_dir} for rank {rank} so cannot resume iterations.")
        state = None

    return state



def checkpoint(checkpoint_dir, iteration_number, ridges, points_to_update, comm):
    """
    Save the progress of the ridge estimation to a checkpoint file.

    Parameters
    -----------
    checkpoint_dir : str or None
        The directory where the checkpoint files will be saved.
        If None, no checkpoint will be saved.

    iteration_number : int
        The current iteration number, used to name the checkpoint file.

    ridges : numpy.ndarray
        The current state of the ridge points to be saved.

    points_to_update : numpy.ndarray
        A mask of points that need to be updated in the next iteration.

    comm : mpi4py.MPI.Comm or None
        The MPI communicator. If provided, the function will gather
        the ridge points from all processes before saving.
    """
    if checkpoint_dir is None:
        return

    # If needed, collect together ridge information
    rank = 0 if comm is None else comm.rank
    np.savez(f"{checkpoint_dir}/ridges_{iteration_number}_{rank}.npz", ridges=ridges, points_to_update=points_to_update)
