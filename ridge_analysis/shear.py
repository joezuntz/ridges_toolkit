import numpy as np
import time
from .io import ShearMeasurement, RidgeSegmentCatalog, SourceCatalog
import numba
from dredge.tree import HealpixTree


@numba.njit
def get_position_angle(ra_source, dec_source, ra_filament, dec_filament):
    """
    All inputs are expected to be in radians
    """
    deltalon = ra_filament - ra_source
    colat = np.cos(dec_filament)

    x = np.sin(dec_filament) * np.cos(dec_source) - colat * np.sin(dec_source) * np.cos(deltalon)
    y = np.sin(deltalon) * colat
    return np.arctan2(y, x) % (2 * np.pi) + np.pi / 2


def measure_shear(
    ridge_catalog: RidgeSegmentCatalog,
    source_catalog: SourceCatalog,
    output_shear_file: str,
    num_bins: int = 20,
    comm=None,
    flip_g1: bool = False,
    flip_g2: bool = False,
    nside_coverage: int = 32,
    min_distance_arcmin: float = 1.0,
    max_distance_arcmin: float = 60.0,
    skip_end_points: bool = False,
    min_filament_points: int = 0,
    add_sigma_e: float = 0.0,
    seed: int = 0,
):
    """
    Measure the tangential shear in the source catalog around the ridges in the ridge catalog.

    The result is saved to output_shear_file, which is an ASCII file with columns for the separation
    bin centers, the weighted average separation, the tangential shear, the cross shear, the counts,
    and the weights in each bin.

    The separations are saved in arcminutes.

    Parameters
    ----------
    ridge_catalog : RidgePointCatalog
        The catalog of ridge points to measure around.
    source_catalog : SourceCatalog
        The catalog of source galaxies to measure the shear of.
    output_shear_file : str
        The name of the file to write the shear measurement results to.
    k : int
        The number of nearest neighbors to use when determining the distance from sources to filaments. Default is 1 (i.e. use the nearest filament point).
    num_bins : int
        The number of bins to use in the shear measurement. Default is 20.
    comm : mpi4py.MPI.Comm
        The MPI communicator to use for parallel processing. If None, no parallel processing is used.
    flip_g1 : bool
        Whether to flip the sign of g1 when computing the tangential shear. Default is False.
    flip_g2 : bool
        Whether to flip the sign of g2 when computing the tangential shear. Default is False.
    nside_coverage : int
        The nside of the low-resolution healpix map to use for precomputing source regions. Default is 32.
        Higher values will slow the precomputation but may speed up the source lookup later on.
    min_distance_arcmin : float
        The minimum distance from the filament to include in the shear measurement, in arcminutes. Default is 1.0 arcmin.
    max_distance_arcmin : float
        The maximum distance from the filament to include in the shear measurement, in arcminutes. Default is 60.0 arcmin (i.e. 1 degree).
    skip_end_points : bool
        Whether to skip pairs where the filament point is at the end of the filament. Default is False.
    min_filament_points : int
        The minimum number of points in a filament segment to include in the shear measurement. Default is 0 (i.e. include all filaments).
    add_sigma_e: float
        Sigma e per component to add
    seed: int
        Random seed for adding sigma_e
    """

    min_ang_rad = np.radians(min_distance_arcmin / 60)
    max_ang_rad = np.radians(max_distance_arcmin / 60)

    start_time = time.time()

    # The source catalog is split among the different processes.
    # The should have been done already when loading in the first place,
    # in the calling function.

    bin_sums_plus = np.zeros(num_bins)
    bin_sums_cross = np.zeros(num_bins)
    bin_weighted_distances = np.zeros(num_bins)
    bin_weights = np.zeros(num_bins)
    bin_counts = np.zeros(num_bins)
    rank = comm.rank if comm is not None else 0

    # Extract the source catalog columns that we will need
    source_phi = np.radians(source_catalog.ra)
    source_theta = np.pi/2 - np.radians(source_catalog.dec)
    source_g1 = source_catalog.g1
    source_g2 = source_catalog.g2

    if add_sigma_e > 0:
        if rank == 0:
            print(f"Adding noise with sigma_e={add_sigma_e} and seed {seed} to shears")
        rng = np.random.default_rng(seed)
        source_g1 += rng.normal(0, add_sigma_e, source_g1.size)
        source_g2 += rng.normal(0, add_sigma_e, source_g2.size)

    if flip_g1:
        source_g1 *= -1

    if flip_g2:
        source_g2 *= -1


    source_weights = source_catalog.weight
    source_catalog.unload()
    ntotal = source_phi.size

    # Build an object that lets us quickly find nearby points and their distances
    tree = HealpixTree(theta=source_theta, phi=source_phi, nside=nside_coverage)

    # Angular bins for the shear measurement in radians.
    # We use a similar configuration to galaxy-galaxy lensing.
    bins = np.logspace(np.log10(min_ang_rad), np.log10(max_ang_rad), num_bins + 1)

    start_time = time.time()
    n_filaments = ridge_catalog.metadata["n_filaments"]
    if rank == 0:
        print(f"Starting shear measurement with {n_filaments} filaments and {ntotal} sources")

    if rank == 0:
        import tqdm
        pbar = tqdm.tqdm(total=n_filaments)

    for (filament_index, filament_ra, filament_dec) in ridge_catalog.iterate_ridges(radians=True):

        if filament_dec.size < min_filament_points:
            print(f"[{rank}] Skipping filament {filament_index} (label {filament_index}) with only {filament_dec.size} points")
            continue
        filament_theta = np.pi/2 - filament_dec
        filament_phi = filament_ra
        indices, distances, nearest_query_point = tree.query_radii(filament_theta, filament_phi, max_ang_rad)

        # # Pull out sources within adjacent low-resolution healpix pixels
        matched_filament_ra = filament_ra[nearest_query_point]
        matched_filament_dec = filament_dec[nearest_query_point]
        ra_subset = source_phi[indices]
        dec_subset = np.pi/2 - source_theta[indices]
        g1_subset = source_g1[indices]
        g2_subset = source_g2[indices]
        weights_subset = source_weights[indices]

        # Get the rotation angle phi between the background galaxy and the filament point
        phi = get_position_angle(
            ra_source=ra_subset,
            dec_source=dec_subset,
            ra_filament=matched_filament_ra,
            dec_filament=matched_filament_dec,
        )

        # Rotate the shear into the filament frame
        g_plus = -g1_subset * np.cos(2 * phi) + g2_subset * np.sin(2 * phi)
        g_cross = g1_subset * np.sin(2 * phi) - g2_subset * np.cos(2 * phi)

        # Bin the distances between 1 arcmin and 1 degree
        bin_indices = np.digitize(distances, bins) - 1
        valid_bins = (bin_indices >= 0) & (bin_indices < num_bins)

        if skip_end_points:
            # optionally skip pairs where the filament point is at the end of the filament
            valid_bins &= (nearest_query_point > 0) & (nearest_query_point < filament_ra.size - 1)

        # Accumulate the total tangential and cross shear in each bin,
        # together with the counts, weights, and actual (as opposed to nominal) distances.
        np.add.at(bin_sums_plus, bin_indices[valid_bins], weights_subset[valid_bins] * g_plus[valid_bins])
        np.add.at(bin_sums_cross, bin_indices[valid_bins], weights_subset[valid_bins] * g_cross[valid_bins])
        np.add.at(bin_weighted_distances, bin_indices[valid_bins], weights_subset[valid_bins] * distances[valid_bins])
        np.add.at(bin_weights, bin_indices[valid_bins], weights_subset[valid_bins])
        np.add.at(bin_counts, bin_indices[valid_bins], 1)

        if rank == 0:
            pbar.update(1)

    # Sum up the results from all processes, so the totals are correct
    sum_in_place(bin_sums_plus, comm)
    sum_in_place(bin_sums_cross, comm)
    sum_in_place(bin_weighted_distances, comm)
    sum_in_place(bin_weights, comm)
    sum_in_place(bin_counts, comm)

    # Only the root process computes the final division and writes out the final results
    if comm is not None and comm.rank != 0:
        return

    # Avoid NaNs by only dividing where bin_weights > 0
    weighted_g_plus = np.divide(bin_sums_plus, bin_weights, out=np.zeros_like(bin_sums_plus), where=bin_weights > 0)
    weighted_g_cross = np.divide(bin_sums_cross, bin_weights, out=np.zeros_like(bin_sums_cross), where=bin_weights > 0)
    weighted_real_distances = np.divide(
        bin_weighted_distances, bin_weights, out=np.zeros_like(bin_weighted_distances), where=bin_weights > 0
    )

    # Get the nominal bin centers
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    output_table = ShearMeasurement(output_shear_file)
    for i in range(num_bins):
        output_table.data.add_row(
            (
                radians_to_arcmin(bin_centers[i]),
                radians_to_arcmin(weighted_real_distances[i]),
                weighted_g_plus[i],
                weighted_g_cross[i],
                bin_counts[i],
                bin_weights[i],
            )
        )

    print(f"Shear processing completed in {time.time() - start_time:.2f} seconds.")
    return output_table


def radians_to_arcmin(radians):
    return np.degrees(radians) * 60.0


def sum_in_place(data, comm):
    """
    Use MPI to sum up the data from all the different processes in an array.

    Parameters
    ----------
    data : numpy.ndarray
        The data to sum up.

    comm : mpi4py.MPI.Comm
        The MPI communicator to use. If None, this function does nothing.
    """
    if comm is None:
        return

    import mpi4py.MPI

    if comm.Get_rank() == 0:
        comm.Reduce(mpi4py.MPI.IN_PLACE, data)
    else:
        comm.Reduce(data, None)
