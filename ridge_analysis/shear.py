import numpy as np
import h5py
import time
import healpy as hp
from sklearn.neighbors import NearestNeighbors
from .io import ShearMeasurement, RidgeSegmentCatalog, SourceCatalog
from astropy.coordinates import SkyCoord
from astropy import units as u
import numba


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


@numba.njit
def haversine_distance(source_coords, filament_coords):
    """
    Compute the haversine distance between two points on the sphere.
    All inputs are expected to be in radians.

    Supports scalar inputs, same-shaped arrays, and pairwise distances for
    different-length 1D arrays (returns an N1 x N2 matrix).
    """
    ra1 = source_coords[:, 1]
    dec1 = source_coords[:, 0]
    ra2 = filament_coords[:, 1]
    dec2 = filament_coords[:, 0]

    # ra1 = np.asarray(ra1)
    # dec1 = np.asarray(dec1)
    # ra2 = np.asarray(ra2)
    # dec2 = np.asarray(dec2)

    # If both are 1D and have different lengths, compute all pairwise distances.
    # if ra1.ndim == dec1.ndim == ra2.ndim == dec2.ndim == 1 and ra1.size != ra2.size:
    ra1 = ra1[:, None]
    dec1 = dec1[:, None]
    ra2 = ra2[None, :]
    dec2 = dec2[None, :]

    deltalon = ra2 - ra1
    deltalat = dec2 - dec1
    a = np.sin(deltalat / 2) ** 2 + np.cos(dec1) * np.cos(dec2) * np.sin(deltalon / 2) ** 2
    a = np.clip(a, 0.0, 1.0)
    return 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def precompute_pixel_regions(ras, decs, g1, g2, weights, nside_coverage):
    """
    Split up the source catalot into low-resolution healpix pixels for fast lookup later on.

    Parameters
    ----------
    ras : numpy.ndarray
        The RA values of the sources in degrees.
    decs : numpy.ndarray
        The DEC values of the sources in degrees.
    g1 : numpy.ndarray
        The G1 shear values of the sources.
    g2 : numpy.ndarray
        The G2 shear values of the sources.
    nside_coverage : int
        The nside of the low-resolution healpix map to use for coverage.

    Returns
    -------
    pixel_regions : dict
        A dictionary mapping low-resolution healpix pixel indices to tuples of
        (ras, decs, g1, g2) for the sources in that pixel.
    """
    assert np.max(ras) < 3 * np.pi, "RA values should be in radians"

    # 110 arcmin, so as we are only going out to 1 degree that should be enough for any
    # adjacent pixels.
    source_healpix_low_res = hp.ang2pix(nside_coverage, np.pi / 2 - decs, ras)
    unique_pix = np.unique(source_healpix_low_res)
    n_unique_pixels = len(unique_pix)
    print(f"Precomputing source pixel regions: {n_unique_pixels} unique pixels at nside {nside_coverage}")
    # 1715 pixels, so a nice reduction but not too small.
    pixel_regions = {}

    # This takes < 1 minute.
    # I tried speeding it up with numba and by making it a single
    # pass through the data, but it was much slower.
    for i in unique_pix:
        index = np.where(source_healpix_low_res == i)[0]
        pixel_regions[i] = (ras[index], decs[index], g1[index], g2[index], weights[index])

    return pixel_regions


def get_nearby_sources(raf, decf, pixel_regions, nside_coverage, max_ang_rad):
    """
    Extract from the pixel regions the sources that are near to the filament points.
    """
    # we can improve this.
    # we just want the pixels that have any points within max_distance_arcmin of the filament points.
    vecs = hp.ang2vec(np.pi / 2 - decf, raf)
    filament_pix_low_res = []
    for v in vecs:
        ipix = hp.query_disc(nside_coverage, v, max_ang_rad, inclusive=True)
        filament_pix_low_res.append(ipix)

    pixels_needed = np.unique(np.concatenate(filament_pix_low_res))
    # hp.query_disc
    # filament_pix_low_res = hp.ang2pix(nside_coverage, np.pi / 2 - decf, raf)
    # filament_pix_low_res = np.unique(filament_pix_low_res)
    # pixels_needed = hp.get_all_neighbours(nside_coverage, filament_pix_low_res).flatten()

    # Include the filament pixels themselves, and then get unique values
    # pixels_needed = np.unique(np.concatenate((pixels_needed, filament_pix_low_res)))
    ra = []
    dec = []
    g1 = []
    g2 = []
    weight = []

    for p in pixels_needed:
        if p in pixel_regions:
            ras_s, decs_s, g1_s, g2_s, w_s = pixel_regions[p]
            ra.append(ras_s)
            dec.append(decs_s)
            g1.append(g1_s)
            g2.append(g2_s)
            weight.append(w_s)

    if len(ra) == 0:
        return None, None, None, None, None, None

    ra = np.concatenate(ra)
    dec = np.concatenate(dec)
    g1 = np.concatenate(g1)
    g2 = np.concatenate(g2)
    weight = np.concatenate(weight)
    source_coords = np.array([dec, ra]).T

    return source_coords, ra, dec, g1, g2, weight


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
    neighbour_algorithm: str = "brute",
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
    neighbour_algorithm : str
        The algorithm to use for finding nearest neighbors when determining the nearest filament point to each source.
        Must be one of 'auto', 'ball_tree', 'kd_tree', or 'brute'. Default is 'brute'.
    """
    if neighbour_algorithm not in ["auto", "ball_tree", "kd_tree", "brute"]:
        raise ValueError(
            f"Invalid neighbour_algorithm: {neighbour_algorithm}. Must be one of 'auto', 'ball_tree', 'kd_tree', 'brute'."
        )

    min_ang_rad = np.radians(min_distance_arcmin / 60)
    max_ang_rad = np.radians(max_distance_arcmin / 60)

    coverage_pixel_size = hp.nside2resol(nside_coverage, arcmin=True)
    assert (
        coverage_pixel_size > max_distance_arcmin
    ), "Coverage pixel size ({coverage_pixel_size} arcmin) must be larger than max_distance_arcmin ({max_distance_arcmin} arcmin). Decrease nside_coverage."

    start_time = time.time()

    # Ensure RA values are between 0 and 2pi.
    # Every process loads the filament catalog in full.
    # ra_filaments = np.radians(ridge_catalog.ra) % (2 * np.pi)
    # dec_filaments = np.radians(ridge_catalog.dec)
    # labels = ridge_catalog.ridge_id

    # The source catalog is split among the different processes.
    # The should have been done already when loading in the first place,
    # in the calling function.
    ra_background = np.radians(source_catalog.ra) % (2 * np.pi)
    dec_background = np.radians(source_catalog.dec)
    g1_background = source_catalog.g1
    g2_background = source_catalog.g2
    weights_background = source_catalog.weight

    if flip_g1:
        g1_background *= -1
    if flip_g2:
        g2_background *= -1

    bin_sums_plus = np.zeros(num_bins)
    bin_sums_cross = np.zeros(num_bins)
    bin_weighted_distances = np.zeros(num_bins)
    bin_weights = np.zeros(num_bins)
    bin_counts = np.zeros(num_bins)
    rank = comm.rank if comm is not None else 0

    # from sklearn docs:
    # Note that the haversine distance metric requires data in the form of [latitude, longitude]
    # and both inputs and outputs are in units of radians.
    # check units are in degrees
    # assert ra_filaments.max() < 3 * np.pi, "Filament RA values should be in radians {}".format(ra_filaments.max())
    # assert ra_background.max() < 3 * np.pi, "Background RA values should be in radians {}".format(ra_background.max())

    # Pre-split the catalog into a low-resolution map, so that we can just look up pixels
    # that are relatively close to the filament points later.
    pixel_regions = precompute_pixel_regions(
        ra_background,
        dec_background,
        g1_background,
        g2_background,
        weights_background,
        nside_coverage,
    )

    # Angular bins for the shear measurement in radians.
    # We use a similar configuration to galaxy-galaxy lensing.
    bins = np.logspace(np.log10(min_ang_rad), np.log10(max_ang_rad), num_bins + 1)

    start_time = time.time()
    n_filaments = ridge_catalog.metadata["n_filaments"]

    for (filament_index, filament_ra, filament_dec) in ridge_catalog.iterate_ridges(radians=True):
        filament_coords = np.array([filament_dec, filament_ra]).T

        if filament_dec.size < min_filament_points:
            print(f"[{rank}] Skipping filament {filament_index} (label {filament_index}) with only {filament_dec.size} points")
            continue

        # Pull out sources within adjacent low-resolution healpix pixels
        source_coords, ra_subset, dec_subset, g1_subset, g2_subset, weights_subset = get_nearby_sources(
            filament_ra, filament_dec, pixel_regions, nside_coverage, max_ang_rad
        )

        # There may be no sources near this filament segment
        if source_coords is None:
            continue

        if (rank == 0) and (filament_index % 10 == 0):
            print(
                f"[{rank}] Processing filament {filament_index} / {n_filaments} - {source_coords.shape[0]} nearby sources"
            )

        # The brute force algorithm here is faster for the tests I've been doing than a
        # tree, because the number of filament points is not that high. That might change with
        # different ridge choices.
        nbrs = NearestNeighbors(n_neighbors=1, leaf_size=75, metric="haversine", algorithm="brute").fit(filament_coords)
        distances, indices = nbrs.kneighbors(source_coords)
        indices = indices[:, 0]
        distances = distances[:, 0]
        matched_filament_points = filament_coords[indices]

        # Get the rotation angle phi between the background galaxy and the filament point
        phi = get_position_angle(
            ra_source=ra_subset,
            dec_source=dec_subset,
            ra_filament=matched_filament_points[:, 1],
            dec_filament=matched_filament_points[:, 0],
        )

        # Rotate the shear into the filament frame
        g_plus = -g1_subset * np.cos(2 * phi) + g2_subset * np.sin(2 * phi)
        g_cross = g1_subset * np.sin(2 * phi) - g2_subset * np.cos(2 * phi)

        # Bin the distances between 1 arcmin and 1 degree
        bin_indices = np.digitize(distances, bins) - 1
        valid_bins = (bin_indices >= 0) & (bin_indices < num_bins)

        if skip_end_points:
            # optionally skip pairs where the filament point is at the end of the filament
            valid_bins &= (indices > 0) & (indices < filament_coords.shape[0] - 1)

        # Accumulate the total tangential and cross shear in each bin,
        # together with the counts, weights, and actual (as opposed to nominal) distances.
        np.add.at(bin_sums_plus, bin_indices[valid_bins], weights_subset[valid_bins] * g_plus[valid_bins])
        np.add.at(bin_sums_cross, bin_indices[valid_bins], weights_subset[valid_bins] * g_cross[valid_bins])
        np.add.at(bin_weighted_distances, bin_indices[valid_bins], weights_subset[valid_bins] * distances[valid_bins])
        np.add.at(bin_weights, bin_indices[valid_bins], weights_subset[valid_bins])
        np.add.at(bin_counts, bin_indices[valid_bins], 1)

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
