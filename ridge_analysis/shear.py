import numpy as np
import h5py
import time
import healpy as hp
from sklearn.neighbors import NearestNeighbors
from .io import ShearMeasurement
from astropy.coordinates import SkyCoord


def get_position_angle(ra_source, dec_source, ra_filament, dec_filament):
    """
    All inputs are expected to be in radians
    """
    # matched_filament_points and bg_coords are in radians right now.
    # So we need to convert them back to degrees for SkyCoord
    bg_sky = SkyCoord(ra=ra_source * u.rad, dec=dec_source * u.rad)
    filament_sky = SkyCoord(ra=ra_filament * u.rad, dec=dec_filament * u.rad)

    # Compute position angle of filament point relative to background galaxy
    phi = bg_sky.position_angle(filament_sky).rad + np.pi/2  # radians, same as before
    return phi


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

    # 110 arcmin, so as we are only going out to 1 degree that should be enough for any
    # adjacent pixels.
    source_healpix_low_res = hp.ang2pix(nside_coverage, np.pi/2 - decs, ras)
    n_unique_pixels = len(np.unique(source_healpix_low_res))
    print(f"Precomputing source pixel regions: {n_unique_pixels} unique pixels at nside {nside_coverage}")
    # 1715 pixels, so a nice reduction but not too small.
    pixel_regions = {}

    # This takes < 1 minute
    for j, i in enumerate(np.unique(source_healpix_low_res)):
        index = np.where(source_healpix_low_res == i)[0]
        pixel_regions[i] = (ras[index], decs[index], g1[index], g2[index], weights[index])

    return pixel_regions



def get_nearby_sources(raf, decf, pixel_regions, nside_coverage):
    """
    Extract from the pixel regions the sources that are near to the filament points.
    """
    filament_pix_low_res = hp.ang2pix(nside_coverage, np.pi/2 - decf, raf)
    filament_pix_low_res = np.unique(filament_pix_low_res)
    pixels_needed = hp.get_all_neighbours(nside_coverage, filament_pix_low_res).flatten()

    # Include the filament pixels themselves, and then get unique values
    pixels_needed = np.unique(np.concatenate((pixels_needed, filament_pix_low_res)))
    ra = []
    dec = []
    g1 = []
    g2 = []
    z = []
    weight = []

    for p in pixels_needed:
        if p in pixel_regions:
            ras_s, decs_s, g1_s, g2_s, z_s, w_s  = pixel_regions[p]
            ra.append(ras_s)
            dec.append(decs_s)
            g1.append(g1_s)
            g2.append(g2_s)
            z.append(z_s)
            weight.append(w_s)

    if len(ra) == 0:
        return None, None, None, None, None, None, None

    ra = np.concatenate(ra)
    dec = np.concatenate(dec)
    g1 = np.concatenate(g1)
    g2 = np.concatenate(g2)
    z = np.concatenate(z)
    weight = np.concatenate(weight)
    source_coords = np.array([dec, ra]).T

    return source_coords, ra, dec, g1, g2, z, weight






def measure_shear(ridge_catalog,
                  source_catalog,
                  output_shear_file,
                  k=1,
                  num_bins=20,
                  comm=None,
                  flip_g1=False, flip_g2=False,
                  nside_coverage=32,
                  min_distance_arcmin=1.0,
                  max_distance_arcmin=60.0,
                  skip_end_points=False, 
                  min_filament_points=0
    ):

    min_ang_rad = np.radians(min_distance_arcmin/60)
    max_ang_rad = np.radians(max_distance_arcmin/60)
    
    # this line was defined lower in the code
    #bins = np.logspace(np.log10(min_ang_rad), np.log10(max_ang_rad), num_bins + 1)
    
    coverage_pixel_size = hp.nside2resol(nside_coverage, arcmin=True)
    assert coverage_pixel_size > max_distance_arcmin, "Coverage pixel size ({coverage_pixel_size} arcmin) must be larger than max_distance_arcmin ({max_distance_arcmin} arcmin). Increase nside_coverage."

    start_time = time.time()

    ra_filaments = np.radians(ridge_catalog.ra)
    dec_filaments = np.radians(ridge_catalog.dec)
    labels = ridge_catalog.ridge_id

    ra_background = np.radians(source_catalog.ra)
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
    #  Note that the haversine distance metric requires data in the form of [latitude, longitude] 
    # and both inputs and outputs are in units of radians.
    # check units are in degrees
    assert ra_filaments.max() < 3 * np.pi, "Filament RA values should be in radians {}".format(ra_filaments.max())
    assert ra_background.max() < 3 * np.pi, "Background RA values should be in radians {}".format(ra_background.max())


    # Pre-split the catalog into a low-resolution map, so that we can just look up pixels
    # that are relatively close to the filament points later.
    pixel_regions = precompute_pixel_regions(
        ra_background,
        dec_background,
        g1_background,
        g2_background,
        weights_background,
        nside_coverage,
    )
    

    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != -1]  # Exclude noise label (-1)

    for filament_index, label in enumerate(unique_labels):
        filament_mask = np.where(labels == label)[0]
        filament_coords = np.array([dec_filaments[filament_mask], ra_filaments[filament_mask]]).T

        if filament_mask.size < min_filament_points:
            print(f"[{rank}] Skipping filament {filament_index} (label {label}) with only {filament_mask.size} points")
            continue

        # Pull out sources within adjacent low-resolution healpix pixels
        source_coords, ra_subset, dec_subset, g1_subset, g2_subset, weights_subset = get_nearby_sources(
            ra_filaments[filament_mask], dec_filaments[filament_mask], pixel_regions, nside_coverage
        )

        # There may be no sources near this filament segment
        if source_coords is None:
            continue


        if (rank == 0) and (filament_index % 10 == 0):
            print(f"[{rank}] Processing filament {filament_index} / {len(unique_labels)} - {source_coords.shape[0]} nearby sources")


        # For each background galaxy, find nearest filament point
        nbrs = NearestNeighbors(n_neighbors=1, leaf_size=100, metric="haversine").fit(filament_coords)
        distances, indices = nbrs.kneighbors(source_coords)
        matched_filament_points = filament_coords[indices[:, 0]]


        # Get the rotation angle phi between the background galaxy and the filament point
        phi = get_position_angle(
            ra_source=ra_subset,
            dec_source=dec_subset,
            ra_filament=matched_filament_points[:, 1],
            dec_filament=matched_filament_points[:, 0],
        )
        
        # Rotate the shear into the filament frame
        g_plus = -g1_subset * np.cos(2 * phi) + g2_subset * np.sin(2 * phi)
        g_cross = g1_subset * np.sin(2 * phi) - g2_subset * np.cos(2 * phi)

        # Bin the distances between 1 arcmin and 1 degree
        bins = np.logspace(np.log10(min_ang_rad), np.log10(max_ang_rad), num_bins + 1) # This is now moved to the top
        bin_indices = np.digitize(distances[:, 0], bins) - 1
        valid_bins = (bin_indices >= 0) & (bin_indices < num_bins)

        if skip_end_points:
            # optionally skip pairs where the filament point is at the end of the filament
            mask = (indices[:, 0] > 0) & (indices[:, 0] < len(filament_coords) - 1)
            valid_bins = valid_bins & mask

        # Accumulate the total tangential and cross shear in each bin,
        # together with the counts, weights, and actual (as opposed to nominal) distances.
        np.add.at(bin_sums_plus, bin_indices[valid_bins], weights_subset[valid_bins] * g_plus[valid_bins])
        np.add.at(bin_sums_cross, bin_indices[valid_bins], weights_subset[valid_bins] * g_cross[valid_bins])
        np.add.at(bin_weighted_distances, bin_indices[valid_bins], weights_subset[valid_bins] * distances[valid_bins, 0])
        np.add.at(bin_weights, bin_indices[valid_bins], weights_subset[valid_bins])
        np.add.at(bin_counts, bin_indices[valid_bins], 1)
    
    # Sum up the results from all processes, so the totals are correct
    sum_in_place(bin_sums_plus, comm)
    sum_in_place(bin_sums_cross, comm)
    sum_in_place(bin_weighted_distances, comm)
    sum_in_place(bin_weights, comm)
    sum_in_place(bin_counts, comm)

    # Only the root process computes the final division and writes out the final results
    if comm is not None and comm.rank != 0:
        return
        
    # Avoid NaNs by only dividing where bin_weights > 0
    weighted_g_plus = np.divide(bin_sums_plus, bin_weights, out=np.zeros_like(bin_sums_plus), where=bin_weights > 0)
    weighted_g_cross = np.divide(bin_sums_cross, bin_weights, out=np.zeros_like(bin_sums_cross), where=bin_weights > 0)
    weighted_real_distances = np.divide(bin_weighted_distances, bin_weights, out=np.zeros_like(bin_weighted_distances), where=bin_weights > 0)

    # Get the nominal bin centers
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    output_table = ShearMeasurement(output_shear_file)
    for i in range(num_bins):
        output_table.data.add_row((bin_centers[i], weighted_real_distances[i], weighted_g_plus[i], weighted_g_cross[i], bin_counts[i], bin_weights[i]))
    
    return output_table


    print(f"Shear processing completed in {time.time() - start_time:.2f} seconds.")

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