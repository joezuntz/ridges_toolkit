import h5py
import healpix as hx
import healpy as hp
import numpy as np
import os 
import gc
import time
import glob
import traceback

SIM_NSIDE = 1024
MASK_NSIDE = 4096


def create_resizable_dataset(group, name, dtype, max_size):
    group.create_dataset(
        name,
        shape=(max_size,),
        maxshape=(max_size,),
        dtype=dtype,
        chunks=True,
        # compression='gzip',
        shuffle=True
    )

def initialize_catalogue(filename, max_size):
    # Delete existing file if it exists
    if os.path.exists(filename):
        os.remove(filename)
    with h5py.File(filename, 'w') as f:
        for field in ['ra', 'dec']:
            create_resizable_dataset(f, field, np.float64, max_size)
    return filename

def initialize_shear_catalogue(filename, max_size):
    # Delete existing file if it exists
    if os.path.exists(filename):
        os.remove(filename)
    with h5py.File(filename, 'w') as f:
        for field in ['ra', 'dec', 'g1', 'g2']:
            create_resizable_dataset(f, field, np.float64, max_size)
        create_resizable_dataset(f, 'weight', 'i2', max_size)
    return filename

def append_to_hdf5(f, ra, dec, n_old):
    n_new = len(ra)
    max_size = f['ra'].size
    if n_new + n_old > max_size:
        raise ValueError(f"Joe must have made a mistake estimating the total size: {n_new}, {n_old}, {max_size}")
    f['ra'][n_old:n_old + n_new] = ra
    f['dec'][n_old:n_old + n_new] = dec


def append_shear_to_hdf5(f, gamma1, gamma2, weight, n_old):
    n_new = len(gamma1)

    f["g1"][n_old:n_old+n_new] = gamma1
    f["g2"][n_old:n_old+n_new] = gamma2
    f["weight"][n_old:n_old+n_new] = weight


def extract_poisson_counts_from_map(map, n_density):
    '''
    Extract galaxy counts from a density map by Poisson sampling.

    Parameters
    ----------
    map : array
        HEALPix map of galaxy overdensity (delta_g)
    n_density : float
        Mean galaxy number density (in galaxies per arcmin^2)

    Returns
    -------
    gal_counts : array
        Array of galaxy counts for each pixel, sampled from a Poisson distribution
    '''
    nside = hp.get_nside(map)
    # pixel area in arcmin2
    pixel_area = hp.nside2pixarea(nside, degrees=True) * 3600
    # mean number of galaxies per pixel area
    nbar = n_density * (1. + map) * pixel_area
    nbar[nbar<0] = 0

    gal_counts = np.random.poisson(nbar)
    return gal_counts


def write_dg_catalogue_from_map(delta_g, number_density, h5filename, mask=None, chunk_size=10_000, will_be_gamma=False):
    '''
    Write galaxy catalogue from density maps using Poisson sampling.

    Parameters
    ----------
    dg_maps : list of array
        List of HEALPix overdensity maps to be converted into catalogues
    number_density : array or list of float
        Galaxy number density for each redshift bin (same length as maps)
    h5filename : str
        Name of the HDF5 file to write the catalogue to
    chunk_size : int, optional
        Number of pixels to process in each chunk (default is 10,000)
    '''

    # recenter map
    delta_g = delta_g - np.mean(delta_g)

    nside = hp.get_nside(delta_g)
    # get map of galaxy counts with Poisson distribution
    galaxy_counts = extract_poisson_counts_from_map(delta_g, number_density)
    total_count = galaxy_counts.sum()

    if will_be_gamma:
        initialize_shear_catalogue(h5filename, total_count)
    else:
        initialize_catalogue(h5filename, total_count)

    f = h5py.File(h5filename, 'a')
    current_size = 0

    for start in range(0, len(galaxy_counts), chunk_size):
        end = min(start + chunk_size, len(galaxy_counts))
        galaxy_counts_chunk = galaxy_counts[start:end]
        pix_chunk = np.arange(start, end)
        
        # avoid wasting time on pixels that are 100%
        # masked out. We do that by making a low-res mask
        # that is True if any of the high-res pixels are unmasked,
        # and then applying it to the chunk of pixels before sampling
        # the galaxy positions.
        if mask is not None:
            _, mask_low_res = mask
            # consider only pixels in the chunk
            mask_chunk = mask_low_res[pix_chunk]
            # mask pixels (needs to be of the same len of galaxy_counts_chunk once masked)
            pix_chunk = pix_chunk[mask_chunk]
            # mask galaxy counts
            galaxy_counts_chunk = galaxy_counts_chunk[mask_chunk]

        pixels = np.repeat(pix_chunk, galaxy_counts_chunk)

        if len(pixels) == 0:
            continue

        theta, phi = hx.randang(nside, ipix=pixels)

        # cut to objects within high-res mask
        # hopefully not too much slower.
        # we could make things faster if needed
        if mask is not None:
            mask_high_res, _ = mask
            pix_at_mask_res = hp.ang2pix(MASK_NSIDE, theta, phi)
            mask_values = mask_high_res[pix_at_mask_res]
            theta = theta[mask_values]
            phi = phi[mask_values]


        ra = np.degrees(phi)
        dec = 90. - np.degrees(theta)

        append_to_hdf5(f, ra, dec, current_size)
        current_size += ra.size

    # because we masked out some pixels the final size
    # will not be the same as the initial one, so we cut
    # down the size now
    for dataset in f.values():
        dataset.resize((current_size,))
    # print total number of galaxy (after mask)
    print(f'Total number of galaxies: {current_size/1e6:.2f} million')
    f.close()


def kappa2gamma_lm_map(kappa_lm, lmax):
    '''
    Convert convergence alm to shear alm using the relation:
    gamma_lm = kappa_lm * sqrt( (ell+2)(ell-1) / (ell(ell+1)) ) for ell >= 1, and gamma_lm = 0 for ell=0 (monopole) 
    
    Parameters
    ----------
    kappa_lm : array
        Spherical harmonic coefficients of the convergence map
    lmax : int
        Maximum multipole order of the spherical harmonic coefficients  

    Returns
    -------
    gamma_lm : array
        Spherical harmonic coefficients of the shear map
    '''
    ell = np.linspace(0, lmax, len(kappa_lm))

    factor = np.zeros_like(ell)

    # skip monopole (factor should be inf for ell=0)
    valid = ell >= 1
    factor[valid] = -np.sqrt( (ell[valid]+2) * (ell[valid]-1)\
                             / (ell[valid] * (ell[valid]+1)) )

    gamma_lm = kappa_lm * factor
    return gamma_lm


def write_shear_catalogue_from_dg_catalogue(gamma1_map, gamma2_map, h5filename, chunk_size=200_000, kappa=None):
    '''
    Build a shear-annotated catalogue from a density-galaxy catalogue.

    Parameters
    ----------
    dg_catalogue : str
        Path to the input density-galaxy catalogue in HDF5 format. The catalogue should contain columns 'ra', 'dec' and 'z'.
    gamma1_map, gamma2_map : array
        HEALPix maps of the shear components (gamma1 and gamma2) at the same nside as the density maps.
    h5filename : str, optional
        Output HDF5 file name.
    chunk_size : int, optional
        Number of objects processed per chunk.

    Returns
    -------
    str
        The output h5filename.
    '''

    with h5py.File(h5filename, 'r+') as f:
        N = len(f['ra'])
        current_size = 0

        for start in range(0, N, chunk_size):

            end = min(start + chunk_size, N)

            # sequential reads
            ra_chunk = f['ra'][start:end]
            dec_chunk = f['dec'][start:end]
            
            theta = np.radians(90 - dec_chunk)
            phi = np.radians(ra_chunk)

            # get corresponding shear (interpolate on neighbours)
            gamma1_chunk = hp.get_interp_val(gamma1_map, theta, phi)
            gamma2_chunk = hp.get_interp_val(gamma2_map, theta, phi)

            n = len(ra_chunk)

            # redshift tag for this chunk
            w = np.full(n, 1.)
            
            # write on hdf5 using global index
            append_shear_to_hdf5(f, gamma1_chunk, gamma2_chunk, w, current_size)
            current_size += gamma1_chunk.size


def maps2catalogues(cosmogrid_filename, filenames, n_g_maglim, n_g_metacal, mask, bin_i, path_to_files=''):
    '''
    Convert density and convergence maps into galaxy catalogues using Poisson sampling.
    Parameters
    ----------
    filenames : dict
        Dictionary containing the paths to the input maps and output catalogues. 
        Should have keys 'lens' and 'source'.
    n_g_maglim, n_g_metacal : array-like
        Arrays containing the galaxy number density for each redshift bin for the metacal and maglim samples, respectively.
    mask : array
        HEALPix mask to apply to the maps (same nside as the maps).
    bin_i : int
        Index of the redshift bin to process (0-based).
    path_to_files : str, optional
        Base path to the input maps and output catalogues.
    '''
    f = h5py.File(cosmogrid_filename, 'r')

    # Load overdensity maps from CosmoGrid
    maglim_dg_map = f['map']['dg'][f"maglim{bin_i + 1}"][:]

    # Regrade maps to desired resolution (if needed)
    maglim_dg_map = hp.ud_grade(maglim_dg_map, nside_out=SIM_NSIDE, order_in='RING', order_out='RING')

    dg_cat_maglim_hd5 = f"{path_to_files}{filenames['lens']}"

    # Make the lens density catalogue.
    # We don't need a separate density catalogue for the source sample
    # because the gamma map below already contains that information
    # and it would just be a duplicate.
    print('Making maglim catalogue...')
    write_dg_catalogue_from_map(delta_g=maglim_dg_map,
                                number_density=n_g_maglim[bin_i],
                                mask=mask,
                                h5filename=dg_cat_maglim_hd5
                                )

    # Delete maps (free up memory)
    del maglim_dg_map
    gc.collect()
    print("Making shear catalogue...")

    metacal_dg_map = f['map']['dg'][f"metacal{bin_i + 1}"][:]
    metacal_dg_map = hp.ud_grade(metacal_dg_map, nside_out=SIM_NSIDE, order_in='RING', order_out='RING')


    gamma_cat_h5 = f"{path_to_files}{filenames['source']}"

    # first write the RA/Dec parts of the shear catalog
    write_dg_catalogue_from_map(delta_g=metacal_dg_map,
                                number_density=n_g_metacal[bin_i],
                                mask=mask,
                                h5filename=gamma_cat_h5,
                                will_be_gamma=True)
    
    del metacal_dg_map
    gc.collect()

    # Load convergence maps from CosmoGrid
    m = f['map']['kg'][f"metacal{bin_i + 1}"][:]
    kappa_metacal_map = m - np.mean(m)
    kappa_metacal_map = hp.ud_grade(kappa_metacal_map, nside_out=SIM_NSIDE, order_in='RING', order_out='RING')

    kappa_lm = hp.sphtfunc.map2alm(kappa_metacal_map)
    lmax = hp.Alm.getlmax(len(kappa_lm))

    # Convert convergence to shear (in Harmonic space)
    print('Getting shear from convergence...')
    gamma_lm = kappa2gamma_lm_map(kappa_lm=kappa_lm, lmax=lmax)

    # Delete convergence maps
    del kappa_metacal_map, kappa_lm

    # Convert shear lm to shear map
    zeros = np.zeros_like(gamma_lm)
    g1, g2 = hp.alm2map_spin([gamma_lm, zeros], SIM_NSIDE, lmax=lmax, spin=2) 

    # Write shear catalogue in an hdf5 file
    print('Making shear catalogue...')
    write_shear_catalogue_from_dg_catalogue(gamma1_map=g1, 
                                            gamma2_map=g2,
                                            h5filename=gamma_cat_h5, 
                                            )
    gc.collect()

    # Free up memory
    del g1, g2, gamma_lm
    gc.collect()
    f.close()

def load_mask(gold_mask_filename):
    gold_mask = np.load(gold_mask_filename)
    mask = np.zeros(hp.nside2npix(MASK_NSIDE), dtype=bool)
    mask[gold_mask] = True
    # It's a little faster up/downgrade a NEST map than a RING one
    # because the NEST pixels are inherently arranged hierarchically.
    mask_low_res = hp.ud_grade(mask.astype(np.float32), nside_out=SIM_NSIDE, order_in='NEST', order_out='NEST')
    mask = hp.reorder(mask, n2r=True)
    mask_low_res = hp.reorder(mask_low_res, n2r=True)
    # take any value for low-res mask as we do a secondary cut later 
    # on the full mask
    mask_low_res = mask_low_res > 0
    return mask, mask_low_res


def main(cosmogrid_filename, gold_mask_filename, output_dir, prefix=""):

    n_g_metacal = np.array([1.476, 1.479, 1.484, 1.461])
    n_g_maglim = np.array([0.150, 0.107, 0.109, 0.146])
    nbins = 1

    start_time = time.time()

    mask, mask_low_res = load_mask(gold_mask_filename)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(nbins):
        print(f'Processing bin {i + 1}/{nbins}')
        filenames = {'lens': f'{prefix}lens_catalog_{SIM_NSIDE}_{i}.hdf5',
                    'source': f'{prefix}source_catalog_{SIM_NSIDE}_{i}.hdf5'}
        
        maps2catalogues(
            cosmogrid_filename=cosmogrid_filename,
            filenames=filenames,
            n_g_metacal=n_g_metacal,
            n_g_maglim=n_g_maglim,
            mask=(mask, mask_low_res),
            bin_i=i,
            path_to_files=output_dir
        )    
        print(f'Done with bin {i+1}')

    print('-------------------------------')
    end_time = time.time()
    print(f'Total time taken: {(end_time - start_time)/60:.2f} minutes')

def run_on_full_cosmogrid():
    from mpi4py.MPI import COMM_WORLD as comm
    rank = comm.rank
    size = comm.size

    # Basic input and output directories
    base_dir = "/global/cfs/cdirs/des/cosmogrid/processed/v11desy3/CosmoGrid/bary/grid"
    output_base_dir = "/pscratch/sd/z/zuntz/ridges/v1"
    gold_mask_filename = f"{output_base_dir}/desy3_gold_mask.npy"


    # Each different cosmology has its own directory.
    # They are numbered but some numbers are missing, so
    # we just find all of their names instead of using a range.
    cosmo_dirs = sorted(glob.glob(f"cosmo_*", root_dir=base_dir))

    log_failures = False

    # once the main code is working we will check for occasional failures
    # and save them to this file. We have to do one file per rank
    # otherwise they will all get mixed up. I'll leave this commented
    # out for now.
    if log_failures:
        failure_log_filename = f"{output_base_dir}/v1_fails.{rank}.log"
        failure_log = open(failure_log_filename, "w")

    for permutation in range(20):
        perm_dir = f"perm_{permutation:04d}"

        for i, cosmo_dir in enumerate(cosmo_dirs):
            if i % size != rank:
                continue

            # construct all the file paths we need
            cosmogrid_file = f"{base_dir}/{cosmo_dir}/{perm_dir}/projected_probes_maps_v11dmb.h5"
            output_dir = f"{output_base_dir}/{cosmo_dir}"
            os.makedirs(output_dir, exist_ok=True)

            marker_file = os.path.join(output_dir, f"complete.{permutation}")
            # skip if the completion marker is already done.
            if os.path.exists(marker_file):
                continue
            # we don't want to have too many directories so we collect
            # the different permutations together in the same directory
            prefix = perm_dir + "_"

            # Try running the main function, but if something goes wrong, log it to the file
            try:
                main(cosmogrid_file, gold_mask_filename, output_dir, prefix=prefix)
                # when all has completed, add a marker file to the output dir
                # so we know this permutation is done
                open(marker_file, "w").close()

            except Exception as error:
                # While testing, let errors cause a crash as normal.
                # once things are working we can catch any occasional errors
                if log_failures:
                    failure_log.write("\n\n" + str(error) + "\n")
                    failure_log.write(f"perm={i} cosmo_dir={cosmo_dir}\n")
                    failure_log.write(traceback.format_exc())
                else:
                    raise
    if log_failures:
        failure_log.close()



if __name__ == "__main__":
    cosmogrid_filename = './data/projected_probes_maps_v11dmb.h5'
    gold_mask_filename = './data/desy3_gold_mask.npy'
    output_dir = './data/catalogues/'
    main(cosmogrid_filename, gold_mask_filename, output_dir)