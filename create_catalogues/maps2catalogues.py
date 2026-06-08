import h5py
import healpix as hx
import healpy as hp
import numpy as np
import os 
import gc
import time



SIM_NSIDE = 1024
MASK_NSIDE = 4096


def create_resizable_dataset(group, name, dtype):
    group.create_dataset(
        name,
        shape=(0,),
        maxshape=(None,),
        dtype=dtype,
        chunks=True,
        compression='gzip',
        shuffle=True
    )

def initialize_catalogue(filename):
    # Delete existing file if it exists
    if os.path.exists(filename):
        os.remove(filename)
    with h5py.File(filename, 'w') as f:
        for field in ['ra', 'dec']:
            create_resizable_dataset(f, field, np.float64)
    return filename

def initialize_shear_catalogue(filename, include_kappa=False):
    # Delete existing file if it exists
    if os.path.exists(filename):
        os.remove(filename)
    with h5py.File(filename, 'w') as f:
        for field in ['ra', 'dec', 'g1', 'g2']:
            create_resizable_dataset(f, field, np.float64)
        create_resizable_dataset(f, 'weight', 'i2')
        if include_kappa:
            create_resizable_dataset(f, 'kappa', np.float64)
    return filename

def append_to_hdf5(f, ra, dec):
    n_new = len(ra)
    old_size = len(f['ra'])
    new_size = old_size + n_new

    for field, values in (('ra', ra), ('dec', dec)):
        f[field].resize((new_size,))
        f[field][old_size:new_size] = values


def append_shear_to_hdf5(f, ra, dec, gamma1, gamma2, weight, kappa=None):
    n_new = len(ra)
    old_size = len(f['ra'])
    new_size = old_size + n_new

    fields_and_values = {
        'ra': ra,
        'dec': dec,
        'g1': gamma1,
        'g2': gamma2,
        'weight': weight,
    }

    for field, values in fields_and_values.items():
        f[field].resize((new_size,))
        f[field][old_size:new_size] = values

    if kappa is not None:
        f['kappa'].resize((new_size,))
        f['kappa'][old_size:new_size] = kappa


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


def write_dg_catalogue_from_map(delta_g, number_density, h5filename, mask=None, chunk_size=10_000):
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

    f = h5py.File(h5filename, 'a')

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

        append_to_hdf5(f, ra, dec)
    # print total number of galaxy (after mask)
    print(f'Total number of galaxies: {len(f["ra"])/1e6:.2f} million')
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


def write_shear_catalogue_from_dg_catalogue(dg_catalogue, gamma1_map, gamma2_map, h5filename='new_gamma_catalogue.h5', chunk_size=200_000, kappa=None):
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

    with h5py.File(dg_catalogue, 'r') as fin, \
         h5py.File(h5filename, 'a') as fout:

        N = len(fin['ra'])

        for start in range(0, N, chunk_size):

            end = min(start + chunk_size, N)

            # sequential reads
            ra_chunk = fin['ra'][start:end]
            dec_chunk = fin['dec'][start:end]
            
            theta = np.radians(90 - dec_chunk)
            phi = np.radians(ra_chunk)

            # get corresponding shear (interpolate on neighbours)
            gamma1_chunk = hp.get_interp_val(gamma1_map, theta, phi)
            gamma2_chunk = hp.get_interp_val(gamma2_map, theta, phi)

            n = len(ra_chunk)

            # redshift tag for this chunk
            w = np.full(n, 1.)
            
            # interpolate kappa if provided
            kappa_chunk = None
            pix = hp.ang2pix(hp.get_nside(gamma1_map), theta, phi)
            if kappa is not None:
                # get kappa values at galaxy positions (no interpolation needed, take directly from kappa)
                kappa_chunk = kappa[pix]
            
            # write on hdf5 using global index
            append_shear_to_hdf5(fout, ra_chunk, dec_chunk, gamma1_chunk, gamma2_chunk, w, kappa_chunk)

        fout.close()
        fin.close()


def maps2catalogues(cosmogrid_filename, filenames, n_g_maglim, n_g_metacal, mask, bin_i, include_kappa=False, path_to_files=''):
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
    include_kappa : bool, optional
        Whether to include kappa values in the shear catalogue (default is False).
    path_to_files : str, optional
        Base path to the input maps and output catalogues.
    '''
    f = h5py.File(cosmogrid_filename, 'r')

    dg_cat_maglim_hd5  = initialize_catalogue(f"{path_to_files}{filenames['lens']}")
    dg_cat_metacal_hd5 = initialize_catalogue(f"{path_to_files}dg_metacal_catalogue.hdf5",)
    gamma_cat_h5 = initialize_shear_catalogue(f"{path_to_files}{filenames['source']}", include_kappa)

    # Load overdensity maps from CosmoGrid
    dg_maps = {
        'maglim': np.array(f['map']['dg'][f"maglim{bin_i + 1}"]),
        'metacal': np.array(f['map']['dg'][f"metacal{bin_i + 1}"]),
    }

    # Regrade maps to desired resolution (if needed)
    for key in dg_maps:
        dg_maps[key] = hp.ud_grade(dg_maps[key], nside_out=SIM_NSIDE, order_in='RING', order_out='RING')

    # Make lens and source density catalogue (and write it on hdf5 file)
    print('Making galaxy catalogues...')
    catalogue_inputs = [
        (dg_maps['maglim'], n_g_maglim[bin_i], dg_cat_maglim_hd5),
        (dg_maps['metacal'], n_g_metacal[bin_i], dg_cat_metacal_hd5),
    ]
    for delta_g, number_density, filename in catalogue_inputs:
        write_dg_catalogue_from_map(delta_g=delta_g,
                                    number_density=number_density,
                                    mask=mask,
                                    h5filename=filename)

    # Delete maps (free up memory)
    del dg_maps
    gc.collect()
    print("Making shear catalogue...")

    # Load convergence maps from CosmoGrid
    m = f['map']['kg'][f"metacal{bin_i + 1}"][:]
    kappa_metacal_map = m - np.mean(m)
    kappa_metacal_map = hp.ud_grade(kappa_metacal_map, nside_out=SIM_NSIDE, order_in='RING', order_out='RING')

    kappa_lm = hp.sphtfunc.map2alm(kappa_metacal_map)
    lmax = hp.Alm.getlmax(len(kappa_lm))

    # Convert convergence to shear (in Harmonic space)
    print('Getting shear from convergence...')
    gamma_lm = kappa2gamma_lm_map(kappa_lm=kappa_lm, lmax=lmax)


    # Convert shear lm to shear map
    zeros = np.zeros_like(gamma_lm)
    g1, g2 = hp.alm2map_spin([gamma_lm, zeros], SIM_NSIDE, lmax=lmax, spin=2) 

    # Write shear catalogue in an hdf5 file
    print('Making shear catalogue...')
    write_shear_catalogue_from_dg_catalogue(dg_catalogue=dg_cat_metacal_hd5, 
                                            gamma1_map=g1, 
                                            gamma2_map=g2,
                                            h5filename=gamma_cat_h5, 
                                            kappa=kappa_metacal_map)
    # Delete convergence maps
    del kappa_metacal_map, kappa_lm
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


def main(cosmogrid_filename, gold_mask_filename, output_dir):

    n_g_metacal = np.array([1.476, 1.479, 1.484, 1.461])
    n_g_maglim = np.array([0.150, 0.107, 0.109, 0.146])
    nbins = 1

    start_time = time.time()

    mask, mask_low_res = load_mask(gold_mask_filename)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(nbins):
        print(f'Processing bin {i + 1}/{nbins}')
        filenames = {'lens': f'lens_catalog_{SIM_NSIDE}_{i}.hdf5',
                    'source': f'source_catalog_{SIM_NSIDE}_{i}.hdf5'}
        
        maps2catalogues(
            cosmogrid_filename=cosmogrid_filename,
            filenames=filenames,
            n_g_metacal=n_g_metacal,
            n_g_maglim=n_g_maglim,
            mask=(mask, mask_low_res),
            bin_i=i,
            include_kappa=True,
            path_to_files=output_dir
        )    
        print(f'Done with bin {i+1}')

    print('-------------------------------')
    end_time = time.time()
    print(f'Total time taken: {(end_time - start_time)/60:.2f} minutes')


if __name__ == "__main__":
    cosmogrid_filename = './data/projected_probes_maps_v11dmb.h5'
    gold_mask_filename = './data/desy3_gold_mask.npy'
    output_dir = './data/catalogues/'
    main(cosmogrid_filename, gold_mask_filename, output_dir)