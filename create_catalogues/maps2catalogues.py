import h5py
import healpix as hx
import healpy as hp
import numpy as np
import os 
import gc
import time


current_dir = os.path.dirname(os.path.abspath(__file__))
f = h5py.File(current_dir+'/data/projected_probes_maps_v11dmb.h5')


def initialize_catalogue(filename):
    # Delete existing file if it exists
    if os.path.exists(filename):
        os.remove(filename)
    with h5py.File(filename, 'w') as f:
        f.create_dataset(
            'ra',
            shape=(0,),
            maxshape=(None,),
            dtype=np.float64,
            chunks=True,
            compression='gzip',
            shuffle=True
        )
        f.create_dataset(
            'dec',
            shape=(0,),
            maxshape=(None,),
            dtype=np.float64,
            chunks=True,
            compression='gzip',
            shuffle=True
        )
    return filename

def initialize_shear_catalogue(filename, include_kappa=False):
    # Delete existing file if it exists
    if os.path.exists(filename):
        os.remove(filename)
    with h5py.File(filename, 'w') as f:
        f.create_dataset(
            'ra',
            shape=(0,),
            maxshape=(None,),
            dtype=np.float64,
            chunks=True,
            compression='gzip',
            shuffle=True
        )
        f.create_dataset(
            'dec',
            shape=(0,),
            maxshape=(None,),
            dtype=np.float64,
            chunks=True,
            compression='gzip',
            shuffle=True
        )
        f.create_dataset(
            'g1',
            shape=(0,),
            maxshape=(None,),
            dtype=np.float64,
            chunks=True,
            compression='gzip',
            shuffle=True
        )
        f.create_dataset(
            'g2',
            shape=(0,),
            maxshape=(None,),
            dtype=np.float64,
            chunks=True,
            compression='gzip',
            shuffle=True
        )
        f.create_dataset(
            'weight',
            shape=(0,),
            maxshape=(None,),
            dtype='i2',
            chunks=True,
            compression='gzip',
            shuffle=True
        )
        if include_kappa:
            f.create_dataset(
                'kappa',
                shape=(0,),
                maxshape=(None,),
                dtype=np.float64,
                chunks=True,
                compression='gzip',
                shuffle=True
            )
    return filename

def append_to_hdf5(f, ra, dec):
    n_new = len(ra)
    old_size = len(f['ra'])
    new_size = old_size + n_new

    # resize
    f['ra'].resize((new_size,))
    f['dec'].resize((new_size,))

    # append
    f['ra'][old_size:new_size] = ra
    f['dec'][old_size:new_size] = dec


def append_shear_to_hdf5(f, ra, dec, gamma1, gamma2, weight, kappa=None):
    n_new = len(ra)
    old_size = len(f['ra'])
    new_size = old_size + n_new

    # resize datasets
    f['ra'].resize((new_size,))
    f['dec'].resize((new_size,))
    f['g1'].resize((new_size,))
    f['g2'].resize((new_size,))
    f['weight'].resize((new_size,))

    # append
    f['ra'][old_size:new_size] = ra
    f['dec'][old_size:new_size] = dec
    f['g1'][old_size:new_size] = gamma1
    f['g2'][old_size:new_size] = gamma2
    f['weight'][old_size:new_size] = weight

    if kappa is not None:
        f['kappa'].resize((new_size,))
        f['kappa'][old_size:new_size] = kappa


def extract_poisson_position_from_map(map, n_density):
    '''
    Extracts galaxy positions from given map using Poisson sampling.

    Parameters
    ----------
    map : array
        HEALPix map of galaxy overdensity (delta_g)
    n_density : float
        Mean galaxy number density (in galaxies per arcmin^2)

    Returns
    -------
    galaxy_position : array
        Array of galaxy counts for each pixel, sampled from a Poisson distribution
    '''
    nside = hp.get_nside(map)
    # pixel area in arcmin2
    pixel_area = hp.nside2pixarea(nside, degrees=True) * 3600
    # mean number of galaxies per pixel area
    nbar = n_density * (1. + map) * pixel_area
    nbar[nbar<0] = 0

    return np.random.poisson(nbar)


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
    galaxy_position = extract_poisson_position_from_map(delta_g, number_density)

    f = h5py.File(h5filename, 'a')

    for start in range(0, len(galaxy_position), chunk_size):
        end = min(start + chunk_size, len(galaxy_position))
        counts_chunk = galaxy_position[start:end]

        pix_chunk = np.arange(start, end)
        
        if mask is not None:
            mask_chunk = mask[pix_chunk]
            pix_chunk = pix_chunk[mask_chunk]
            counts_chunk = counts_chunk[mask_chunk]

        pixels = np.repeat(pix_chunk, counts_chunk)

        if len(pixels) == 0:
            continue

        theta, phi = hx.randang(nside, ipix=pixels)
        ra = np.degrees(phi)
        dec = 90. - np.degrees(theta)

        append_to_hdf5(f, ra, dec)
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


def maps2catalogues(filenames, n_g_maglim, n_g_metacal, mask, bin_i, include_kappa=False, path_to_files=''):
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

    dg_cat_maglim_hd5  = initialize_catalogue(path_to_files + filenames['lens'])
    dg_cat_metacal_hd5 = initialize_catalogue(path_to_files + '/data/catalogues/dg_metacal_catalogue.hdf5',)
    gamma_cat_h5 = initialize_shear_catalogue(path_to_files + filenames['source'], include_kappa)

    # Load overdensity maps from CosmoGrid
    dg_maglim_map = np.array(f['map']['dg']['maglim'+str(bin_i+1)])
    dg_metacal_map = np.array(f['map']['dg']['metacal'+str(bin_i+1)])

    # Make lens and source density catalogue (and write it on hdf5 file)
    print('Making galaxy catalogues...')
    write_dg_catalogue_from_map(delta_g=dg_maglim_map,
                                number_density=n_g_maglim[bin_i],
                                mask=mask,
                                h5filename=dg_cat_maglim_hd5)
    write_dg_catalogue_from_map(delta_g=dg_metacal_map,
                                number_density=n_g_metacal[bin_i],
                                mask=mask,
                                h5filename=dg_cat_metacal_hd5)

    # Delete maps (free up memeory)
    del dg_maglim_map, dg_metacal_map
    gc.collect()

    # Load convergence maps from CosmoGrid
    m = f['map']['kg']['metacal'+str(bin_i+1)][:]
    kappa_metacal_map = m - np.mean(m)


    kappa_lm = hp.sphtfunc.map2alm(kappa_metacal_map)
    lmax = hp.Alm.getlmax(len(kappa_lm))

    # Convert convergence to shear (in Harmonic space)
    print('Getting shear from convergence...')
    gamma_lm = kappa2gamma_lm_map(kappa_lm=kappa_lm, lmax=lmax)


    # Convert shear lm to shear map
    zeros = np.zeros_like(gamma_lm)
    g1, g2 = hp.alm2map_spin([gamma_lm, zeros], 1024, lmax=lmax, spin=2) 

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


def main():
    n_g_metacal = np.array([1.476, 1.479, 1.484, 1.461])
    n_g_maglim = np.array([0.150, 0.107, 0.109, 0.146])
    nbins = 4

    start_time = time.time()

    gold_mask = np.load(current_dir+'/data/desy3_gold_mask.npy')
    mask = np.zeros(hp.nside2npix(4096))
    mask[gold_mask] = 1.
    mask = hp.ud_grade(mask, nside_out=1024, order_in='RING', order_out='RING')
    mask = hp.reorder(mask, n2r=True)
    mask = mask > 0.8 # change threshold to be more conservative (keep only pixels with >80% coverage)

    path = current_dir+'/data/catalogues/'
    if not os.path.exists(path):
        os.makedirs(path)

    for i in range(nbins):
        print('Processing bin '+str(i+1)+'/'+str(nbins))
        filenames = {'lens': 'lens_catalog_'+str(i)+'.hdf5',
                    'source': 'source_catalog_'+str(i)+'.hdf5'}
        
        maps2catalogues(
            filenames=filenames,
            n_g_metacal=n_g_metacal,
            n_g_maglim=n_g_maglim,
            mask=mask,
            bin_i=i,
            include_kappa=True,
            path_to_files=path
        )    
        print(f'Done with bin {i+1}')

    print('-------------------------------')
    end_time = time.time()
    print(f'Total time taken: {(end_time - start_time)/60:.2f} minutes')


if __name__ == "__main__":
    main()