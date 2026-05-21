import h5py
import healpix as hx
import healpy as hp
import numpy as np
import os 
import gc
import time


folder = os.path.dirname(os.path.abspath(__file__))
f = h5py.File(folder+'/data/projected_probes_maps_v11dmb.h5')


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
        f.create_dataset(
            'z',
            shape=(0,),
            maxshape=(None,),
            dtype=np.float64,
            chunks=True,
            compression='gzip',
            shuffle=True
        )
    return filename

def initialize_shear_catalogue(filename):
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
            'z',
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
    return filename

def append_to_hdf5(f, ra, dec, z):
    n_new = len(ra)
    old_size = len(f['ra'])
    new_size = old_size + n_new

    # resize
    f['ra'].resize((new_size,))
    f['dec'].resize((new_size,))
    f['z'].resize((new_size,))

    # append
    f['ra'][old_size:new_size] = ra
    f['dec'][old_size:new_size] = dec
    f['z'][old_size:new_size] = z


def append_shear_to_hdf5(f, ra, dec, z, gamma1, gamma2, weight):
    n_new = len(ra)
    old_size = len(f['ra'])
    new_size = old_size + n_new

    # resize datasets
    f['ra'].resize((new_size,))
    f['dec'].resize((new_size,))
    f['g1'].resize((new_size,))
    f['g2'].resize((new_size,))
    f['z'].resize((new_size,))
    f['weight'].resize((new_size,))

    # append
    f['ra'][old_size:new_size] = ra
    f['dec'][old_size:new_size] = dec
    f['g1'][old_size:new_size] = gamma1
    f['g2'][old_size:new_size] = gamma2
    f['z'][old_size:new_size] = z
    f['weight'][old_size:new_size] = weight


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


def write_dg_catalogue_from_map(delta_g, number_density, zbin, h5filename, mask=None, chunk_size=10_000):
    '''
    Write galaxy catalogue from density maps using Poisson sampling.

    Parameters
    ----------
    dg_maps : list of array
        List of HEALPix overdensity maps to be converted into catalogues
    number_density : array or list of float
        Galaxy number density for each redshift bin (same length as maps)
    zbin : int
        Redshift bin index (starting from 0)
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

        z = np.full(len(ra), zbin)
        append_to_hdf5(f, ra, dec, z)
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


def write_shear_catalogue_from_dg_catalogue(dg_catalogue, gamma1_map, gamma2_map, zbin, h5filename='new_gamma_catalogue.h5', chunk_size=200_000):
    '''
    Build a shear-annotated catalogue from a density-galaxy catalogue.

    Parameters
    ----------
    dg_catalogue : str
        Path to the input density-galaxy catalogue in HDF5 format. The catalogue should contain columns 'ra', 'dec' and 'z'.
    gamma1_map, gamma2_map : array
        HEALPix maps of the shear components (gamma1 and gamma2) at the same nside as the density maps.
    zbin : int
        Redshift bin index (starting from 0)
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

        N = len(fin['z'])

        for start in range(0, N, chunk_size):

            end = min(start + chunk_size, N)

            # sequential reads
            z_chunk = fin['z'][start:end]

            mask_zbin = (z_chunk == zbin)

            if not np.any(mask_zbin):
                continue

            ra_chunk = fin['ra'][start:end][mask_zbin]
            dec_chunk = fin['dec'][start:end][mask_zbin]
            
            theta = np.radians(90 - dec_chunk)
            phi = np.radians(ra_chunk)

            # get corresponding shear (interpolate on neighbours)
            gamma1_chunk = hp.get_interp_val(gamma1_map, theta, phi)
            gamma2_chunk = hp.get_interp_val(gamma2_map, theta, phi)

            n = len(ra_chunk)

            # redshift tag for this chunk
            z = np.full(n, zbin)
            w = np.full(n, 1.)
            # write on hdf5 using global index
            append_shear_to_hdf5(fout, ra_chunk, dec_chunk, z, gamma1_chunk, gamma2_chunk, w)

        fout.close()
        fin.close()


def mask_map(mask_filename, map, nside_mask=4096, nside_map=256, order_in='RING', order_out='RING'):
    '''
    Mask a HEALPix map using a mask defined by a list of hit pixels.
    Parameters
    ----------
    mask_filename : str
        Path to the file containing the list of hit pixels for the mask.
    map : array
        HEALPix map to be masked.
    nside_mask : int, optional
        Nside of the mask (default is 4096).
    nside_map : int, optional
        Nside of the input map (default is 256).
    order_in : str, optional
        Ordering scheme of the input map (default is 'RING').
    order_out : str, optional
        Ordering scheme of the output map (default is 'RING').
    
    Returns
    -------
    masked_map : array
        The input map after applying the mask.
    '''
    hit_pix = np.load(mask_filename)
    mask = np.zeros(hp.nside2npix(nside_mask))
    mask[hit_pix] = 1

    if nside_mask != nside_map:
        mask = hp.ud_grade(mask, nside_out=nside_map, order_in=order_in, order_out=order_out)
    
    if order_out=='RING':
        mask = hp.reorder(mask, n2r=True)
    
    map *= mask
    return map


##############################
############ MAIN ############
##############################

n_g_metacal = np.array([1.476, 1.479, 1.484, 1.461])
n_g_maglim = np.array([0.150, 0.107, 0.109, 0.146])
nbins = 4

start_time = time.time()

dg_cat_maglim_hd5  = initialize_catalogue(folder+'/data/catalogues/dg_maglim_catalogue.hdf5')
dg_cat_metacal_hd5 = initialize_catalogue(folder+'/data/catalogues/dg_metacal_catalogue.hdf5')
gamma_cat_h5 = initialize_shear_catalogue(folder+'/data/catalogues/gamma_catalogue.hdf5')

gold_mask = np.load(folder+'/../data/desy3_gold_mask.npy')
mask = np.zeros(hp.nside2npix(4096))
mask[gold_mask] = 1.
mask = hp.ud_grade(mask, nside_out=1024, order_in='RING', order_out='RING')
mask = hp.reorder(mask, n2r=True)
mask = mask > 0.8 # change threshold to be more conservative (keep only pixels with >80% coverage)


for i in range(nbins):
    print(f'---- Processing bin {i+1} ----')

    # Load overdensity maps from CosmoGrid
    dg_maglim_map = np.array(f['map']['dg']['maglim'+str(i+1)])
    dg_metacal_map = np.array(f['map']['dg']['metacal'+str(i+1)])

    # Make lens and source density catalogue (and write it on hdf5 file)
    print('Making galaxy catalogues...')
    write_dg_catalogue_from_map(delta_g=dg_maglim_map,
                                number_density=n_g_maglim[i],
                                zbin=i,
                                mask=mask,
                                h5filename=dg_cat_maglim_hd5)
    write_dg_catalogue_from_map(delta_g=dg_metacal_map,
                                number_density=n_g_metacal[i],
                                zbin=i,
                                mask=mask,
                                h5filename=dg_cat_metacal_hd5)

    # Delete maps (free up memeory)
    del dg_maglim_map, dg_metacal_map
    gc.collect()

    # Load convergence maps from CosmoGrid
    m = f['map']['kg']['metacal'+str(i+1)][:]
    kappa_metacal_map = m - np.mean(m)

    kappa_lm = hp.sphtfunc.map2alm(kappa_metacal_map)
    lmax = hp.Alm.getlmax(len(kappa_lm))

    # Convert convergence to shear (in Harmonic space)
    print('Getting shear from convergence...')
    gamma_lm = kappa2gamma_lm_map(kappa_lm=kappa_lm, lmax=lmax)

    # Delete convergence maps
    del kappa_metacal_map, kappa_lm
    gc.collect()

    # Convert shear lm to shear map
    zeros = np.zeros_like(gamma_lm)
    g1, g2 = hp.alm2map_spin([gamma_lm, zeros], 256, lmax=lmax, spin=2) 

    # Write shear catalogue in an hdf5 file
    print('Making shear catalogue...')
    write_shear_catalogue_from_dg_catalogue(dg_catalogue=dg_cat_metacal_hd5, 
                                            gamma1_map=g1, 
                                            gamma2_map=g2,
                                            zbin=i, 
                                            h5filename=gamma_cat_h5)

    # Free up memory
    del g1, g2, gamma_lm
    gc.collect()

    print(f'Done with bin {i+1}')


print('-------------------------------')
end_time = time.time()
print(f'Total time taken: {(end_time - start_time)/60:.2f} minutes')