import h5py
import healpy as hp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

folder = os.path.dirname(os.path.abspath(__file__))+'/../data/cataloguess/'

def plot_dg_catalogue(catalogue, z_bin, nside=1024, title='', lon=[0,10], lat=[0,10], chunk_size=2_000_000, save_name=''):
    npix = hp.nside2npix(nside)
    counts = np.zeros(npix, dtype=np.int64)

    with h5py.File(catalogue, 'r') as f:
        ra_ds = f['ra']
        dec_ds = f['dec']
        z_ds = f['z']

        N = len(ra_ds)

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)

            ra = ra_ds[start:end]
            dec = dec_ds[start:end]
            z = z_ds[start:end]

            # Select one bin only
            mask_zbin = (z == z_bin)
            if np.any(mask_zbin):
                theta = np.radians(90 - dec[mask_zbin])
                phi = np.radians(ra[mask_zbin])

                pix = hp.ang2pix(nside, theta, phi)
                counts += np.bincount(pix, minlength=npix)

    fig = plt.figure(figsize=(14, 5))
    hp.mollview(counts, fig=fig.number, sub=(1, 2, 1), title='', cmap='viridis')
    ax1 = plt.gca()
    ax1.set_position([0.05, 0.15, 0.60, 0.70])
    hp.cartview(counts, fig=fig.number, sub=(1, 2, 2), title='', cmap='viridis', lonra=lon, latra=lat)
    ax2 = plt.gca()
    ax2.set_position([0.72, 0.22, 0.22, 0.45])

    axes = fig.get_axes()
    for ax in axes:
        # Skip colorbar axes
        if ax.__class__.__name__ != 'HpxMollweideAxes' and \
        ax.__class__.__name__ != 'HpxCartesianAxes':
            continue
        ax.set_xlabel('Right Ascension')
        ax.set_ylabel('Declination')

    for ax in fig.get_axes():
        # Healpy colorbars are regular matplotlib axes
        if ax.__class__.__name__ == 'Axes':
            if len(ax.images) == 0 and len(ax.collections) == 0:
                ax.set_title('Counts', fontsize=10)

    # Better spacing
    plt.tight_layout()
    fig.suptitle(title)
    # plt.show()
    plt.savefig(folder+'../figs/dg_catalogue'+save_name+'.png')


def plot_shear_catalogue(catalogue, nside, z_bin, title='', lon=[0,50], lat=[0,50], chunk_size=2_000_000, save_name=''):
    gamma1_map_rec = np.zeros(hp.nside2npix(nside))
    gamma2_map_rec = np.zeros(hp.nside2npix(nside))
    counts = np.zeros_like(gamma1_map_rec)

    with h5py.File(catalogue, 'r') as f:
        for start in range(0, len(f['ra']), chunk_size):
            end = start + chunk_size

            z = f['z'][start:end]
            # Select one bin only
            mask_zbin = (z == z_bin)

            if not np.any(mask_zbin):
                continue

            ra  = f['ra'][start:end][mask_zbin]
            dec = f['dec'][start:end][mask_zbin]
            gamma1 = f['g1'][start:end][mask_zbin]
            gamma2 = f['g2'][start:end][mask_zbin]

            theta = np.radians(90 - dec)
            phi   = np.radians(ra)

            pix = hp.ang2pix(nside, theta, phi)

            gamma1_map_rec += np.bincount(pix, weights=gamma1,
                                        minlength=len(gamma1_map_rec))
            gamma2_map_rec += np.bincount(pix, weights=gamma2,
                                        minlength=len(gamma2_map_rec))
            counts += np.bincount(pix, minlength=len(counts))

    mask_zbin = counts > 0
    gamma1_map_rec[mask_zbin] /= counts[mask_zbin]
    gamma2_map_rec[mask_zbin] /= counts[mask_zbin]
    
    fig = plt.figure(figsize=(12, 5))
    # set masked pixels to hp.UNSEEN
    gamma1_map_rec[~mask_zbin] = hp.UNSEEN
    gamma2_map_rec[~mask_zbin] = hp.UNSEEN
    hp.cartview(gamma1_map_rec, fig=fig.number, sub=(1, 2, 1), title='', cmap=plt.cm.viridis, lonra=lon, latra=lat)
    hp.cartview(gamma2_map_rec, fig=fig.number, sub=(1, 2, 2), title='', cmap=plt.cm.viridis, lonra=lon, latra=lat)
    fig.suptitle(title)
    # plt.show()
    plt.savefig(folder+'../figs/reconstructed_shear_map'+save_name+'.png')


z_bin = 0
plot_dg_catalogue(folder+'dg_maglim_catalogue.h5', z_bin=z_bin, title=f'Galaxy catalogue (maglim), bin {z_bin+1}', save_name='_maglim', lon=[0,50], lat=[-40,10])

print('plotting metacal galaxies...')
plot_dg_catalogue(folder+'dg_metacal_catalogue.h5', z_bin=z_bin, title=f'Galaxy catalogue (metacal), bin {z_bin+1}', save_name='_metacal', lon=[0,50], lat=[-40,10])

print('plotting reconstructed shear maps...')
plot_shear_catalogue(folder+'gamma_catalogue.h5', nside=1024, z_bin=z_bin, title='Reconstructed shear1 map - bin'+str(z_bin+1), lon=[0,50], lat=[-40,10])