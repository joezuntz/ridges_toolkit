import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
from .io import RidgePointCatalog, LensCatalog, RidgeSegmentCatalog


def make_density_map(lens_catalog, nside, smoothing_degrees):
    # ra and dec are in degrees by default.
    # convert to healpix index
    ra = lens_catalog.ra
    dec = lens_catalog.dec
    pix = hp.ang2pix(nside, ra, dec, lonlat=True)

    # generate count map
    npix = hp.nside2npix(nside)
    m = np.zeros(npix, dtype=int)
    np.add.at(m, pix, 1)

    # smooth and return
    m1 = hp.smoothing(m, fwhm=np.radians(smoothing_degrees), verbose=False)
    return m1


def plot_ridges_on_density(plot_filename, dredge_config, nside, smoothing_degrees):
    """
    Make a plot of a density map and ridge points on top.
    """
    # make the density map from the lens catalog
    lens_cat = LensCatalog(dredge_config.lens_catalog_file)
    lens_cat.load()
    density_map = make_density_map(lens_cat, nside, smoothing_degrees)

    # load the ridge points. Also saved in degrees
    ridge_cat = RidgePointCatalog(dredge_config.ridge_point_file)
    ridge_cat.load()

    # Plot the density map
    hp.cartview(density_map, min=0, lonra=[0, 10], latra=[-5, 5], cbar=True)
    hp.graticule()

    # and the ridge points on top
    hp.projplot(ridge_cat.ra, ridge_cat.dec, "r.", markersize=1, lonlat=True)
    plt.savefig(plot_filename, bbox_inches="tight", dpi=300)


def plot_segments_on_density(plot_filename, dredge_config, segmentation_config, nside, smoothing_degrees):
    """
    Make a plot of a density map and ridge points on top.
    """
    # make the density map from the lens catalog
    lens_cat = LensCatalog(dredge_config.lens_catalog_file)
    lens_cat.load()
    density_map = make_density_map(lens_cat, nside, smoothing_degrees)

    # load the ridge points. Also saved in degrees
    ridge_cat = RidgeSegmentCatalog(segmentation_config.ridge_file)
    ridge_cat.load()

    # Plot the density map
    hp.cartview(density_map, min=0, lonra=[0, 10], latra=[-5, 5], cbar=True)
    hp.graticule()

    # and the ridge points on top
    hp.projscatter(ridge_cat.ra, ridge_cat.dec, s=0.5, c=ridge_cat.ridge_id, lonlat=True, cmap="tab20")
    plt.savefig(plot_filename, bbox_inches="tight", dpi=300)
