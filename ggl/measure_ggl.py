
import healpy as hp
import numpy as np
import healpix
import treecorr
from astropy.table import Table
import os
    
GOLD_MASK_FILENAME = "/pscratch/sd/z/zuntz/ridges/v1/desy3_gold_mask.npy"
RANDOM_DENSITY = 3 # per square arcmin
CAT_BASE_DIR = "/pscratch/sd/z/zuntz/ridges/v2/catalogs"
FID_CAT_BASE_DIR = "/pscratch/sd/z/zuntz/ridges/fiducial/catalogs"
MIN_SEP = 1.0
MAX_SEP = 60.0
NUM_BINS = 20
NUM_THREADS = int(os.environ['OMP_NUM_THREADS'])
MASK_NSIDE = 4096



def load_mask(gold_mask_filename):
    gold_mask = np.load(gold_mask_filename)
    mask = np.zeros(hp.nside2npix(MASK_NSIDE), dtype=bool)
    mask[gold_mask] = True
    mask = hp.reorder(mask, n2r=True)
    return mask

def generate_randoms(rng, gold_mask_filename = GOLD_MASK_FILENAME):
    mask = load_mask(gold_mask_filename)
    npix_hit = mask.sum()
    pix_hit = np.where(mask)[0]
    npix = len(mask)
    nside = hp.npix2nside(npix)
    pix_area = hp.nside2pixarea(nside, degrees=True) * 60 ** 2 # in arcmin
    mean_num_per_pix = pix_area * RANDOM_DENSITY
    nrandom_per_pix = rng.poisson(lam=mean_num_per_pix, size=npix_hit)
    nrandom = nrandom_per_pix.sum()
    print("Generated random counts")

    ipix = np.repeat(pix_hit, nrandom_per_pix)
    print("Setup random indices")

    ra, dec = healpix.randang(nside, ipix, nest=True, lonlat=True, rng=rng)
    print("Generated random locations")
    cat = treecorr.Catalog(ra=ra, dec=dec, ra_units="degrees", dec_units="degrees")
    print("Generated random catalog of size", cat.ntot)
    return cat


def run_treecorr(cosmo, perm, source_bin, lens_bin, seed, output_filename):
    rng = np.random.default_rng(seed=[abs(hash(cosmo)), perm, source_bin, lens_bin, seed])
    if cosmo == "fiducial":
        cat_dir = FID_CAT_BASE_DIR
    else:
        cat_dir = os.path.join(CAT_BASE_DIR, cosmo)

    random_cat = generate_randoms(rng)


    source_filename = f"perm_{perm:04d}_source_catalog_1024_{source_bin}.hdf5"
    lens_filename = f"perm_{perm:04d}_lens_catalog_1024_{lens_bin}.hdf5"
    source_path = os.path.join(cat_dir, source_filename)
    lens_path = os.path.join(cat_dir, lens_filename)

    source_cat = treecorr.Catalog(file_name=source_path, ra_col='ra', dec_col='dec', g1_col='g1', g2_col='g2', w_col='weight', ra_units='degrees', dec_units='degrees')
    print("Set up source cat of size", source_cat.ntot)

    lens_cat = treecorr.Catalog(file_name=lens_path, ra_col='ra', dec_col='dec', ra_units='degrees', dec_units='degrees')
    print("Set up lens cat or size", lens_cat.ntot)

    config = dict(
        min_sep=MIN_SEP,
        max_sep=MAX_SEP,
        nbins=NUM_BINS,
        sep_units='arcmin',
        num_threads=NUM_THREADS,
        verbose=2,
    )

    ng = treecorr.NGCorrelation(config)
    ng.process(lens_cat, source_cat)
    print("Processed NG")

    rg = treecorr.NGCorrelation(config)
    rg.process(random_cat, source_cat)
    print("Processed RG")

    ng.calculateXi(rg=rg)

    bin_centers = np.exp(ng.meanlogr)
    xi = ng.xi
    err = np.sqrt(ng.varxi)

    nominal_bin_edges = np.geomspace(MIN_SEP, MAX_SEP, NUM_BINS + 1)
    nominal_bin_centers = (nominal_bin_edges[1:] + nominal_bin_edges[:-1]) / 2

    columns = ["sep_bin_center", "weighted_sep", "g_plus",  "err"]
    dtypes = [float, float, float, float]
    output = Table(names=columns, dtype=dtypes)

    for i in range(NUM_BINS):
        output.add_row(
            (
                nominal_bin_centers[i],
                bin_centers[i],
                xi[i],
                err[i]
            )
        )
    
    output.write(output_filename, format="ascii.commented_header", overwrite=True)


if __name__ == "__main__":
    cosmo = "fiducial"
    perm = 0
    source_bin = 1
    lens_bin = 0
    output_filename = "temp.txt"
    seed = 6235
    run_treecorr(cosmo, perm, source_bin, lens_bin, seed, output_filename)
