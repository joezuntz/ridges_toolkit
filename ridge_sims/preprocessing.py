import numpy as np
import h5py
from astropy.io import fits
from astropy.table import Table
from scipy.interpolate import make_smoothing_spline


gold_mask_file = "des-data/DESY3_GOLD_2_2.1.h5"
redmagic_nz_file = "2pt_NG_final_2ptunblind_02_24_21_wnz_covupdate.v2.fits"
maglim_nz_file = "2pt_NG_final_2ptunblind_02_26_21_wnz_maglim_covupdate.fits"
index_file = "des-data/DESY3_indexcat.h5"
metacal_file = "des-data/DESY3_metacal_v03-004.h5"
lens_file = "des-data/DESY3_maglim_redmagic_v0.5.1.h5"
dnf_file = "des-data/DESY3_GOLD_2_2.1_DNF.h5"


def calibrate_shears(index_file, metacal_file):
    delta_gamma = 0.02

    with h5py.File(index_file, 'r') as f:
        # for the selection bias calculation.
        # These are index arrays into the full set.
        s00 = f["/index/metacal/select"][:]
        s1m = f["/index/metacal/select_1m"][:]
        s1p = f["/index/metacal/select_1p"][:]
        s2m = f["/index/metacal/select_2m"][:]
        s2p = f["/index/metacal/select_2p"][:]


    with h5py.File(metacal_file, 'r') as f:
        # The main response term is just the mean of the Rij columns
        # selected by the s00
        R11 = f['catalog/unsheared/R11'][:][s00].mean()
        R22 = f['catalog/unsheared/R22'][:][s00].mean()
        R12 = f['catalog/unsheared/R12'][:][s00].mean()
        R21 = f['catalog/unsheared/R21'][:][s00].mean()
        R_gamma = np.array([[R11, R12], [R21, R22]])

        e1 = f['catalog/unsheared/e_1'][:]
        e2 = f['catalog/unsheared/e_2'][:]


    S11 = (e1[s1p].mean() - e1[s1m].mean()) / delta_gamma
    S22 = (e2[s2p].mean() - e2[s2m].mean()) / delta_gamma
    S12 = (e1[s2p].mean() - e1[s2m].mean()) / delta_gamma
    S21 = (e2[s1p].mean() - e2[s1m].mean()) / delta_gamma
    R_S = np.array([[S11, S12], [S21, S22]])

    R = R_gamma + R_S

    R_inv = np.linalg.inv(R)
    e1, e2 = R_inv @ [e1, e2]

    return s00, e1, e2



def extract_source_samples(index_file, metacal_file, shear_output_file):
    # 1 load the metacal sample, apply the /index/metacal/select selection,
    # calibrate it (R and S), and save it.

    sel, e1, e2 = calibrate_shears(index_file, metacal_file)
    print("Calibrated shears")
    with h5py.File(metacal_file, 'r') as f:
        ra = f['/catalog/unsheared/ra'][:][sel]
        dec = f['/catalog/unsheared/dec'][:][sel]
        weight = f['/catalog/unsheared/weight'][:][sel]

    print("Read source sample")

    with h5py.File(shear_output_file, "w") as f:
        f.create_dataset("ra", data=ra)
        f.create_dataset("dec", data=dec)
        f.create_dataset("e1", data=e1)
        f.create_dataset("e2", data=e2)
        f.create_dataset("weight", data=weight)
    
    # we can just use the source n(z) file for this since it should match,
    # as we are not doing any additional cuts, so no need to extract it from anywhere

def extract_maglim_sample(index_file, lens_file, dnf_file, maglim_output_file):
    with h5py.File(index_file, 'r') as f:
        sel = f["/index/maglim/select"][:]
    print("Read maglim index")

    with h5py.File(lens_file, "r") as f:
        ra = f["/catalog/maglim/ra"][:][sel]
        dec = f["/catalog/maglim/dec"][:][sel]
        weight = f["/catalog/maglim/weight"][:][sel]

    print("Read maglim sample")
    
    with h5py.File(dnf_file, "r") as f:
        # used for estimating the ensemble
        z_mc = f["/catalog/unsheared/zmc_sof"][:][sel]
        # used for the cut
        z_mean = f["/catalog/unsheared/zmean_sof"][:][sel]

    print("Read maglim redshifts")

    with h5py.File(maglim_output_file, "w") as f:
        f.create_dataset("ra", data=ra)
        f.create_dataset("dec", data=dec)
        f.create_dataset("weight", data=weight)
        f.create_dataset("z_sample", data=z_mc)
        f.create_dataset("z", data=z_mean)

    print("Saved maglim info")


def extract_redmagic_sample(index_file, lens_file, redmagic_output_file):

    with h5py.File(index_file, 'r') as f:
        sel = f["/index/redmagic/combined_sample_fid/select"][:]

    print("Read redmagic index")

    with h5py.File(lens_file, "r") as f:
        ra = f["/catalog/redmagic/combined_sample_fid/ra"][:][sel]
        dec = f["/catalog/redmagic/combined_sample_fid/dec"][:][sel]
        weight = f["/catalog/redmagic/combined_sample_fid/weight"][:][sel]
        z = f["/catalog/redmagic/combined_sample_fid/zredmagic"][:][sel]
        z_sample = f["/catalog/redmagic/combined_sample_fid/zredmagic_samp"][:][sel]
    print("Read redmagic sample")

    with h5py.File(redmagic_output_file, "w") as f:
        f.create_dataset("ra", data=ra)
        f.create_dataset("dec", data=dec)
        f.create_dataset("weight", data=weight)
        f.create_dataset("z_sample", data=z_sample)
        f.create_dataset("z", data=z)

def estimate_lens_nz_with_cut(input_file, zmax, output_file):
    with h5py.File(input_file) as f:
        z = f["z"][:]
        z_mc = f["z_sample"][:]   
        weight = f["weight"][:]

    cut = z < zmax
    z_mc = z_mc[cut]
    weight = weight[cut]

    dz = 0.01
    # we go up to well above the max z here because
    # there can be catastrophic outliers sometimes
    bins = np.arange(0, 3.005, dz)

    if z_mc.ndim == 2:
        # the redmagic sample has multiple redshift sample draws
        counts = 0
        nsamp = z_mc.shape[1]
        for i in range(nsamp):
            counts_i, edges = np.histogram(z_mc[:, i], weights=weight, bins=bins)
            counts += counts_i / nsamp
    else:
        counts, edges = np.histogram(z_mc, weights=weight, bins=bins)
    mids = 0.5 * (edges[1:] + edges[:-1])
    np.savetxt(output_file, np.transpose([mids, counts]), header="z n_z")

def estimate_smooth_source_nz(index_file, metacal_file, dnf_file, output_file):

    with h5py.File(index_file, 'r') as f:
        # for the selection bias calculation.
        # These are index arrays into the full set.
        sel = f["/index/metacal/select"][:]

    with h5py.File(metacal_file, 'r') as f:
        weight = f['/catalog/unsheared/weight'][:][sel]

    with h5py.File(dnf_file, "r") as f:
        # used for estimating the ensemble
        z_mc = f["/catalog/unsheared/zmc_sof"][:][sel]

    dz = 0.01
    bins = np.arange(0, 3.005, dz)
    counts, edges = np.histogram(z_mc, weights=weight, bins=bins, density=True)

    mids = 0.5 * (edges[1:] + edges[:-1])

    # smoothing
    s = make_smoothing_spline(mids, counts)
    smooth_nz = s(mids)

    np.savetxt(output_file, np.transpose([mids, smooth_nz]), header="z n_z")



def extract_des_mask_from_gold(mask_file):
    """
    This is a one-off pre-run step to pull
    the DES Y3 mask from the gold mask file.

    Once it's done you can just load the mask
    from the numpy mask file.
    """
    with h5py.File(gold_mask_file, 'r') as f:
        # This is an index of hea
        mask = f["/masks/gold/hpix"][:]
    np.save(mask, mask_file)

def extract_nz(nz_fits_file, lens_output_file, source_output_file):
    """
    Extract the redmagic and maglim n(z) from the
    respective files.
    """
    source_extname = "nz_source"
    lens_extname = "nz_lens"
    nbin_source = 4

    if "maglim" in lens_output_file:
        nbin_lens = 6
    else:
        nbin_lens = 5
    with fits.open(nz_fits_file) as f:
        source_data = f[source_extname].data
        lens_data = f[lens_extname].data
    
        source_z = source_data["Z_MID"]
        nz_source = [source_data[f"BIN{i}"] for i in range(1, nbin_source+1)]

        source_table = Table([source_z] + nz_source, names=["Z"] + [f"BIN{i}" for i in range(1, nbin_source+1)])
        source_table.write(source_output_file, format='ascii.commented_header', overwrite=True)

        lens_z = lens_data["Z_MID"]
        nz_lens = [lens_data[f"BIN{i}"] for i in range(1, nbin_lens+1)]

        lens_table = Table([lens_z] + nz_lens, names=["Z"] + [f"BIN{i}" for i in range(1, nbin_lens+1)])
        lens_table.write(lens_output_file, format='ascii.commented_header', overwrite=True)

def extract_source_nz(nz_fits_file, source_output_file):
    source_extname = "nz_source"
    nbin_source = 4
    with fits.open(nz_fits_file) as f:
        source_data = f[source_extname].data    
        source_header = f[source_extname].header
        source_z = source_data["Z_MID"]
        nz_source = [source_data[f"BIN{i}"] for i in range(1, nbin_source+1)]
        nz_ngal = [source_header[f'NGAL_{i}'] for i in range(1, nbin_source+1)]
        nz_source_combined =  sum([nz_source[i] * nz_ngal[i] for i in range(nbin_source)])

    cut = source_z < 2.0
    source_z = source_z[cut]
    nz_source_combined = nz_source_combined[cut]
    with open(source_output_file, 'w') as f:
        f.write("# z n(z)\n")
        for z, nz in zip(source_z, nz_source_combined):
            f.write(f"{z} {nz}\n")


def extract_all_nz():
    # The source n(z) is the same in the two files so the overwrite doesn't matter here
    extract_nz(redmagic_nz_file, "redmagic_nz.txt", "source_nz.txt")
    extract_nz(maglim_nz_file, "maglim_nz.txt", "source_nz.txt")


if __name__ == "__main__":
    # extract_all_nz()
    # extract_source_nz(maglim_nz_file, "source_nz_combined.txt")
    shear_output_file = "des-data/ridge-shear-sample.h5"
    maglim_output_file = "des-data/ridge-maglim-sample.h5"
    redmagic_output_file = "des-data/ridge-redmagic-sample.h5"
    zmax = 0.9
    maglim_nz_file = "des-data/maglim_nz_zcut_0.9.txt"
    redmagic_nz_file = "des-data/redmagic_nz_zcut_0.9.txt"
    source_nz_file = "des-data/source_nz_smooth.txt"
    # extract_source_samples(index_file, metacal_file, shear_output_file)
    # extract_maglim_sample(index_file, lens_file, dnf_file, maglim_output_file)
    # extract_redmagic_sample(index_file, lens_file, redmagic_output_file)

    # estimate_lens_nz_with_cut(maglim_output_file, zmax, maglim_nz_file)
    # estimate_lens_nz_with_cut(redmagic_output_file, zmax, redmagic_nz_file)

    estimate_smooth_source_nz(index_file, metacal_file, dnf_file, source_nz_file)
