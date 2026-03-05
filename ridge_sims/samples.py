import numpy as np
import healpy
import yaml
from types import SimpleNamespace

class SampleInfo(SimpleNamespace):
    pass

mask_filename = "des-data/desy3_gold_mask.npy"
lsst_mask_filename_y1 = "lsst-data/lsst_y1_wfd_exgal_mask_nside_64.fits"
lsst_mask_filename_y10 = "lsst-data/lsst_y10_wfd_exgal_mask_nside_64.fits"

source_nz_filename = "des-data/source_nz_smooth.txt"
tomographic_maglim_nz_filename = "des-data/maglim_nz.txt"
tomographic_redmagic_nz_filename = "des-data/redmagic_nz.txt"

combined_maglim_nz_filename = "des-data/maglim_nz_zcut_0.9.txt"
combined_redmagic_nz_filename = "des-data/redmagic_nz_zcut_0.9.txt"


# from https://arxiv.org/pdf/2105.13546
tomographic_maglim_bias = [
    1.40,
    1.60,
    1.82,
    1.70,
    1.91,
    1.73,
]
tomographic_redmagic_bias = [
    1.74,
    1.82,
    1.92,
    2.15,
    2.32,
]

combined_maglim_bias = np.mean(tomographic_maglim_bias)
combined_redmagic_bias = np.mean(tomographic_redmagic_bias)

tomographic_maglim_number_densities = [
    0.15,
    0.107,
    0.109,
    0.146,
    0.106,
    0.1,
]

maglim_combined_number_densities = sum(tomographic_maglim_number_densities)

tomographic_redmagic_number_densities = [
    0.022,
    0.038,
    0.058,
    0.029,
    0.025,
]


redmagic_combined_number_densities = sum(tomographic_redmagic_number_densities)

tomographic_sigma_e = [0.27, 0.27, 0.27, 0.27]
combined_sigma_e = [np.mean(tomographic_sigma_e)]

# Number density per bin for DES (Giulia)
tomographic_source_number_densities = [
    1.475584985490327, 
    1.479383426887689,
    1.483671693529899,
    1.461247850098986,
]

combined_source_number_densities = sum(tomographic_source_number_densities)

def load_lsst_sample_information(lsst, combined):
    """
    Load LSST-like sample information from files from the LSST forecasting repository.

    Parameters
    ----------
    lsst : int
        If 1, use LSST Y1 number densities.
        If 10, use LSST Y10 number densities.
    combined : bool
        Whether to use the combined source and lens samples. Default is True.

    Returns
    -------
    sample : SampleInfo
        The object containing the number density information.
    
    """
    if lsst not in [1, 10]:
        raise ValueError("lsst must be 1 or 10 if not False/0")
    
    # Load the n_eff from the numpy files from forecasting. They are already in units of
    # objects per arcmin^2
    clustering_neff = np.load(f'lsst-data/lsst_n_eff_clustering_year_{lsst}.npy', allow_pickle=True).item()["n_gal"]
    lensing_neff = np.load(f'lsst-data/lsst_n_eff_lensing_year_{lsst}.npy', allow_pickle=True).item()["n_gal"]
    clustering_neff = np.array(clustering_neff)
    lensing_neff =  np.array(lensing_neff)
    print(clustering_neff.shape, lensing_neff.shape, clustering_neff.dtype, lensing_neff.dtype)

    nbin_source = len(lensing_neff)
    nbin_lens = len(clustering_neff)

    # Load the n(z) from the numpy files from forecasting. I think these are all normalized to 1
    # already, but just in case we will normalize them again below.
    source_nz_data = np.load(f"lsst-data/srd_source_bins_year_{lsst}.npy", allow_pickle=True).item()
    source_z = source_nz_data['redshift_range']
    source_nz = np.array([source_nz_data['bins'][i] for i in range(nbin_source)])

    lens_nz_data = np.load(f"lsst-data/srd_lens_bins_year_{lsst}.npy", allow_pickle=True).item()
    lens_z = lens_nz_data['redshift_range']
    lens_nz = np.array([lens_nz_data['bins'][i] for i in range(nbin_lens)])


    # Load galaxy bias from the yaml file from forecasting
    galaxy_bias = np.zeros(nbin_lens)
    with open(f"lsst-data/linear_galaxy_bias_parameters_y{lsst}.yaml") as f:
        galaxy_bias_data = yaml.safe_load(f)
        for i in range(nbin_lens):
            galaxy_bias[i] = galaxy_bias_data[f"b_{i+1}"]

    # In the forecasting they always use sigma_e = 0.26
    sigma_e  = np.full(lensing_neff.shape, 0.26)


    # Normalize both the source and lens
    for i in range(nbin_source):
        source_nz[i] /= np.trapezoid(source_nz[i], source_z)
        source_nz[i] *= lensing_neff[i]
    
    for i in range(nbin_lens):
        lens_nz[i] /= np.trapezoid(lens_nz[i], lens_z)
        lens_nz[i] *= clustering_neff[i]

    # If we are combining into a single tomographic bin then:
    if combined:
        # The n_effs just sum
        lensing_neff = [np.sum(lensing_neff)]
        clustering_neff = [np.sum(clustering_neff)]
        # The galaxy bias is the mean. We should ideally use
        # the weighted mean but this is close enough as there is
        # not much variation in the neffs
        galaxy_bias = [np.mean(galaxy_bias)]
        # This one is correct as the source sample has equal numbers
        # of objets in each bin by construction
        sigma_e = [np.mean(sigma_e)]

        # The n(z)'s sum because they are normalized by the neff above.
        source_nz = np.array([np.sum(source_nz, axis=0)])
        lens_nz = np.array([np.sum(lens_nz, axis=0)])
        # The counts are now just one bin each
        nbin_source = 1
        nbin_lens = 1

    # Collect everything into the SampleInfo object
    sample = SampleInfo()
    sample.nbin_source = nbin_source
    sample.nbin_lens = nbin_lens
    sample.source_z = source_z
    sample.source_nz = source_nz
    sample.lens_z = lens_z
    sample.lens_nz = lens_nz
    sample.lens_number_densities = lensing_neff
    sample.galaxy_bias = galaxy_bias
    sample.source_number_densities = clustering_neff
    sample.sigma_e = sigma_e

    return sample

    

def load_sample_information(lens_type, combined=True, lsst=False): 
    """
    Create and return a SampleInfo object with the number density information
    for the source and lens samples, and galaxy bias for the latter. 
    
    The number densities are taken from the DES Y3 data release.

    Parameters
    ----------
    lens_type : str
        The type of lens sample to use. Must be 'maglim' or 'redmagic'.

    combined : bool 
        Whether to use the combined source and lens samples. Default is True.
        If False use tomographic samples.

    lsst: int:
        If 0, use DES Y3 number densities.
        If 1, use LSST Y1 number densities.
        If 10, use LSST Y10 number densities.

    Returns
    -------
    sample : SampleInfo
        The object containing the number density information.
    """

    if lsst:
        return load_lsst_sample_information(lsst, combined)

    source_data = np.loadtxt(source_nz_filename).T

    source_z = source_data[0]
    source_nz = source_data[1:]

    if combined:
        source_number_densities = [combined_source_number_densities]
        sigma_e = combined_sigma_e
    else:
        source_number_densities = tomographic_source_number_densities
        sigma_e = tomographic_sigma_e


    if lens_type == "maglim":
        if combined:
            lens_number_densities = [maglim_combined_number_densities]
            galaxy_bias = [np.mean(combined_maglim_bias)]
            lens_data = np.loadtxt(combined_maglim_nz_filename).T
        else:
            sample.lens_number_densities = tomographic_maglim_number_densities
            sample.galaxy_bias = tomographic_maglim_bias
            lens_data = np.loadtxt(tomographic_maglim_nz_filename).T
            
    else:
        if combined:
            lens_number_densities = [redmagic_combined_number_densities]
            galaxy_bias = np.mean(tomographic_redmagic_bias)
            lens_data = np.loadtxt(combined_redmagic_nz_filename).T
        else:
            lens_number_densities = tomographic_redmagic_number_densities
            galaxy_bias = tomographic_redmagic_bias
            lens_data = np.loadtxt(tomographic_redmagic_nz_filename).T

    lens_z = lens_data[0]
    lens_nz = lens_data[1:]
    nbin_lens = len(lens_nz)
    nbin_source = len(source_nz)

    for i in range(nbin_source):
        source_nz[i] /= np.trapezoid(source_nz[i], source_z)
        source_nz[i] *= source_number_densities[i]
    
    for i in range(nbin_lens):
        lens_nz[i] /= np.trapezoid(lens_nz[i], lens_z)
        lens_nz[i] *= lens_number_densities[i]


    sample = SampleInfo()
    sample.nbin_source = nbin_source
    sample.nbin_lens = nbin_lens
    sample.source_z = source_z
    sample.source_nz = source_nz
    sample.lens_z = lens_z
    sample.lens_nz = lens_nz
    sample.lens_number_densities = lens_number_densities
    sample.galaxy_bias = galaxy_bias
    sample.source_number_densities = source_number_densities
    sample.sigma_e = sigma_e


    return sample



##################################################################################
#      Test functions to combine DES simulation and LSST Y10 nz samples
##################################################################################
def _load_lsst_y10_bins(path, nbin=None):
    """
    Load LSST SRD bins from npy dict with keys:
      - 'redshift_range'
      - 'bins' (indexable container)

    If nbin is given, reads only the first nbin bins.
    Returns (z, nz) with nz shape (nbin, nz).
    """
    d = np.load(path, allow_pickle=True).item()
    z = np.asarray(d["redshift_range"])
    bins = d["bins"]

    if nbin is None:
        # try to infer length
        try:
            nbin = len(bins)
        except TypeError:
            raise ValueError(f"Cannot infer number of bins in {path}")

    nz = np.array([bins[i] for i in range(nbin)])
    if nz.ndim == 1:
        nz = nz[None, :]
    if nz.ndim != 2:
        raise ValueError(f"{path}: bins has unexpected shape {nz.shape}")
    return z, nz



def _normalize_then_scale(nz, z, target_densities):
    nz = np.asarray(nz, dtype=float)
    z = np.asarray(z, dtype=float)
    target_densities = np.asarray(target_densities, dtype=float)

    if nz.ndim != 2:
        raise ValueError(f"nz must be 2D (nbin,nz). Got {nz.shape}")
    if nz.shape[0] != target_densities.shape[0]:
        raise ValueError(f"Bin mismatch: nz bins={nz.shape[0]} but target bins={target_densities.shape[0]}")

    out = np.empty_like(nz, dtype=float)
    for i in range(nz.shape[0]):
        norm = np.trapezoid(nz[i], z)
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError(f"Non-positive/invalid normalization for bin {i}: {norm}")
        out[i] = (nz[i] / norm) * target_densities[i]
    return out



def load_sample_information_advanced(lens_type, combined=True, lsst=False, lsst10_nz=False):
    """
    Same as load_sample_information, but if lsst10_nz=True (and lsst is False/None),
    replace DES n(z) shapes with LSST Y10 shapes while keeping DES densities/bias/sigma_e.
    """
    # 1) Build the canonical sample first (DES or full LSST)
    sample = load_sample_information(lens_type, combined=combined, lsst=lsst)

    # 2) Full LSST mode: hybrid is irrelevant
    if lsst:
        return sample

    # 3) Hybrid: overwrite ONLY the n(z) shapes
    if lsst10_nz:
        y = 10

        if not combined:
            
            raise ValueError("Hybrid LSST10_nz mode is implemented only for combined=True.")

        # Load LSST Y10 shapes; for combined=True we want 1 bin each.
        source_z, source_nz_lsst = _load_lsst_y10_bins(
            f"lsst-data/srd_source_bins_year_{y}.npy", nbin=None
        )
        lens_z, lens_nz_lsst = _load_lsst_y10_bins(
            f"lsst-data/srd_lens_bins_year_{y}.npy", nbin=None
        )

        # Combine LSST bins into a single shape (sum the per-bin curves)
        source_shape = np.sum(source_nz_lsst, axis=0)[None, :]
        lens_shape   = np.sum(lens_nz_lsst, axis=0)[None, :]

        # Overwrite z grids + renormalize to DES densities already stored in sample
        sample.source_z = source_z
        sample.source_nz = _normalize_then_scale(source_shape, source_z, sample.source_number_densities)

        sample.lens_z = lens_z
        sample.lens_nz = _normalize_then_scale(lens_shape, lens_z, sample.lens_number_densities)

    return sample


#################################################################################################




def load_mask(nside, lsst=False):
    if lsst:
        if lsst not in [1, 10]:
            raise ValueError("lsst must be 1 or 10 if not False/0")
    if lsst:
        input_mask_nside = 64
    else:
        input_mask_nside = 4096

    if lsst == 1:
        mask = healpy.read_map(lsst_mask_filename_y1, verbose=False)
    elif lsst == 10:
        mask = healpy.read_map(lsst_mask_filename_y10, verbose=False)
    else:
        hit_pix = np.load(mask_filename)
        mask = np.zeros(healpy.nside2npix(input_mask_nside))
        mask[hit_pix] = 1
        mask = healpy.reorder(mask, n2r = True)

    # degrade mask to nside quality of simulation
    mask = healpy.ud_grade(mask, nside_out = nside)
    return mask

