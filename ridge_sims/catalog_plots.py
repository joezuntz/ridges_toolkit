import numpy as np
import matplotlib.pyplot as plt
import healpy
import pyccl
import treecorr
import ridge_sims.config
import ridge_sims.samples
import multiprocessing
import functools
import os

def make_count_map(nside, cat):
    npix = healpy.nside2npix(nside)
    pix = healpy.ang2pix(nside, cat["RA"], cat["DEC"], lonlat=True)
    m = np.bincount(pix, minlength=npix)
    return m

def get_density(cat):
    nside = 512
    m = make_count_map(nside, cat)
    m = healpy.ud_grade(m, nside, power=-2)
    npix_hit = (m>0).sum()
    nsqdeg_per_pixel = healpy.nside2pixarea(nside, degrees=True)
    nsqarcmin_per_pixel = nsqdeg_per_pixel * 3600
    area_hit_sqdeg = npix_hit * nsqdeg_per_pixel
    area_hit_sqarcmin = npix_hit * nsqarcmin_per_pixel
    density = len(cat) / area_hit_sqarcmin
    return density, area_hit_sqdeg

def make_maps(l0, s0):
    nside = 512
    lmap = make_count_map(nside, l0)
    smap = make_count_map(nside, s0)
    print("Made count maps")
    healpy.mollview(lmap,  title="Lens count")
    plt.savefig("./sim-fiducial/lens_count.png")
    plt.close()
    healpy.mollview(smap,  title="Source count")
    plt.savefig("./sim-fiducial/source_count.png")
    plt.close()

    healpy.cartview(lmap, lonra=[30, 60], latra=[-45, -15], xsize=800, title="Lens count", max=50)
    plt.savefig("./sim-fiducial/lens_count_zoom.png")
    plt.close()
    healpy.cartview(smap, lonra=[30, 60], latra=[-45, -15], xsize=800, title="Source density", max=400)
    plt.savefig("./sim-fiducial/source_count_zoom.png")
    plt.close()

def catalog_checks(l0, s0):
    print("Lens density:", get_density(l0))
    print("Source density:", get_density(s0))

def redshift_histograms(l0, s0):
    plt.hist(s0['Z_TRUE'], bins=100, histtype='step', label='Source');
    plt.hist(l0['Z_TRUE'], bins=100, histtype='step', label='Lens');   
    plt.legend()
    plt.xlabel('Redshift')
    plt.ylabel('Count')
    plt.title('Redshift distribution')
    plt.savefig('./sim-fiducial/redshift_histogram.png')
    plt.close()
    print("plotted z hists")

def gen_random(i, n, mask, nside):
    ra = np.random.uniform(0, 360, n)
    dec = np.rad2deg(np.arcsin(np.random.uniform(-1, 1, n)))
    pix = healpy.ang2pix(nside, ra, dec, lonlat=True)
    # get mask value of pixel                                                                                                                                                          
    mask_value = mask[pix]
    # keep only points in mask                                                                                                                                                         
    ra = ra[mask_value > 0]
    dec = dec[mask_value > 0]
    print("Generated random sample ", i)
    return (ra, dec)

def random_points_in_mask(nside, density_sqarcmin=10, thin=10):
    filename = "sim-fiducial/randoms.npy"
    if os.path.exists(filename):
        ra, dec = np.load(filename)
    else:
        mask = ridge_sims.samples.load_mask(nside)
        print("Loaded mask")
        # hugely lazy way to do this for a small sky fraction!
        nsqarcmin_sky = 41252.96125 * 60 * 60
        ntot = int(nsqarcmin_sky * density_sqarcmin)
        print(ntot)
        npix = len(mask)
        nside = healpy.npix2nside(npix)
        chunks = 10
        n = ntot // chunks
        index = list(range(chunks))
        gen = functools.partial(gen_random, n=n, mask=mask, nside=nside)
        with multiprocessing.Pool(chunks) as pool:
            radecs = pool.map(gen, index)

        ra = [radec[0] for radec in radecs]
        dec = [radec[1] for radec in radecs]
        del radecs

        ra = np.concatenate(ra)
        dec = np.concatenate(dec)

        np.save(filename, [ra, dec])
        print("Saving random cat")
    rcat = {"RA": ra, "DEC": dec}
    rsub = {"RA": ra[::thin], "DEC": dec[::thin]}
    return rcat, rsub

def measure_xi(cat):
    c = treecorr.Catalog(
        ra=cat["RA"],
        dec=cat["DEC"],
        g1=cat["G1"],
        g2=cat["G2"],
        ra_units="deg",
        dec_units="deg",
    )
    config = {
        "min_sep": 0.5,
        "max_sep": 300.,
        "nbins": 20,
        "bin_slop": 0.1,
        "sep_units": "arcmin",
        "verbose": 2,
        "flip_g2": True,
    }
    gg = treecorr.GGCorrelation(config)
    gg.process(c)
    return gg

def measure_w(lcat, rancat, rancatsub):
    c = treecorr.Catalog(
        ra=lcat["RA"],
        dec=lcat["DEC"],
        ra_units="deg",
        dec_units="deg",        
    )
    r = treecorr.Catalog(
        ra=rancat["RA"],
        dec=rancat["DEC"],
        ra_units="deg",
        dec_units="deg",
    )
    rsub = treecorr.Catalog(
        ra=rancatsub["RA"],
        dec=rancatsub["DEC"],
        ra_units="deg",
        dec_units="deg",
    )
    config = {
        "min_sep": 0.5,
        "max_sep": 300.,
        "nbins": 20,
        "bin_slop": 0.1,
        "sep_units": "arcmin",
        "verbose": 2,
    }
    nn = treecorr.NNCorrelation(config)
    nr = treecorr.NNCorrelation(config)
    rr = treecorr.NNCorrelation(config)
    nn.process(c)
    print("Done NN")
    nr.process(c, r)
    print("Done NR")
    rr.process(rsub)
    print("Done RR")
    nn.calculateXi(rr=rr, dr=nr, rd=None)
    return nn


def measure_gammat(scat, lcat, rancat):
    sc = treecorr.Catalog(
        ra=scat["RA"],
        dec=scat["DEC"],
        g1=scat["G1"],
        g2=scat["G2"],
        ra_units="deg",
        dec_units="deg",
    )
    lc = treecorr.Catalog(
        ra=lcat["RA"],
        dec=lcat["DEC"],
        ra_units="deg",
        dec_units="deg",        
    )
    r = treecorr.Catalog(
        ra=rancat["RA"],
        dec=rancat["DEC"],
        ra_units="deg",
        dec_units="deg",
    )
    config = {
        "min_sep": 0.5,
        "max_sep": 300.,
        "nbins": 20,
        "bin_slop": 0.1,
        "sep_units": "arcmin",
        "verbose": 2,
        "flip_g2": True,
    }
    ng = treecorr.NGCorrelation(config)
    rg = treecorr.NGCorrelation(config)
    ng.process(lc, sc)
    print("Done NG")
    rg.process(r, sc)
    print("Done RG")
    ng.calculateXi(rg=rg)

    return ng

def get_expected_xi():
    """
    Compute the theoretical correlation functions for the fiducial cosmology
    """
    h = ridge_sims.config.FIDUCIAL_PARAMS["h"]
    Omega_m = ridge_sims.config.FIDUCIAL_PARAMS["Omega_m"]
    Omega_b = ridge_sims.config.FIDUCIAL_PARAMS["Omega_b"]
    sigma8 = ridge_sims.config.FIDUCIAL_PARAMS["sigma8"]
    Omega_c = Omega_m - Omega_b

    # Should be specifying these in the config - do that!
    import camb
    cp = camb.set_params()
    n_s = cp.InitPower.ns

    sample_info = ridge_sims.samples.load_sample_information(lens_type="maglim")


    cosmo = pyccl.Cosmology(Omega_c=Omega_c, Omega_b=Omega_b, h=h, sigma8=sigma8, n_s=n_s)
    stracer = pyccl.WeakLensingTracer(
        cosmo,
        dndz=(sample_info.source_z, sample_info.source_nz[0]),
    )
    bias = np.repeat(sample_info.galaxy_bias[0], len(sample_info.source_z))
    ltracer = pyccl.NumberCountsTracer(
        cosmo,
        dndz=(sample_info.lens_z, sample_info.lens_nz[0]),
        bias=(sample_info.lens_z, bias),
        has_rsd = False
    )
    ell = np.arange(2, 5000)
    theta_arcmin = np.geomspace(1.0, 300.0, 100)
    theta_deg = theta_arcmin / 60

    c_ell_ee = cosmo.angular_cl(stracer, stracer, ell)
    xi_plus = cosmo.correlation(ell=ell, C_ell=c_ell_ee, theta=theta_deg, type="GG+")
    xi_minus = cosmo.correlation(ell=ell, C_ell=c_ell_ee, theta=theta_deg, type="GG-")

    c_ell_density = cosmo.angular_cl(ltracer, ltracer, ell)
    xi_density = cosmo.correlation(ell=ell, C_ell=c_ell_density, theta=theta_deg, type="NN")

    c_ell_ggl = cosmo.angular_cl(ltracer, stracer, ell)
    xi_ggl = cosmo.correlation(ell=ell, C_ell=c_ell_ggl, theta=theta_deg, type="NG")
    return theta_arcmin, xi_plus, xi_minus, xi_density, xi_ggl



def plot_shear(s0, theta_theory, xip_theory, xim_theory):
    xi = measure_xi(s0)

    # Shear plot
    plt.figure()
    plt.errorbar(xi.meanr, xi.xip, xi.varxip**0.5, fmt='.', label='Measured xi+')
    plt.errorbar(xi.meanr, xi.xim, xi.varxim**0.5, fmt='.', label='Measured xi-')
    plt.plot(theta_theory, xip_theory, label='Theory xi+')
    plt.plot(theta_theory, xim_theory, label='Theory xi-')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig("./sim-fiducial/xi.png")
    plt.close()

def plot_w(l0, randoms, randoms_sub, theta_theory, w_theory):
    wtheta = measure_w(l0, randoms, randoms_sub)
    plt.figure()
    plt.errorbar(wtheta.meanr, wtheta.xi, wtheta.varxi**0.5, fmt='.', label='Measured w(theta)')
    plt.plot(theta_theory, w_theory, label='Theory w(theta)')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.savefig("./sim-fiducial/wtheta.png")
    plt.close()

def plot_ggl(s0, l0, rancat, theta_theory, gammat_theory):
    gammat = measure_gammat(s0, l0, rancat)
    plt.figure()
    plt.errorbar(gammat.meanr, gammat.xi, gammat.varxi**0.5, fmt='.', label='Measured gamma_t')
    plt.plot(theta_theory, gammat_theory, label='Theory gamma_t')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    plt.savefig("./sim-fiducial/gammat.png")
    plt.close()



    

def correlation_functions(l0, s0):
    nside = 512
    theta_theory, xip_theory, xim_theory, w_theory, gammat_theory = get_expected_xi()
    print("Got theory")
    rancat, rancat_sub = random_points_in_mask(nside, density_sqarcmin=10)
    print("Got randoms")

    plot_shear(s0, theta_theory, xip_theory, xim_theory)
    plot_w(l0, rancat, rancat_sub, theta_theory, w_theory)
    plot_ggl(s0, l0, rancat, theta_theory, gammat_theory)



if __name__ == "__main__":
    l0 = np.load("./sim-fiducial/lens_catalog_0.hdf5")
    print("Loaded lens sample")
    s0 = np.load("./sim-fiducial/source_catalog_0.hdf5")
    print("Loaded shear sample")
    #catalog_checks(l0, s0)
    #redshift_histograms(l0, s0)
    #make_maps(l0, s0)
    correlation_functions(l0, s0)
