import yaml
import scipy.stats.qmc
import os

FIDUCIAL_PARAMS = dict(
    h = 0.7,
    Omega_m = 0.3,
    Omega_b = 0.045,
    sigma8 = 0.8,
)

DEFAULT_NPROCESS = int(os.environ.get("RIDGE_NPROCESS", "1"))

class Config:
    """
    This object stores our configuration for the simulation.
    """
    def __init__(self,
                 sim_dir="sim-fiducial",
                 lens_type="maglim",
                 lmax=10_000,
                 combined=True,
                 zmax=3.0,
                 dx=150.0,
                 nside=4096,
                 nprocess=DEFAULT_NPROCESS,
                 h=FIDUCIAL_PARAMS["h"],
                 Omega_m=FIDUCIAL_PARAMS["Omega_m"],
                 Omega_b=FIDUCIAL_PARAMS["Omega_b"],
                 sigma8=FIDUCIAL_PARAMS["sigma8"],
                 seed=42,
                 include_shape_noise=None, # option for almost no noise,
                 lsst=None,
                 lsst10_nz=False,  # NEW: DES with lsst nz option
                 ):
        self.sim_dir = sim_dir
        self.lens_type = lens_type
        self.lmax = lmax
        self.combined = combined
        self.zmax = zmax
        self.dx = dx
        self.nside = nside
        self.nprocess = nprocess
        self.h = h
        self.Omega_m = Omega_m
        self.Omega_b = Omega_b
        self.sigma8 = sigma8
        self.seed = seed
        self.include_shape_noise = include_shape_noise # no noise option
        self.lsst = lsst
        self.lsst10_nz = lsst10_nz  # NEW: DES with lsst nz option
        # Set file output output names based on sim_dir
        self.set_file_names()


    def set_file_names(self):
        sim_dir = self.sim_dir
        self.shell_cl_file = f"{sim_dir}/shell_cls.npy"
        self.g_ell_file = f"{sim_dir}/g_ell.pkl"
        self.source_cat_file = f"{sim_dir}/source_catalog_{{}}.hdf5"
        self.lens_cat_file = f"{sim_dir}/lens_catalog_{{}}.hdf5"

    @classmethod
    def from_yaml(cls, filename):
        with open(filename) as f:
            d = yaml.safe_load(f)
        obj = cls()
        obj.__dict__.update(d)
        obj.set_file_names()

    def to_yaml(self, filename):
        with open(filename, "w") as f:
            yaml.dump(self.__dict__, f)

    def save(self):
        os.makedirs(self.sim_dir, exist_ok=True)
        filename = f"{self.sim_dir}/config.yaml"
        with open(filename, "w") as f:
            yaml.dump(self, f)




def latin_hypercube_points(n, bounds=None):
    """
    Iterate through Latin Hypercube samples 
    """
    if bounds is None:
        bounds = [
            (0.5, 0.9), # h
            (0.15, 0.45), # Omega_m
            # (0.02, 0.07), # Omega_b  #skip omega_b to start with
            (0.7, 0.8), # sigma8
        ]
    sampler = scipy.stats.qmc.LatinHypercube(len(bounds))
    for sample in sampler.random(n):
        x = [b[0] + s * (b[1] - b[0]) for s, b in zip(sample, bounds)]
        yield x

def latin_hypercube_configurations(n):
    for i, params in enumerate(latin_hypercube_points(n)):
        config = Config(
            sim_dir = f"lhc/sim-i{i}",
            h = params[0],
            Omega_m = params[1],
            sigma8 = params[2],
        )
        yield config
