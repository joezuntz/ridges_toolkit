import dataclasses
import yaml
import numpy as np


@dataclasses.dataclass(kw_only=True)
class Config:
    # The dataclass decorator auto-creates an __init__ method and other utility methods.
    def to_yaml(self, filename):
        d = dataclasses.asdict(self)
        with open(filename, "w") as f:
            yaml.dump(d, f)

    @classmethod
    def from_yaml(cls, filename):
        with open(filename, "r") as f:
            d = yaml.safe_load(f)
        return cls(**d)


@dataclasses.dataclass(kw_only=True)
class DredgeConfig:
    lens_catalog_file: str
    ridge_point_file: str
    checkpoint_dir: str
    bandwidth: float = 6.0  # in arcmin
    num_ridge_points: int = 500_000
    tree_nside: int = 128
    convergence: float = 0.03  # in arcmin
    seed: int = 0
    lens_zmin: float = 0.0
    lens_zmax: float = 100.0
    # Whether to shift longitudes to avoid 0/360 degree boundary issues
    shift_180: bool = False

    help = {
        "lens_catalog_file": "Path to the lens catalog file (HDF5 format).",
        "ridge_point_file": "Path to the output ridge point file (HDF5 format).",
        "checkpoint_dir": "Directory to save intermediate results and checkpoints.",
        "bandwidth": "Bandwidth for density estimation in arcminutes.",
        "num_ridge_points": "Number of ridge points to generate.",
        "tree_nside": "The nside parameter of the tree-like structure used in finding nearby points.",
        "convergence": "Convergence threshold for ridge point identification in arcminutes.",
        "seed": "Random seed for reproducibility.",
        "lens_zmin": "Minimum redshift of lenses to consider.",
        "lens_zmax": "Maximum redshift of lenses to consider.",
        "shift_180": "Whether to shift longitudes by 180 degrees and back at the end to avoid boundary issues."
    }

    def bandwidth_radians(self):
        return np.radians(self.bandwidth / 60.0)

    def convergence_radians(self):
        return np.radians(self.convergence / 60.0)


@dataclasses.dataclass(kw_only=True)
class SegmentationConfig:
    ridge_point_file: str
    ridge_file: str
    density_percentile: float = 0.0
    mst_neighbours: int = 10
    do_spline: bool = False
    n_spline_points: int = 100

    help = {
        "ridge_point_file": "Path to the input ridge point file (HDF5 format).",
        "ridge_file": "Path to the output ridge file (HDF5 format).",
        "density_percentile": "Percentile threshold for density to filter ridge points (0-100).",
        "mst_neighbours": "Number of nearest neighbors to consider when building the minimum spanning tree for segmentation.",
        "do_spline": "Whether to perform spline interpolation on the segmented ridges.",
        "spline_points": "Number of points to use for spline interpolation if do_spline is True."
    }   


@dataclasses.dataclass(kw_only=True)
class ShearConfig:
    output_shear_file: str
    source_catalog_file: str
    ridge_file: str
    flip_g1: bool = False
    flip_g2: bool = False
    num_bins: int = 20
    min_distance_arcmin: float = 1.0
    max_distance_arcmin: float = 60.0
    nside_coverage: int = 32
    min_filament_points: int = 0
    skip_end_points: bool = False
    source_zmin: float = 0.0
    source_zmax: float = 100.0
    add_sigma_e: float = 0.0
    seed: int = 0,

    help = {
        "output_shear_file": "Path to the output shear file (text format)",
        "source_catalog_file": "Path to the source catalog file (HDF5 format).",
        "ridge_file": "Path to the input ridge file (HDF5 format).",
        "flip_g1": "Whether to flip the sign of the g1 shear component.",
        "flip_g2": "Whether to flip the sign of the g2 shear component.",
        "num_bins": "Number of logarithmic bins for shear measurement.",
        "min_distance_arcmin": "Minimum distance from ridge points to consider for shear measurement in arcminutes.",
        "max_distance_arcmin": "Maximum distance from ridge points to consider for shear measurement in arcminutes.",
        "nside_coverage": "Healpix nside for determining ridge point coverage.",
        "min_filament_points": "Minimum number of points in a filament to be included in the shear measurement.",
        "skip_end_points": "Whether to skip the end points of filaments in the shear measurement (not functional).",
        "source_zmin": "Minimum redshift of source galaxies to consider (not functional).",
        "source_zmax": "Maximum redshift of source galaxies to consider (not functional).",
        "add_sigma_e": "sigma_e value to add to shears"
    }
