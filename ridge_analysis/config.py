import dataclasses
import yaml
import numpy as np

@dataclasses.dataclass(kw_only=True)
class Config:
    # The dataclass decorator auto-creates an __init__ method and other utility methods.
    def to_yaml(self, filename):
        d = dataclasses.asdict(self)
        with open(filename, 'w') as f:
            yaml.dump(d, f)
    
    @classmethod
    def from_yaml(cls, filename):
        with open(filename, 'r') as f:
            d = yaml.safe_load(f)
        return cls(**d)


@dataclasses.dataclass(kw_only=True)
class DredgeConfig:
    lens_catalog_file: str
    ridge_point_file: str
    checkpoint_dir: str
    checkpoint_dir: str
    bandwidth: float = 6.0  # in arcmin
    ridge_points: int = 500_000
    neighbours: int = 5000
    convergence: float = 0.03 # in arcmin
    seed: int = 0
    lens_zmin: float = 0.0
    lens_zmax: float = 100.0


    def bandwidth_radians(self):
        return np.radians(self.bandwidth / 60.0)
    
    def convergence_radians(self):
        return np.radians(self.convergence / 60.0)


@dataclasses.dataclass(kw_only=True)
class SegmentationConfig:
    ridge_point_file: str
    ridge_file: str
    density_percentile: float = 0.0


@dataclasses.dataclass(kw_only=True)
class ShearConfig:
    source_catalog_file: str
    shear_file: str
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
