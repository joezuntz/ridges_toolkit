# import os
import numpy as np
# import pandas as pd
import h5py
# import time
# from scipy.spatial import KDTree
# from sklearn.neighbors import NearestNeighbors
# import matplotlib.pyplot as plt
# 
# import astropy.units as u
# import dredge
import healpy as hp
from ridge_analysis.shear import measure_shear
from .io import RidgeSegmentCatalog, LensCatalog, SourceCatalog, RidgePointCatalog
from .config import DredgeConfig, SegmentationConfig, ShearConfig


###############################################################

# ------------------ RIDGE FINDER -----------------------------

##############################################################

def locate_ridge_points(lens_catalog: LensCatalog, dredge_config: DredgeConfig, comm) -> RidgePointCatalog:
    from .. import dredge
    # In Dredge every rank needs the whole catalog, and the splitting is done
    # over the mesh points. We could certainly improve this if needed by giving
    # each rank only the lens catalog nearby its mesh points, but for now we
    # just load the whole catalog on every rank.
    lens_catalog.load(comm=comm, split_over_ranks=False)

    # Apply any redshift cuts
    if dredge_config.lens_zmin is not None or dredge_config.lens_zmax is not None:
        lens_catalog.cut_to_redshift_range(dredge_config.lens_zmin, dredge_config.lens_zmax)

    # Run the filament finder with the given configuration.
    # This will save checkpoints to the specified directory, and resume from them if they exist,
    # so be careful to delete them if you change the configuration and want to re-run in the 
    # same directory.
    ridges, _, final_density = dredge.find_filaments(
        lens_catalog.dec_ra_in_radians(),
        bandwidth=dredge_config.bandwidth_radians(),
        convergence=dredge_config.convergence_radians(),
        distance_metric='haversine',
        n_neighbors=dredge_config.neighbours,
        comm=comm,
        checkpoint_dir=dredge_config.checkpoint_dir,
        resume=True,
        seed=dredge_config.seed,
        mesh_size=dredge_config.mesh_size
    )

    if comm is None or comm.rank == 0:
        output = RidgePointCatalog(dredge_config.ridge_point_file)
        output.set_column("density", final_density)
        output.set_column("ra", np.degrees(ridges[:, 1]))
        output.set_column("dec", np.degrees(ridges[:, 0]))
        output.save()

    return output


def segment_ridges(ridge_point_catalog: RidgePointCatalog, segmentation_config: SegmentationConfig) -> RidgeSegmentCatalog:
    from .segmentation import build_mst, detect_branch_points, split_mst_at_branches, segment_filaments_with_dbscan

    ridge_point_catalog.load()

    # Apply density cut if specified
    if segmentation_config.density_percentile > 0.0:
        ridge_point_catalog.apply_density_cut(segmentation_config.density_percentile)

    # Apply edge filter to remove points near survey boundaries
    print("TODO EDGE FILTER!")

    ridges = ridge_point_catalog.dec_ra_in_radians()

    mst = build_mst(ridges)
    branch_points = detect_branch_points(mst)
    filament_segments = split_mst_at_branches(mst, branch_points)
    filament_labels = segment_filaments_with_dbscan(ridges, filament_segments)

    output = RidgeSegmentCatalog(segmentation_config.ridge_file)
    output.set_column("ra", np.degrees(ridges[:, 1]))
    output.set_column("dec", np.degrees(ridges[:, 0]))
    output.set_column("ridge_id", filament_labels)
    output.save()
    return output

def measure_ridge_shear(ridge_segments: RidgeSegmentCatalog, source_catalog: SourceCatalog, shear_config: ShearConfig):
    # --- Shear calculations ---

    shear_table = measure_shear(ridge_segments,
                  source_catalog,
                  shear_config.output_shear_file,
                  k=1,
                  num_bins=shear_config.num_bins,
                  comm=None,
                  flip_g1=shear_config.flip_g1,
                  flip_g2=shear_config.flip_g2,
                  nside_coverage=shear_config.nside_coverage,
                  min_distance_arcmin=shear_config.min_distance_arcmin,
                  max_distance_arcmin=shear_config.max_distance_arcmin,
                  skip_end_points=shear_config.skip_end_points,
                  min_filament_points=shear_config.min_filament_points
    )

    shear_table.save(shear_config.output_shear_file)


