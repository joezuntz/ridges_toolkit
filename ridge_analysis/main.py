import numpy as np
from .shear import measure_shear
from .io import RidgeSegmentCatalog, LensCatalog, SourceCatalog, RidgePointCatalog
from .config import DredgeConfig, SegmentationConfig, ShearConfig


###############################################################

# ------------------ RIDGE FINDER -----------------------------

##############################################################

def locate_ridge_points(dredge_config: DredgeConfig, comm) -> RidgePointCatalog:
    import dredge
    # In Dredge every rank needs the whole catalog, and the splitting is done
    # over the mesh points. We could certainly improve this if needed by giving
    # each rank only the lens catalog nearby its mesh points, but for now we
    # just load the whole catalog on every rank.
    lens_catalog = LensCatalog(dredge_config.lens_catalog_file)
    lens_catalog.load(comm=comm, split_over_ranks=False)

    # Apply any redshift cuts
    if dredge_config.lens_zmin is not None or dredge_config.lens_zmax is not None:
        lens_catalog.cut_to_redshift_range(dredge_config.lens_zmin, dredge_config.lens_zmax)

    coordinates = lens_catalog.dec_ra_in_radians()
    if dredge_config.shift_180:
        coordinates[:, 1] = (coordinates[:, 1] + np.pi) % (2 * np.pi)
        if comm is None or comm.rank == 0:
            print("Shifting longitudes to avoid 0/360 degree boundary issues")
            print("New RA range:", np.degrees(coordinates[:, 1].min()), np.degrees(coordinates[:, 1].max()))


    # Run the filament finder with the given configuration.
    # This will save checkpoints to the specified directory, and resume from them if they exist,
    # so be careful to delete them if you change the configuration and want to re-run in the 
    # same directory.
    ridges, _, final_density = dredge.find_filaments(
        coordinates,
        bandwidth=dredge_config.bandwidth_radians(),
        convergence=dredge_config.convergence_radians(),
        distance_metric='haversine',
        n_neighbors=dredge_config.neighbours,
        comm=comm,
        checkpoint_dir=dredge_config.checkpoint_dir,
        resume=True,
        seed=dredge_config.seed,
        num_ridge_points=dredge_config.num_ridge_points
    )


    if comm is None or comm.rank == 0:
        if dredge_config.shift_180:
            # undo the shift
            ridges[:, 1] = (ridges[:, 1] - np.pi) % (2 * np.pi)
            if comm is None or comm.rank == 0:
                print("Shifting ridge points back to original RA range")
                print("Ridge RA range:", np.degrees(ridges[:, 1].min()), np.degrees(ridges[:, 1].max()))
        output = RidgePointCatalog(dredge_config.ridge_point_file)
        output.set_column("density", final_density)
        output.set_column("ra", np.degrees(ridges[:, 1]))
        output.set_column("dec", np.degrees(ridges[:, 0]))
        output.save()
    else:
        output = None

    return output


def segment_ridges(segmentation_config: SegmentationConfig) -> RidgeSegmentCatalog:
    from .segmentation import build_mst, detect_branch_points, split_mst_at_branches, segment_filaments_with_dbscan


    # If the catalog is already loaded this will do nothing, otherwise it will load the data from disk.
    # Ridge segmentation is very fast, so we don't parallelize this step.
    ridge_point_catalog = RidgePointCatalog(segmentation_config.ridge_point_file)
    ridge_point_catalog.load()

    # Apply density cut if specified
    if segmentation_config.density_percentile > 0.0:
        ridge_point_catalog.apply_density_cut(segmentation_config.density_percentile)

    # Apply edge filter to remove points near survey boundaries
    print("TODO EDGE FILTER!")

    ridges = ridge_point_catalog.dec_ra_in_radians()

    mst = build_mst(ridges, k=segmentation_config.mst_neighbours)
    branch_points = detect_branch_points(mst)
    filament_segments = split_mst_at_branches(mst, branch_points)
    filament_labels = segment_filaments_with_dbscan(ridges, filament_segments)

    output = RidgeSegmentCatalog(segmentation_config.ridge_file)
    output.set_column("ra", np.degrees(ridges[:, 1]))
    output.set_column("dec", np.degrees(ridges[:, 0]))
    output.set_column("ridge_id", filament_labels)
    output.save()
    return output

def measure_ridge_shear(shear_config: ShearConfig):

    ridge_segments = RidgeSegmentCatalog(shear_config.ridge_file)
    ridge_segments.load()

    source_catalog = SourceCatalog(shear_config.source_catalog_file)
    source_catalog.load()

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

    shear_table.save()
    return shear_table


