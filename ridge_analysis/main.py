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
    # each rank only the lens catalog nearby its mesh points, but for now we
    # just load the whole catalog on every rank.
    lens_catalog = LensCatalog(dredge_config.lens_catalog_file)
    lens_catalog.load(comm=comm, split_over_ranks=False)

    # Apply any redshift cuts
    need_zmin = (dredge_config.lens_zmin is not None) and (dredge_config.lens_zmin != 0)
    need_zmax = (dredge_config.lens_zmax is not None) and (dredge_config.lens_zmax < 99.99)
    if need_zmin or need_zmax:
        lens_catalog.cut_to_redshift_range(dredge_config.lens_zmin, dredge_config.lens_zmax)

    coordinates = lens_catalog.dec_ra_in_radians()
    if dredge_config.shift_180:
        coordinates[:, 1] = (coordinates[:, 1] + np.pi) % (2 * np.pi)
        if comm is None or comm.rank == 0:
            print("Shifting longitudes to avoid 0/360 degree boundary issues")
            print("New RA range:", np.degrees(coordinates[:, 1].min()), np.degrees(coordinates[:, 1].max()))

    # Run the filament finder with the given configuration.
    # This will save checkpoints to the specified directory, and resume from them if they exist,
    # so be careful to delete them if you change the configuration and want to re-run in the
    # same directory.
    ridges, _, final_density = dredge.find_filaments(
        coordinates,
        bandwidth=dredge_config.bandwidth_radians(),
        convergence=dredge_config.convergence_radians(),
        distance_metric="haversine",
        n_neighbors=dredge_config.neighbours,
        comm=comm,
        checkpoint_dir=dredge_config.checkpoint_dir,
        resume=True,
        seed=dredge_config.seed,
        num_ridge_points=dredge_config.num_ridge_points,
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


def segment_ridges(segmentation_config: SegmentationConfig, comm) -> RidgeSegmentCatalog:
    from .segmentation import build_mst, detect_branch_points, split_mst_at_branches, segment_filaments_with_dbscan, interpolate_segments

    rank = comm.rank if comm is not None else 0
    size = comm.size if comm is not None else 1
    if rank == 0:
        # Load the data from disk.
        ridge_point_catalog = RidgePointCatalog(segmentation_config.ridge_point_file)
        ridge_point_catalog.load()

        # Apply density cut if specified
        if segmentation_config.density_percentile > 0.0:
            ridge_point_catalog.apply_density_cut(segmentation_config.density_percentile)

        # Apply edge filter to remove points near survey boundaries
        print("TODO EDGE FILTER!")

        ridges = ridge_point_catalog.dec_ra_in_radians()

        # The initial segmentation is very quick so we only do it on
        # rank 0 and then broadcast the results to the other ranks for
        # the optional spline interpolation step.
        print("Building MST")
        mst = build_mst(ridges, k=segmentation_config.mst_neighbours)
        print("Splitting MST into segments")
        branch_points = detect_branch_points(mst)
        filament_segments = split_mst_at_branches(mst, branch_points)
        # filament_segments is a list of graphs.
        print("Clustering segments with DBSCAN")
        filament_labels = segment_filaments_with_dbscan(ridges, filament_segments)
        #filament labels is now a list of indices.

        final_nridge = sum(len(seg) for seg in filament_labels)
        ra_out = np.zeros(final_nridge)
        dec_out = np.zeros(final_nridge)
        ridge_id_out = np.zeros(final_nridge, dtype=int)
        i = 0
        for label, index in enumerate(filament_labels):
            ra_chunk = np.degrees(ridges[index, 1])
            dec_chunk = np.degrees(ridges[index, 0])
            chunk_n = len(index)
            ra_out[i : i + chunk_n] = ra_chunk
            dec_out[i : i + chunk_n] = dec_chunk
            ridge_id_out[i : i + chunk_n] = label
            i += chunk_n

    if size > 1 and segmentation_config.do_spline:
        # Broadcast the results to all ranks for the optional spline interpolation step.
        if rank == 0:
            print(f"Broadcasting {len(ra_out)} ridge points to {size} ranks for interpolation")
            n_filament =  len(filament_labels)
        else:
            print("Waiting to receive ridge points for interpolation")
        ra_out = comm.bcast(ra_out if rank == 0 else None, root=0)
        dec_out = comm.bcast(dec_out if rank == 0 else None, root=0)
        ridge_id_out = comm.bcast(ridge_id_out if rank == 0 else None, root=0)
        n_filament = comm.bcast(n_filament if rank == 0 else None, root=0)
    else:
        n_filament =  len(filament_labels)


    # Now create the catalog object that is going into interpolation
    # or to be saved otherwise.
    output = RidgeSegmentCatalog(segmentation_config.ridge_file)
    output.metadata["n_filaments"] = n_filament
    output.set_column("ra", ra_out)
    output.set_column("dec", dec_out)
    output.set_column("ridge_id", ridge_id_out)

    if segmentation_config.do_spline:
        if rank == 0:
            print(f"Performing spline interpolation on the segmented ridges with n_points = {segmentation_config.n_spline_points} on {size} processes")
        output = interpolate_segments(output, n_points=segmentation_config.n_spline_points, comm=comm)

    if rank == 0:
        output.save()
        return output


def measure_ridge_shear(shear_config: ShearConfig, comm=None):
    """
    Measure the shear around ridge segments and save the results to disk.

    This loads the two catalogs and then calls the measure_shear function
    with the configuration specified in shear_config. The measure_shear function does the
    actual work.

    Parameters
    ----------
    shear_config : ShearConfig
        Configuration for the shear measurement, including file paths and measurement parameters.

    """

    ridge_segments = RidgeSegmentCatalog(shear_config.ridge_file)
    ridge_segments.load()

    source_catalog = SourceCatalog(shear_config.source_catalog_file)
    source_catalog.load(comm=comm, split_over_ranks=True)

    shear_table = measure_shear(
        ridge_segments,
        source_catalog,
        shear_config.output_shear_file,
        num_bins=shear_config.num_bins,
        comm=comm,
        flip_g1=shear_config.flip_g1,
        flip_g2=shear_config.flip_g2,
        nside_coverage=shear_config.nside_coverage,
        min_distance_arcmin=shear_config.min_distance_arcmin,
        max_distance_arcmin=shear_config.max_distance_arcmin,
        skip_end_points=shear_config.skip_end_points,
        min_filament_points=shear_config.min_filament_points,
    )

    if comm is None or comm.rank == 0:
        shear_table.save()
    return shear_table
