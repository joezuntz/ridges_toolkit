import numpy as np
import healpy
import sys
import networkx as nx
from .io import RidgeSegmentCatalog
from collections import defaultdict
from sklearn.cluster import DBSCAN
from scipy.spatial import KDTree

from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import coo_matrix
import tqdm


def build_mst(points, k=10):
    """Constructs a Minimum Spanning Tree (MST) from given points."""
    tree = KDTree(points)
    distances, indices = tree.query(points, k=k + 1)

    row, col, data = [], [], []
    for i in range(len(points)):
        for j in range(1, k + 1):
            row.append(i)
            col.append(indices[i, j])
            data.append(distances[i, j])

    sparse_dist_matrix = coo_matrix((data, (row, col)), shape=(len(points), len(points)))
    mst_sparse = minimum_spanning_tree(sparse_dist_matrix).tocoo()

    G = nx.Graph()
    for i, j, weight in zip(mst_sparse.row, mst_sparse.col, mst_sparse.data):
        G.add_edge(int(i), int(j), weight=weight)

    return G


def detect_branch_points(mst):
    """Find nodes with degree > 2 (branch points)."""
    return [node for node, degree in dict(mst.degree()).items() if degree > 2]


def split_mst_at_branches(mst, branch_points):
    """Splits the MST into ordered paths after removing branch points.

    Since branch points are removed, each remaining connected component is a
    simple path (max degree 2).  The nodes in each component are returned in
    traversal order starting from one of the path endpoints.
    """
    G = mst.copy()
    G.remove_nodes_from(branch_points)

    ordered_segments = []
    for component in nx.connected_components(G):
        subgraph = G.subgraph(component)
        # Endpoints have degree <= 1; start traversal from one of them so the
        # result is a properly ordered path rather than an arbitrary DFS tree.
        endpoints = [n for n in subgraph.nodes() if subgraph.degree(n) <= 1]
        start = endpoints[0] if endpoints else next(iter(component))
        path = list(nx.dfs_preorder_nodes(subgraph, source=start))
        ordered_segments.append(path)

    return ordered_segments


def segment_filaments_with_dbscan(points, filament_segments, eps=0.02, min_samples=5):
    """Clusters MST segments using DBSCAN, returning ordered index arrays.

    Each entry in the returned list is an ordered NumPy array of point indices
    belonging to one filament, in the path order established by the MST.
    DBSCAN is used only for noise removal; noise points (label -1) are dropped.
    If a segment contains multiple DBSCAN clusters they are each returned as
    a separate ordered array.
    """
    filaments = []

    for segment in filament_segments:
        segment_points = np.array([points[idx] for idx in segment])
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        segment_labels = dbscan.fit_predict(segment_points)

        # Collect indices per cluster label, preserving MST traversal order.
        clusters = defaultdict(list)
        for idx, lbl in zip(segment, segment_labels):
            if lbl != -1:
                clusters[lbl].append(idx)
        filaments.extend(np.asarray(cluster, dtype=np.int32) for cluster in clusters.values())

    return filaments






def interpolate_segments(ridge_segments: RidgeSegmentCatalog, n_points: int=100, comm=None, output_name=None) -> RidgeSegmentCatalog:
    from .sphere_spline import SphericalSpline

    n_filaments = ridge_segments.metadata["n_filaments"]
    ra_out = np.zeros(n_points * n_filaments)
    dec_out = np.zeros(n_points * n_filaments)
    ridge_id_out = np.zeros(n_points * n_filaments, dtype=int)
    rank = comm.rank if comm is not None else 0
    size = comm.size if comm is not None else 1

    i_done = 0
    for filament_index, filament_ra, filament_dec in ridge_segments.iterate_ridges(radians=False):
        if filament_index % size != rank:
            continue

        if rank == 0:
            if i_done % 100 == 0:
                print(f"Rank 0 done {i_done} of about {n_filaments//size} filaments")
            i_done +=1
        # Process the chunk as needed, e.g., store it in an output array
        vec = healpy.ang2vec(filament_ra, filament_dec, lonlat=True)
        spl = SphericalSpline(vec)
        pts = spl(np.linspace(0.01, 0.99, n_points))
        ra_spline, dec_spline = healpy.vec2ang(pts, lonlat=True)
        # Save or process pts as needed
        slc = slice(filament_index * n_points, (filament_index + 1) * n_points)
        ra_out[slc] = ra_spline
        dec_out[slc] = dec_spline
        ridge_id_out[slc] = filament_index
    
    # sum all together because everything else will be zeros.
    # this is lazy. We could also gather all the arrays properly
    # but it would be fiddly and this is fine for our typical array size.
    if comm is not None:
        ra_out = comm.reduce(ra_out)
        dec_out = comm.reduce(dec_out)
        ridge_id_out = comm.reduce(ridge_id_out)

    if rank == 0:
        # By default we use the same filename for the output.
        # Can supply a different one if you want to keep both the original and the spline version.
        output_name = ridge_segments.filename if output_name is None else output_name
        output = RidgeSegmentCatalog(output_name)
        output.metadata["n_filaments"] = n_filaments
        output.set_column("ra", ra_out)
        output.set_column("dec", dec_out)
        output.set_column("ridge_id", ridge_id_out)
    else:
        output = None

    return output

