from .utils import haversine_distance
import numpy as np
import healpy as hp
import numba

# think about jax-healpy for this!
class HealpixTree:
    """
    A tree-like structure that just groups points by 
    their healpix pixel. This is not a real tree, but it allows us to 
    quickly find all points within a given radius by looking at the neighboring pixels.
    """
    def __init__(self, theta=None, phi=None, nside=None):
        # Writing the init method like this so that the
        # tree can be initialized without data and then have data set later
        # allows us to pickle/unpickle the object so it can be sent with MPI
        self.nside = nside
        if theta is not None and phi is not None:
            self.set_data(theta, phi)

    def set_data(self, theta, phi):
        nside = self.nside
        self.nside = nside
        self.theta = theta
        self.phi = phi
        self.pix = hp.ang2pix(nside, theta, phi)
        hit_pix = np.unique(self.pix)
        pixels_to_indices = {}
        for pix in hit_pix:
            pixels_to_indices[pix] = []
    
        for i, pix in enumerate(self.pix):
            pixels_to_indices[pix].append(i)

        # convert into a NUMBA dictionary
        self.pixels_to_indices = numba.typed.Dict.empty(
            key_type=numba.int64,
            value_type=numba.types.Array(numba.int64, 1, "C")
        )
        self.max_npix = 0

        for pix in hit_pix:
            p = np.array(pixels_to_indices[pix])
            self.pixels_to_indices[pix] = p
            if len(p) > self.max_npix:
                self.max_npix = len(p)
    
    def __getstate__(self):
        state = {
            "nside": self.nside,
            "theta": getattr(self, "theta", None),
            "phi": getattr(self, "phi", None),
            "pix": getattr(self, "pix", None),
            "pixels_to_indices": dict(getattr(self, "pixels_to_indices", {})),
            "max_npix": getattr(self, "max_npix", 0),
        }
        return state
    
    def __setstate__(self, state):
        self.nside = state["nside"]
        self.theta = state["theta"]
        self.phi = state["phi"]
        self.pix = state["pix"]
        # Convert back to numba typed dict
        self.pixels_to_indices = numba.typed.Dict.empty(
            key_type=numba.int64,
            value_type=numba.types.Array(numba.int64, 1, "C")
        )
        for key, value in state["pixels_to_indices"].items():
            self.pixels_to_indices[key] = np.array(value)
        self.max_npix = state["max_npix"]

    def query_radius(self, theta, phi, radius):
        """
        Find the indices of the points in the tree that are within a given radius of the input points. 
        The radius is in radians.

        Parameters
        ----------
        theta: float
            Co-latitude of the point to query in radians
        phi: float
            Longitude of the point to query in radians
        radius:
            Distance in radians of points to query

        Returns
        -------
        indices: array, 1D, integer
            The indices of points in self.theta and self.phi that are within the
            distance radius of the query point
        distances: array, 1D, float
            The haversince distance in radians to each point
        """
        nearby_pixels = hp.query_disc(self.nside, hp.ang2vec(theta, phi), radius, inclusive=True)
        max_size = self.max_npix * len(nearby_pixels)
        indices, distances = _query_core(theta, phi, self.theta, self.phi, radius, nearby_pixels, self.pixels_to_indices, max_size)
        return indices, distances

@numba.njit
def _query_core(theta_query, phi_query, theta_bg, phi_bg, radius, nearby_pixels, pixels_to_indices, max_size):
    """
    This internal function finds all points closer than radius
    to the query point.

    Here the query points are assumed to be scalar and the backgrounds to be arrays.
    """
    output = np.empty(max_size, dtype=np.int64)
    output_idx = 0
    # The pixels_to_indices dict maps healpix pixel indices to the indices
    # in the background coordinates of all the objects in that pixel.
    # If any pixels are not in pixels_to_indices then that means that there are
    # no objects in that pixel and it can safely be ignored.
    for p in nearby_pixels:
        if p in pixels_to_indices:
            for idx in pixels_to_indices[p]:
                output[output_idx] = idx
                output_idx += 1
    if output_idx == 0:
        output = np.zeros(0, dtype=np.int64)
        distance = np.zeros(0, dtype=np.float64)
        return output, distance
    output = output[:output_idx]
    # get the great circle distance betwen the input point and output points
    distance = haversine_distance(theta_query, phi_query, theta_bg[output], phi_bg[output])
    output = output[distance < radius]
    # return the indices of the points within the radius
    return output, distance[distance < radius]


def make_tree(coordinates, tree_nside):
    """
    Creates or loads a spatial tree structure based on the given coordinates.
    This function builds a new tree from the provided coordinates. This is now fast
    enough that it doesn't seem worth saving the tree to disc, though we can add that
    easily enough if we want.

    Parameters
    -----------
    coordinates : array-like
        A 2D array-like structure containing the coordinates for which the tree is to be created.

    tree_nside : int
        The nside parameter for the healpix tree, which determines the resolution of the tree.
    Returns
    --------
    tree:  HealpixTree
        Not actually a tree, just a lookup structure.
    """

    print("Building tree from coordinates, with nside = ", tree_nside)
    tree = HealpixTree(np.pi/2 - coordinates[:, 0], coordinates[:, 1], tree_nside)
    return tree


def query_tree(tree: HealpixTree, points, radius):
    """
    Query a HealpixTree to find the points within
    a specified radius of the input points.

    Parameters:
    -----------
    tree : HealpixTree
        The object used for querying nearest neighbors.

    points : array-like
        An array of points for which the nearest neighbors
        are to be found. Each point should have the same dimensionality
        as the KD Tree. Shape (npoint, ndim)

    radius : float
        The radius within which to find neighbors for each point.
        In radians.

    Returns:
    --------
    packed_indices : numpy.ndarray
        An array of indices of the nearest neighbors for each input point.
        The shape is (npoint, max_count). Only the start of each array is
        valid.

    counts : numpy.ndarry
        The number of valid indices for each point.

    packed_distances : numpy.ndarray
        An array of distances to the nearest neighbors for
        each input point. Also of shape (npoint, max_count)
    """
    output = []
    output_distances = []
    npoint = len(points)
    # For each query point we ask the tree for nearby points,
    # and then record the complete list
    counts = np.zeros(npoint, dtype=np.int64)
    for i, (theta, phi) in enumerate(points):
        theta = np.pi/2 - theta
        indices, distances = tree.query_radius(theta, phi, radius)
        counts[i] = len(indices)
        output.append(indices)
        output_distances.append(distances)
    
    # Pack the indices and distances into a rectangular array,
    # only part of which is referenced. The valid range is indicated
    # by the counts variable.
    max_output = np.max(counts)
    packed_indices = np.zeros((npoint, max_output), dtype=np.int64)
    packed_distances = np.zeros((npoint, max_output), dtype=np.float64)
    for i, (indices, distances) in enumerate(zip(output, output_distances)):
        packed_indices[i, :len(indices)] = indices
        packed_distances[i, :len(distances)] = distances

    return packed_indices, counts, packed_distances


def cut_points_with_tree(ridges, tree, bandwidth, threshold=4):
    """
    Filters out points in the `ridges` array whose nearest neighbor, as determined
    by the provided `tree`, is farther than a specified threshold distance.
    The threshold distance is calculated as `threshold` times the angular distance
    corresponding to the given `bandwidth`.
    Parameters:
    -----------
    ridges : numpy.ndarray
        An array of points (e.g., coordinates) to be filtered, in [dec, ra]
    tree : HealpixTree
        A healpix tree-like object used to efficiently query the nearest neighbors of points in `ridges`.
    bandwidth : float
        The bandwidth value used to calculate the angular distance threshold.
    threshold : float, optional
        A multiplier for the bandwidth to determine the maximum allowable distance
        for a point to be considered valid. Default is 4.
    Returns:
    --------
    numpy.ndarray
        A filtered array of points from `ridges` that meet the distance criteria.

    """
    keep = np.zeros(len(ridges), dtype=bool)
    for i, (theta, phi) in enumerate(ridges):
        theta = np.pi/2 - theta
        indices, _ = tree.query_radius(theta, phi, threshold * bandwidth)
        keep[i] = len(indices) > 0
    return ridges[keep]
