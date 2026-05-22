import numpy as np
import os
import pickle
from sklearn.neighbors import BallTree

import numpy as np
import healpy as hp
import numba
import tqdm

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
            "theta": self.theta,
            "phi": self.phi,
            "pix": self.pix,
            "pixels_to_indices": dict(self.pixels_to_indices),  # Convert to regular dict for pickling
            "max_npix": self.max_npix
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
        Find the indices of the points in the tree that are within a given radius of the input points. The radius is in radians.
        """
        nearby_pixels = hp.query_disc(self.nside, hp.ang2vec(theta, phi), radius, inclusive=True)
        max_size = self.max_npix * len(nearby_pixels)
        rval = _query_core(theta, phi, self.theta, self.phi, radius, nearby_pixels, self.pixels_to_indices, max_size)
        return rval

@numba.njit
def _query_core(theta_query, phi_query, theta_bg, phi_bg, radius, nearby_pixels, pixels_to_indices, max_size):
    output = np.empty(max_size, dtype=np.int64)
    output_idx = 0
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

@numba.njit
def haversine_distance(theta1, phi1, theta2, phi2):
    """
    Compute the haversine distance between two points on the sphere.
    """
    dlon = phi2 - phi1
    dlat = theta2 - theta1
    a = np.sin(dlat/2)**2 + np.cos(theta1) * np.cos(theta2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return c



def make_tree(coordinates, tree_nside):
    """
    Creates or loads a spatial tree structure based on the given coordinates.
    This function either builds a new tree from the provided coordinates or
    loads an existing tree from a file. Optionally, it can save the newly
    created tree to a file for future use.

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


def query_tree(tree: BallTree, points, radius):
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
    indices : numpy.ndarray
        An array of indices of the nearest neighbors for each input point.

    distances : numpy.ndarray
        An array of distances to the nearest neighbors for
        each input point.
    """
    output = []
    output_distances = []
    npoint = len(points)
    counts = np.zeros(npoint, dtype=np.int64)
    for i, (theta, phi) in enumerate(points):
        theta = np.pi/2 - theta
        indices, distances = tree.query_radius(theta, phi, radius)
        counts[i] = len(indices)
        output.append(indices)
        output_distances.append(distances)
    
    max_output = np.max(counts)
    packed_output = np.zeros((npoint, max_output), dtype=np.int64)
    packed_distances = np.zeros((npoint, max_output), dtype=np.float64)
    for i, (indices, distances) in enumerate(zip(output, output_distances)):
        packed_output[i, :len(indices)] = indices
        packed_distances[i, :len(distances)] = distances

    return packed_output, counts, packed_distances


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
    tree : scipy.spatial.cKDTree
        A KD-tree object used to efficiently query the nearest neighbors of points in `ridges`.
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
    pbar = tqdm.tqdm(total=len(ridges), desc="Filtering points with tree")
    keep = np.zeros(len(ridges), dtype=bool)
    for i, (theta, phi) in enumerate(ridges):
        theta = np.pi/2 - theta
        indices, _ = tree.query_radius(theta, phi, threshold * bandwidth)
        keep[i] = len(indices) > 0
        pbar.update(1)
    return ridges[keep]
