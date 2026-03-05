import numpy as np
import os
import pickle
from sklearn.neighbors import BallTree

def make_tree(coordinates, metric="haversine", tree_file=None):
    """
    Creates or loads a spatial tree structure based on the given coordinates.
    This function either builds a new tree from the provided coordinates or 
    loads an existing tree from a file. Optionally, it can save the newly 
    created tree to a file for future use.

    Parameters
    -----------
    coordinates : array-like
        A 2D array-like structure containing the coordinates for which the tree is to be created.

    metric : str, optional
        The distance metric to use for the tree. Default is "haversine".
        Other options include "euclidean" for standard Euclidean distance.
    
    tree_file : str, optional
        The file path where the tree should be saved or loaded from.

    Returns
    --------
    tree:  BallTree
        The spatial tree structure created or loaded.
    """

    
    load_tree = (tree_file is not None) and os.path.exists(tree_file)
    save_tree = (tree_file is not None) and not os.path.exists(tree_file)

    if load_tree:
        print(f"Loading tree from {tree_file}")
        with open (tree_file, 'rb') as f:
            tree = pickle.load(f)
    else:
        print("Building tree from coordinates")
        tree  = BallTree(coordinates, metric=metric)

    # Save the tree file if it does not exist
    if save_tree:
        with open(tree_file, 'wb') as f:
            print(f"Saving tree to {tree_file}")
            pickle.dump(tree, f)

    return tree


def query_tree(tree, points, n_neighbors):
    """
    Query a KD Tree to find the nearest neighbors for a set of points, 
    optionally using parallel processing. Despite the name, the n_process
    parameter is actually the number of threads to use, not processes.

    Parameters:
    -----------
    tree : scipy.spatial.KDTree or similar
        The KD Tree object used for querying nearest neighbors.

    points : array-like
        An array of points for which the nearest neighbors 
        are to be found. Each point should have the same dimensionality 
        as the KD Tree. Shape (npoint, ndim)

    n_neighbors : int
        The number of nearest neighbors to query for each point.

    Returns:
    --------
    indices : numpy.ndarray
        An array of indices of the nearest neighbors for each input point.

    distances : numpy.ndarray
        An array of distances to the nearest neighbors for 
        each input point.
    """
    distances, indices = tree.query(points, k=n_neighbors, return_distance=True)
    return indices, distances


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
    distances, _ = tree.query(ridges, k=1, return_distance=True)
    keep = distances[:, 0] < threshold * bandwidth
    return ridges[keep]
