import numpy as np
import warnings
from .tree import query_tree
from numba import njit, prange
from numba.core.errors import NumbaPerformanceWarning

# Suppress Numba performance warnings
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)



def ridge_update(ridges, coordinates, bandwidth, tree, n_neighbors):
    all_nearby_indices, all_distances = query_tree(tree, ridges, n_neighbors)
    updates = ridge_update_inner(ridges, coordinates, bandwidth, all_nearby_indices, all_distances)
    return updates


@njit(parallel=True)
def ridge_update_inner(ridges, coordinates, bandwidth, all_nearby_indices, all_distances):
    # Create a list to store all update values
    update_sizes = np.zeros(ridges.shape[0])
    updates = np.zeros(ridges.shape)
    for i in prange(ridges.shape[0]):
        # Compute the update movements for each point
        # get all the points within the 3 sigma bandwidth
        nearby_indices = all_nearby_indices[i]
        distance = all_distances[i]
        nearby_coordinates = coordinates[nearby_indices].copy()
    
        updates[i] = update_function(ridges[i], nearby_coordinates, bandwidth, distance)

        # Store the change between updates to check convergence
        update_sizes[i] = np.sum(np.abs(updates[i]))
    ridges += updates
    return update_sizes




@njit
def update_function(point, coordinates, bandwidth, distance):
    """Calculate the mean shift update for a provided mesh point.
    
    This function calculates the mean shift update for a given point of 
    the mesh at the current iteration. This is done through a spectral
    decomposition of the local inverse covariance matrix, shifting the
    respective point closer towards the nearest estimated ridge. The
    updates are provided as a tuple in the latitude-longitude space to
    be added to the point's coordinate values.
    
    Parameters:
    -----------
    point : array-like
        The latitude-longitude coordinate tuple for a single mesh point.
        
    coordinates : array-like
        The set of latitudes and longitudes as a two-column array of floats.

    bandwidth : float
        The bandwidth used for the update.

    distance : array-like
        Pre-computed distances between the mesh point and the dataset.
        
    Returns:
    --------
    point_updates : float
        The tuple of latitude and longitude updates for the mesh point.
        
    Attributes:
    -----------
    None
    """
    squared_distance = distance ** 2
    # evaluate the kernel at each distance
    weights = gaussian_kernel(squared_distance, bandwidth)
    # now reweight each point
    shift = coordinates.T @ weights / np.sum(weights)
    # first, we evaluate the mean shift update
    update = shift - point
    # Calculate the local inverse covariance for the decomposition
    inverse_covariance = local_inv_cov(point, coordinates, bandwidth)
    # Compute the eigendecomposition of the local inverse covariance
    eigen_values, eigen_vectors = np.linalg.eig(inverse_covariance)
    # Align the eigenvectors with the sorted eigenvalues
    sorted_eigen_values = np.argsort(eigen_values)
    eigen_vectors = eigen_vectors[:, sorted_eigen_values]
    # Cut the eigenvectors according to the sorted eigenvalues
    cut_eigen_vectors = eigen_vectors[:, 1:]
    # Project the update to the eigenvector-spanned orthogonal subspace
    point_updates = cut_eigen_vectors.dot(cut_eigen_vectors.T).dot(update)    
    # Return the projections as the point updates
    return point_updates

@njit
def gaussian_kernel(values,  bandwidth):
    """Calculate the Gaussian kernel evaluation of distance values.
    
    This function evaluates a Gaussian kernel for the squared distances
    between a mesh point and the dataset, and for a given bandwidth.
    
    Parameters:
    -----------
    values : array-like
        The squared distances between a mesh point and provided coordinates.
    
    bandwidth : float
        The bandwidth used for kernel density estimates of data points.
        
    Returns:
    --------
    kernel_value : float
        The Gaussian kernel evaluations for the given distances.
        
    Attributes:
    -----------
    None
    """
    # Compute the kernel value for the given values
    kernel_value = np.exp(-0.5 * values / bandwidth**2)
    return kernel_value

@njit
def mean1(a):
    """
    Calculate the mean of a 2D array along axis 1.

    This is needed because the Numba JIT compiler does not support
    the numpy.mean function with axis argument, so we implement it
    manually.
    
    """
    n1, n2 = a.shape
    res = np.zeros(n1)
    for i in range(n1):
        res[i] = np.sum(a[i, :]) / n2
    return res

@njit
def local_inv_cov(point, 
                  coordinates, 
                  bandwidth):
    """Compute the local inverse covariance from the gradient and Hessian.
    
    This function computes the local inverse covariance matrix for a given
    mesh point and the provided dataset, using a given bandwidth. In order
    to reach this result, the covariance matrix for the distances between
    a mesh point and the dataset is calculated. After that, the Hessian
    matrix is used to calculate the gradient at the given point's location.
    Finally, the latter is used to arrive at the local inverse covariance.
    
    Parameters:
    -----------
    point : array-like
        The latitude-longitude coordinate tuple for a single mesh point.
    
    coordinates : array-like
        The set of latitudes and longitudes as a two-column array of floats.
    
    bandwidth : float
        The bandwidth used for kernel density estimates of data points.
        
    Returns:
    --------
    inverse_covariance : array-like
        The local inverse covariance for the given point and coordinates.
        
    Attributes:
    -----------
    None
    """


    number_points, number_columns = coordinates.shape 

    # Calculate the squared distance between points
    squared_distance = np.sum((coordinates - point)**2, axis=1)
    # Compute the weight kernels called b_j in the paper
    weights = gaussian_kernel(squared_distance, bandwidth)
    weight_sum = np.sum(weights)
    weight_average = weight_sum / number_points

    # Compute the location differences between the point and the dataset
    mu = (coordinates - point) / bandwidth**2

    # Combine terms to get the Hessian matrix following the paper algorithm
    term1 = (weights * mu.T) @ mu / number_points
    term2 = weight_sum * np.eye(number_columns) / bandwidth**2 / number_points
    H = term1 - term2

    # This is an extra term that is not in the paper
    grad = -mean1(weights * mu.T)
    inv_cov = -H / weight_average + (grad @ grad) / weight_average**2
    return inv_cov





def mesh_generation(coordinates, mesh_size, seed=None):
    """Generate a set of uniformly-random distributed points as a mesh.
    
    The subspace-constrained mean shift algorithm operates on either a grid
    or a uniform-random set of coordinates to iteratively shift them towards
    the estimated density ridges. Due to the functionality of the code, the
    second approach is chosen, with a uniformly-random set of coordinates
    in the intervals covered by the provided dataset as a mesh. In order to
    not operate on a too-small or too-large number of mesh points, the size
    of the mesh is constrained to a lower limit of 50,000 and an upper limit
    of 100,000, with the size of the provided dataset being used if it falls
    within these limits. This is done to avoid overly long running times.
    
    Parameters:
    -----------
    coordinates : array-like
        The set of latitudes and longitudes as a two-column array of floats.
    
    mesh_size : int
        The number of mesh points to be used to generate ridges.
    
    seed: int, optional
        If provided, sets the random seed for reproducibility.
        
    Returns:
    --------
    mesh : array-like
        The set of uniform-random coordinates in the dataset's intervals.
        
    Attributes:
    -----------
    None
    """
    # Get the minimum and maximum for the latitudes
    min_latitude = np.min(coordinates[:, 0])
    max_latitude = np.max(coordinates[:, 0])

    # Get the minimum and maximum for the longitudes
    min_longitude = np.min(coordinates[:, 1])
    max_longitude = np.max(coordinates[:, 1])

    # Create an array of uniform-random points as a mesh
    rng = np.random.default_rng(seed)
    mesh_1 = rng.uniform(min_latitude, max_latitude, mesh_size)
    mesh_2 = rng.uniform(min_longitude, max_longitude, mesh_size)
    mesh = np.vstack((mesh_1.flatten(), mesh_2.flatten())).T

    # Return the uniform mesh for the coordinates.
    return mesh
