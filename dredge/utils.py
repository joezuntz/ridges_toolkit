import numba
import numpy as np


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


