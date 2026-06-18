import numba
import numpy as np


@numba.njit
def haversine_distance(theta1, phi1, theta2, phi2):
    """
    Compute the haversine distance between two points on the sphere.
    Careful!

    First, these are healpix coordinates so the theta is the co-latitude
    not the latitude

    Second, many source swap theta and phi around
    """
    dlon = phi2 - phi1
    dlat = theta1 - theta2 #switched because this is co-lat, but doesn't matter as the sin gets squared
    a = np.sin(dlat/2)**2 + np.sin(theta1) * np.sin(theta2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return c


