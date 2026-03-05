"""DREDGE-MOD: A cosmology-oriented modification of the DREDGE package

Introduction:
-------------

This is a modified version of the DREDGE package [1] for geospatial ridge
estimation, which is itself an extension of the subspace-constrained mean 
shift algortihm [2] previously applied to, an extended in, cosmology [3, 4].
Additional extensions for this modified version include the application of
a maximum likelihood cross-validation to optimize the bandwidth, as well
as the use of multiprocessing capabilities for parallelizable parts.

The original package is available on PyPI (https://pypi.org/project/dredge/)
and through its own GitHub repository (https://github.com/moews/dredge).

Quickstart:
-----------
The two-dimensional set of coordinates fed into DREDGE-MOD has to be
provided in the form of a NumPy array with two columns, with the latitudes
in the first and the longitudes in the second column, in radians. 

Additionally, five
optional parameters can be manually set by the user:

(1) The parameter 'n_neighbors' specifies the number of nearest neighbors
    that should be used to calculate the updates. The default is 5000,
    but you may be able to use a smaller number for speed.
    
(2) The parameter 'bandwidth' provides the bandwidth that is used for the 
    kernel density estimator and Gaussian kernel evaluations

(3) The parameter 'convergence' specifies the convergence threshold to
    determine when to stop iterations and return the density ridge points.
    If the resulting density ridges don't follow clearly visible lines,
    this parameter can be set to a lower value. The default is 1e-5 degrees,
    but the value should be set in radians.

(4) The parameter 'percentage' should be set if only density ridge points
    from high-density regions, as per a kernel density estimate of the
    provided set of coordinates, are to be returned. If, fore example, the
    parameter is set to '5', the density ridge points are evaluated via
    the kernel density estimator, and only those above the 95th percentile,
    as opposed to all of them as the default, are returned to the user.
    
(5) The parameter 'distance' can be set if a project is not dealing with 
    latitude-longitude datasets. It can can be set to either 'euclidean' or
    'haversine', with the latter being the default value.
    
(6) The parameter 'n_process' can be set to enable multiprocessing on more
    than one core in order to speed up computation times. The default is
    zero, indicating no multiprocessing, and different positive integers
    can be set to specify the number of cores that should be used.

(7) The parameter 'mesh_size' can be set to specify the number of points
    that should be used to generate the initial mesh from which the ridges
    are formed over the course of multiple iterations. By default, this is
    set to a number that enables reasonably fast computing times, but larger
    numbers of mesh points allow for more complete ridges.

A simple example for using DREDGE-MOD looks like this:

    --------------------------------------------------------------
    |  from dredge-mod import filaments                          |
    |                                                            |
    |  ridges = filaments(coordinates = your_point_coordinates)  |
    |                                                            |
    --------------------------------------------------------------

Authors:
--------

Ben Moews
Institute for Astronomy (IfA)
School of Physics & Astronomy
The University of Edinburgh

Andy Lawler
Dept. of Statistical Science
College of Arts and Sciences
Baylor University

Morgan A. Schmitz
CosmoStat lab
Astrophysics Dept.
CEA Paris-Saclay

Updates & Parallelism by Joe Zuntz
University of Edinburgh


References:
-----------
[1] Moews, B. et al. (2019): "Filaments of crime: Informing policing via
    thresholded ridge estimation", JQC (under review), arXiv:1907.03206
[2] Ozertem, U. and Erdogmus, D. (2011): "Locally defined principal curves 
    and surfaces", JMLR, Vol. 12, pp. 1249-1286
[3] Chen, Y. C. et al. (2015), "Cosmic web reconstruction through density 
    ridges: Method and algorithm", MNRAS, Vol. 454, pp. 1140-1156
[4] Chen, Y. C. et al. (2016), "Cosmic web reconstruction through density 
    ridges: Catalogue", MNRAS, Vol. 461, pp. 3896-3909
    
Packages and versions:
----------------------
The versions listed below were used in the development of DREDGE-MOD, but the 
exact version numbers aren't specifically required. The installation process 
via PyPI will take care of installing or updating every library to at least the
level that fulfills the requirement of providing the necessary functionality.

Python 3.4.5
NumPy 1.11.3
SciPy 0.18.1
Scikit-learn 0.19.1
statsmodels 0.10.1
"""

from .bandwidth import estimate_bandwidth
from .main import find_filaments