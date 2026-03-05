import numpy as np
from statsmodels.nonparametric.kernel_density import KDEMultivariate
from statsmodels.nonparametric.kernel_density import EstimatorSettings


def estimate_bandwidth(coordinates, n_process):
    defaults = EstimatorSettings()
    defaults.n_jobs = n_process
    defaults.efficient = True

    # Generate the initial density estimate
    print("Building density estimator")
    density_estimate = KDEMultivariate(data=coordinates,
                                        var_type='cc',
                                        bw='cv_ml',
                                        defaults=defaults)
    return np.mean(density_estimate.bw)
