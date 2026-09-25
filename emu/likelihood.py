import numpy as np
import json
import yaml

import predict


SIGMA_E = 0.26
NPERM = 1000
FOLDER = 'fiducial-shear-noisy/fiducial/noisy-shear/'

PARAMETER_ORDER = [
    'Omega_m',
    'bary_Mc',
    'bary_nu',
    'sigma8',
    'w0',
    'ns',
    'Omega_b',
    'H0',
]

COSMOGRID_RANGES = {
    'Omega_m': {'lower_wide': 0.1,  'upper_wide': 0.5,   'lower_narrow': 0.15,  'upper_narrow': 0.45},
    'sigma8':  {'lower_wide': 0.4,  'upper_wide': 1.4,   'lower_narrow': 0.5,   'upper_narrow': 1.3},
    'w0':      {'lower_wide': -2.,  'upper_wide': -0.33, 'lower_narrow': -1.25, 'upper_narrow': -0.75},
    'ns':      {'lower_wide': 0.87, 'upper_wide': 1.07,  'lower_narrow': 0.93,  'upper_narrow': 1.},
    'Omega_b': {'lower_wide': 0.03, 'upper_wide': 0.06,  'lower_narrow': 0.04,  'upper_narrow': 0.05},
    'H0':      {'lower_wide': 64.,  'upper_wide': 82.,   'lower_narrow': 65.,   'upper_narrow': 75.},
    # NOTE: I made up the the following two parameter ranges!!! Look for the actual ones
    'bary_Mc': {'lower_wide': 12.,  'upper_wide': 15.,   'lower_narrow': 12.5,  'upper_narrow': 14.},
    'bary_nu': {'lower_wide': 0.,   'upper_wide': 1.,    'lower_narrow': 0.,    'upper_narrow': 0.5}
}


class Likelihood():
    """
    Class to handle the theoretical predictions and likelihood calculations.
    """

    def __init__(self, param_config_file='emu/data/config_data.yaml'):
        self.config_file = param_config_file
        self.bin_pairs = self.load_bin_pairs('emu/data/metadata.json')
        self.fiducials = self.load_fiducials(self.config_file)
        self.data_vector = self.compute_data_vector()
        self.variance = self.compute_diagonal_covariance()
        self.full_cov = self.compute_full_covariance()

    
    def load_fiducials(self, fids_file):
        """
        Load fiducials and priors from a configuration file (yaml)
        """  
        with open(fids_file, 'r') as f:
            params_dict = yaml.safe_load(f)

        # save fiducials
        params_fiducial = {par: params_dict[par]['fid'] for par in params_dict}

        # save fixed parameters
        self.fixed_params = {par: params_dict[par]['fid'] for par in params_dict 
                             if params_dict[par]['type']=='F'}
        return params_fiducial


    def load_bin_pairs(self, metadata_file):
        """
        Load the bin pairs from metadata.json file.
        """
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        return metadata['bin_pairs']


    def read_data(self, loc, nperm, bin_lens, bin_source):
        file = loc + f'perm_{nperm}_shear_lens{bin_lens}_source{bin_source}.txt'
        # print(file)
        sep_bin_center, weighted_sep, g_plus, g_cross, counts, weights = np.genfromtxt(file, unpack=True)
        return sep_bin_center, g_plus, g_cross, counts


    def compute_data_vector(self):
        for i, (l, s) in enumerate(self.bin_pairs):
            prediction = predict.make_prediction(self.fiducials, l, s)

            # append to data vector
            if i == 0:
                data_vector = prediction
            else:
                data_vector = np.concatenate((data_vector, prediction))

        # return flattened data vector
        return data_vector.flatten()


    def compute_diagonal_covariance(self):
        """
        Compute the covariance matrix for the data vector. For now, we assume 
        a diagonal covariance matrix. The diagonal elements are given by the 
        shot noise for each bin pair.
        """
        variance = []
        for i, (l, s) in enumerate(self.bin_pairs):
            # NOTE: at the moment reads simulations 0 shot noise
            shot_noise = predict.get_shot_noise(sim_idx=0,
                                                sigma_e=SIGMA_E,
                                                counts_file='emu/data/counts.txt')[i]
            variance.append(shot_noise.flatten())

        return np.concatenate(variance)


    def compute_full_covariance(self):
        signal = np.zeros((NPERM, len(self.bin_pairs), 20))
        shot_noise = np.zeros((len(self.bin_pairs), 20))

        for i, (l, s) in enumerate(self.bin_pairs):
            for p in range(NPERM):
                # make per number foru digits (eg. 1 --> 0001)
                p_str = str(p).zfill(4)
                sep_bin_center, g_plus, g_cross, counts = self.read_data(FOLDER, p_str, l, s)
                signal[p, i, :] = g_plus
                shot_noise[i, :] = SIGMA_E / np.sqrt(counts) 

        # each bin pair has 20 theta points
        cov_size = len(sep_bin_center) * len(self.bin_pairs)
        cov = np.zeros((cov_size, cov_size))

        # flatten each permutation into one vector [(bin0,theta0), (bin0,theta1), ...]
        signal_flat = signal.reshape(NPERM, cov_size)

        # realisation covariance (diagonal and off diagonal)
        cov = np.cov(signal_flat, rowvar=False, ddof=1)

        # add shape noise variance only on diagonal
        shape_variance = shot_noise.flatten() ** 2
        cov[np.diag_indices_from(cov)] += shape_variance

        return cov

    
    def compute_flatten_prediction(self, params_dict):
        """
        Flattens the prediction from the emulator for given parameters.

        Parameters:
        ----------
        params_dict : dict
            Dictionary of cosmological parameters to use as emulator input

        Returns:
        -------
        Flattened prediction array
        """
        model_prediction = []

        for l, s in self.bin_pairs:
            prediction = predict.make_prediction(params_dict, l, s)
            model_prediction.append(prediction.flatten())

        return np.concatenate(model_prediction)
    

    def check_param_ranges(self, params_dict):
        """
        Check if the parameters are within the valid ranges for the emulators
        at each step of MCMC.
        
        Parameters:
        ----------
        params_dict: dict
            Dictionary of CosmoGrid parameters
            
        Returns:
        -------
        status : boolean
            True if all parameters are within valid ranges, False otherwise.
        """
        for par in PARAMETER_ORDER:

            value = params_dict[par]

            lower = COSMOGRID_RANGES[par]['lower_narrow']
            upper = COSMOGRID_RANGES[par]['upper_narrow']

            if not lower <= value <= upper:
                return False

        return True


    def compute_diag_likelihood(self, params_dict):
        """
        Calculates the loglikelihood using a diagonal covariance (chi2).

        Parameters:
        ----------
        params_dict : dict
            Dictionary of cosmological parameters to use as emulator input

        Returns:
        -------
        Loglikelihood (-0.5*chi2) or -inf if the parameters are outside the emulator range
        """
        are_params_within_ranges = self.check_param_ranges(params_dict)
        
        if are_params_within_ranges:
            model = self.compute_flatten_prediction(params_dict)

            residuals = self.data_vector - model 
            chi2 = np.sum((residuals**2) / self.variance)

            return -0.5*chi2 
        else:
            return -np.inf


    def compute_fullcov_likelihood(self, params_dict):
        are_params_within_ranges = self.check_param_ranges(params_dict)

        if are_params_within_ranges:
            model = self.compute_flatten_prediction(params_dict)

            residuals = self.data_vector - model 
            inv_cov = np.linalg.inv(self.full_cov)

            # Anderson-Hartlapp correction
            bias_factor = NPERM / NPERM - inv_cov.shape[0] - 1
            correct_inv_cov = bias_factor * inv_cov
            
            chi2 = residuals @ correct_inv_cov @ residuals

            return -0.5*chi2 
        else:
            return -np.inf
        

    def test_one_likelihood_iteration(self, params_dict):
        """
        Test the likelihood calculation for a single set of parameters.

        Parameters:
        ----------
        params_dict : dict
            Dictionary of cosmological parameters to use as emulator input

        Returns:
        -------
        Loglikelihood (-0.5*chi2) or -inf if the parameters are outside the emulator range
        """
        loglike = self.compute_diag_likelihood(params_dict)
        print(f"logL with full cov: {self.compute_fullcov_likelihood(params_dict)}")

        return loglike

