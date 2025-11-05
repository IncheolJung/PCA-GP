import numpy as np
import sys

from numpy.random import default_rng
from numpy.linalg import svd

from scipy.signal import find_peaks

from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, PowerTransformer
)
from sklearn.pipeline import Pipeline

from typing import Iterable

from itertools import product
from acquisitionfunctions import *


# -----------------------
# For X normalizer: pass
# -----------------------
class dummy_preprocesser:
    def __init__(self):
        return None
    def fit(self, data):
        return self
    def transform(self, data):
        return np.array(data)
    def inverse_transform(self, data):
        return np.array(data)


# -----------------------
# Logging purpose
# -----------------------
class Tee(object):
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self
    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
    def flush(self):
        self.file.flush()
        self.stdout.flush()


# -----------------------
# Reduced Basis GP class
# -----------------------
class ReducedBasisGPBASE:
    def __init__(
        self, solver, trainer, angles, n_init=6, r=3, adaptive_r=True, 
        acquisition_type=0, Xnormalizer_type=0, terms=1, 
        normalizeY=True, verbose=True, usempi=False, mpicomm=None
    ) -> None:
        self.solver = solver
        self.train_gp = trainer
        self.angles = angles
        self.n_init = n_init
        self.r = r
        self.adaptive_r = adaptive_r
        self.acquisition_type = acquisition_type
        self.acquisition_function = None
        self.Xnormalizer_type = Xnormalizer_type
        self.normalizerX = None
        self.sampler = None
        self.terms = terms
        self.normalizeY = normalizeY
        self.verbose = verbose
        self.freqs = []
        self.responses = []
        self.U = None
        self.S = None
        self.Vh = None
        self.coeffs = None
        self.gps_real = []
        self.gps_imag = []
        self.fbest = None
        self.usempi = usempi
        if usempi:
            self.comm = mpicomm
            self.rank = mpicomm.Get_rank()
            self.size = mpicomm.Get_size()
        return None
        
    def initialize(self, f_min, f_max, sampling_strategy=1):
    
        if (not self.usempi) or (self.usempi and self.rank==0):
            if sampling_strategy == 0:
                self.sampler = lambda f_min, f_max, n_grid: \
                    np.linspace(f_min, f_max, n_grid)[:, None]
            elif sampling_strategy == 1:
                self.sampler = lambda f_min, f_max, n_grid: \
                    self.latin_hypercube_sampling(f_min, f_max, n_grid)[:, None]
                # grid = self.normalizerX.transform(lhc_samples)
            else:
                print(f"sampling_strategy {sampling_strategy} not supported!")
                print(f"falling back to sampling_strategy 0 (grid sampling)")
                return self.initialize(f_min, f_max, 0)
            
            # f_init = self.latin_hypercube_sampling(f_min, f_max, self.n_init)
            f_init = self.sampler(f_min, f_max, self.n_init).squeeze(-1)
            self.freqs = list(f_init)
            angles_init = self.angles
        else:
            f_init, angles_init = None, None

        if self.usempi:
            f_init = self.comm.bcast(f_init, root=0)
            angles_init = self.comm.bcast(angles_init, root=0)
        Y = self.solver(f_init, angles_init)  # (n_init, n_angles)

        if (not self.usempi) or (self.usempi and self.rank==0):
            # Y = np.array([self.solver(f, a) for f, a in product(f_init, self.angles)])  # (n_init, n_angles)
            # self.fbest = np.mean(Y)
            Y = Y.reshape(self.n_init, len(self.angles))  # (n_init, n_angles)
            self.responses = list(Y) # store each freq response (complex vector)

            # Data normalizer
            X = np.array(self.freqs)[:, None]
            self.normalizerX = self.get_data_normalizer(X)

            # Acquisition functions
            if (self.acquisition_type==0):
                self.acquisition_function = lambda _mu, _var: max_variance(_mu, _var)
            elif (self.acquisition_type==1):
                self.acquisition_function = lambda _mu, _var: expected_improvement(_mu, _var, np.array([self.fbest]))
            elif (self.acquisition_type==2):
                self.acquisition_function = lambda _mu, _var: upper_confidence_bound(_mu, _var)
            else: RuntimeError(f"INVALID ACQUISITION TYPE: {self.acquisition_type} must be < 3")

            self._update_basis()
            self._fit_gps()
        return 0
    
    def get_data_normalizer(self, X):
        if (self.Xnormalizer_type==0):      # do nothing
            normalizerX = dummy_preprocesser().fit(X)
        elif (self.Xnormalizer_type==1):    # z-score
            normalizerX = StandardScaler().fit(X)
        elif (self.Xnormalizer_type==2):    # min-max
            normalizerX = MinMaxScaler().fit(X)
        elif (self.Xnormalizer_type==3):    # power-transform
            normalizerX = PowerTransformer().fit(X)
        elif (self.Xnormalizer_type==4):    # standardizer + power-transform
            normalizerX = Pipeline([("std", StandardScaler()), ("pwr", PowerTransformer())]).fit(X)
        elif (self.Xnormalizer_type==5):    # standardizer + min-max
            normalizerX = Pipeline([
                ("std", StandardScaler()), 
                ("scale", MinMaxScaler(feature_range=(0, 100)))
                ]).fit(X)
        else: RuntimeError(f"INVALID ACQUISITION TYPE: {self.Xnormalizer_type} must be < 6")
        return normalizerX
    
    def latin_hypercube_sampling(
        self, val_min, val_max, n
    ) -> np.ndarray:
        step = float((val_max - val_min) / n)
        thres = [val_min + i*step for i in range(n)]
        thres.append(val_max)
        random = default_rng()
        return np.array([random.uniform(thres[i], thres[i+1]) for i in range(n)])
        
    def _update_basis(self):
        # Stack responses into matrix: shape (n_freqs, n_angles)
        Ymat = np.vstack([resp[None, :] for resp in self.responses])
        # Do SVD along angle axis
        U, S, Vh = svd(Ymat, full_matrices=False)
        if self.adaptive_r:
            # threshold
            eps = 1e-4
            e = 1
            r = 0
            for sr in S[1:]:
                r += 1
                e = (sr/S[0])**2
                if (e<(eps**2)): break
            r += 1
            if self.verbose:
                print(f"number of basis updated: {self.r} -> {r}")
                print("S[0, r-1] = ", S[0], S[r-1])
                if (r)<len(S): print("S[r] =", S[r])
            self.r = r
        # Keep first r modes
        self.U, self.S, self.Vh = U[:, :self.r], S[:self.r], Vh[:self.r, :]
        # Compute coefficients (project responses on modes)
        self.coeffs = self.U * self.S  # shape (n_freqs, r)
        # print(self.coeffs.shape, self.Vh.shape)
        return 0
    
    def acquisition_next_frequency(self, f_min, f_max, n_grid=101, n_new_samples=1):
        """Pick frequency that maximizes integrated variance across coefficients"""
        if (not self.usempi) or (self.usempi and self.rank==0):

            # n_grid = max(n_grid, len(self.freqs)+n_new_samples)
            total_mu, total_var, f_domain, POD_energy = self._pred_gps(f_min, f_max, n_grid)
            # print(np.array(preds))
            
            responses_pred = self.reconstruct(self.freqs)   # [freq, angle]
            loss_per_freq = np.mean(np.square(self.responses-responses_pred), axis=1)
            self.fbest = total_mu[np.argmin(loss_per_freq)]

            # Compute acquisition values
            ac_vals = self.acquisition_function(total_mu, total_var)

            # Convert frequencies back to original scale
            f_domain = self.normalizerX.inverse_transform(f_domain)

            # --- Find local maxima (peaks) ---
            peaks, properties = find_peaks(ac_vals, prominence=1e-8)  # adjust prominence if needed

            # If fewer peaks than requested, fallback to global top-n values
            if len(peaks) < n_new_samples:  # global top-n values
                sorted_idx = np.argsort(ac_vals)[::-1]
                mask = ~np.isin(f_domain[sorted_idx, 0], self.freqs)
                chosen_idx = sorted_idx[mask][:n_new_samples]
            else:  # top-n peaks
                # Sort peaks by prominence (descending)
                prom_sorted_idx = np.argsort(properties["prominences"])[::-1]
                top_peaks = peaks[prom_sorted_idx][:n_new_samples]

                # Remove already-sampled frequencies
                mask = ~np.isin(f_domain[top_peaks, 0], self.freqs)
                chosen_idx = top_peaks[mask][:n_new_samples]

            # --- Prepare outputs ---
            new_freq_out = f_domain[chosen_idx, 0].tolist()
            ac_vals_out = ac_vals[chosen_idx].tolist()

            # denom = self.r

            return new_freq_out, ac_vals_out, POD_energy
        
        else:

            return None, None, None
    
    def update(self, f_new):
        # if isinstance(f_new, Iterable): 
        #     return sum([self.update(f) for f in f_new])
        # y_new = np.array([self.solver(f, a) for f, a in product([f_new], self.angles)])
        y_new = self.solver(f_new, self.angles)
        if (not self.usempi) or (self.usempi and self.rank==0):
            # print("y_new.shape = ", y_new.shape)
            # print("(f_new, angles) = ", (len(f_new), len(self.angles)))
            y_new = y_new.reshape(len(f_new), len(self.angles))
            self.freqs.extend(f_new)
            self.responses.extend(y_new)
            self._update_basis()
            self._fit_gps()
        return 0
        
    def _fit_gps(self):
        raise NotImplementedError()

    
    def _pred_gps(self, f_min, f_max, n_grid):
        raise NotImplementedError()

        
    def reconstruct(self, f_query_arr):
        raise NotImplementedError()


# -----------------------
# Reduced Basis GP class with 1D kernel
# -----------------------
class ReducedBasisGP1D(ReducedBasisGPBASE):
        
    def _fit_gps(self):
        
        self.gps_real = []
        self.gps_imag = []
        x_train = self.normalizerX.fit_transform(np.array(self.freqs)[:, None])
        
        self.normalizerY = []
        for i in range(self.r):
            # Separate real and imaginary parts
            y_train = self.coeffs[:, i][:, None]
            gp_r, gp_i = self.train_gp(
                x_train, y_train, terms=self.terms, 
                training_iter=1000, 
                # training_iter=max(min(10*self.r, 1000), 50), 
                verbose=self.verbose, normalize_y=self.normalizeY,
                dims = x_train.shape[-1]
                )
            self.gps_real.append(gp_r)
            self.gps_imag.append(gp_i)

        return 0
    
    def _pred_gps(self, f_min, f_max, n_grid):
        x_pred = self.sampler(f_min, f_max, n_grid)
        x_pred = self.normalizerX.transform(x_pred)
        total_mu, total_var = np.zeros(n_grid), np.zeros(n_grid)
        weight_sum, normalized_total_var = 0, np.zeros(n_grid)
        # preds = []
        for i in range(self.r):
            mu_r, std_r = self.gps_real[i].predict(x_pred, return_std=True)
            mu_i, std_i = self.gps_imag[i].predict(x_pred, return_std=True)
            # weight variance by singular value (importance of mode)
            weight = np.square(self.S[i])
            total_mu += weight * (mu_r**2 + mu_i**2)
            total_var += weight * (std_r**2 + std_i**2)
            std_r /= self.gps_real[i]._y_train_std
            std_i /= self.gps_imag[i]._y_train_std
            normalized_total_var += weight * (std_r**2 + std_i**2)
            weight_sum += weight
            # preds.append(weight * (mu_r**2 + mu_i**2))
        scaled_total_var = total_var / (weight_sum + 1e-30)
        frac_predictive = np.mean(normalized_total_var) / (weight_sum + 1e-30)
        return total_mu, scaled_total_var, x_pred, frac_predictive
        
    def reconstruct(self, f_query_arr):
        """Predict full angle response at new frequency"""
        def get_y_pred_per_f(f_query):
            Xq = self.normalizerX.transform(np.array([[f_query]]))
            coeffs_pred = []
            for i in range(self.r):
                mu_r, _ = self.gps_real[i].predict(Xq, return_std=True)
                mu_i, _ = self.gps_imag[i].predict(Xq, return_std=True)
                coeffs_pred.append(mu_r[0] + 1j*mu_i[0])
            coeffs_pred = np.array(coeffs_pred)
            # reconstruct: coeffs_pred * Vh
            return coeffs_pred @ self.Vh
        return np.array([get_y_pred_per_f(f_query) for f_query in f_query_arr])


# -----------------------
# Reduced Basis GP class with 2D kernel
# -----------------------
class ReducedBasisGP2D(ReducedBasisGPBASE):
        
    def _fit_gps(self):
        "fit 2d data directly"
        self.gps_real = []
        self.gps_imag = []
        x_train = self.normalizerX.fit_transform(np.array(self.freqs)[:, None])
        x_train = np.array(list(product(np.squeeze(x_train, axis=-1), np.arange(self.r))))
        y_train = self.coeffs.reshape(-1, 1, order="C")

        gp_r, gp_i = self.train_gp(
            x_train, y_train, terms=self.terms, training_iter=1000, 
            verbose=self.verbose, normalize_y=self.normalizeY, 
            dims = x_train.shape[-1]
            )
        self.gps_real.append(gp_r)
        self.gps_imag.append(gp_i)
        return 0
    
    def _pred_gps(self, f_min, f_max, n_grid):
        "pred 2d data directly"
        x_pred = self.sampler(f_min, f_max, n_grid)
        x_pred = self.normalizerX.transform(x_pred)
        x_pred = np.array(list(product(np.squeeze(x_pred, axis=-1), np.arange(self.r))))
        # total_mu, total_var = np.zeros(n_grid), np.zeros(n_grid)
        mu_r, std_r = self.gps_real[0].predict(x_pred, return_std=True)
        mu_i, std_i = self.gps_imag[0].predict(x_pred, return_std=True)
        mu_energy_density  = (mu_r**2  + mu_i**2 ).reshape(n_grid, self.r)
        std_energy_density = (std_r**2 + std_i**2).reshape(n_grid, self.r)
        weight = np.square(self.S)[None, :]
        total_mu = np.sum(weight * mu_energy_density, axis=-1)
        total_var = np.sum(weight * std_energy_density, axis=-1)
        std_r /= self.gps_real[0]._y_train_std
        std_i /= self.gps_imag[0]._y_train_std
        normalized_total_var = np.sum(weight * (std_r**2 + std_i**2), axis=-1)
        weight_sum = np.sum(weight)

        scaled_total_var = total_var / (weight_sum + 1e-30)
        frac_predictive = np.mean(normalized_total_var) / (weight_sum + 1e-30)
        return total_mu, scaled_total_var, x_pred, total_var, frac_predictive
        
    def reconstruct(self, f_query_arr):
        """Predict full angle response at new frequency"""
        Xq = self.normalizerX.transform(np.array(f_query_arr)[:, None])
        Xq = np.array(list(product(np.squeeze(Xq, axis=-1), np.arange(self.r))))
        mu_r, _ = self.gps_real[0].predict(Xq, return_std=True)
        mu_i, _ = self.gps_imag[0].predict(Xq, return_std=True)
        mu_r = mu_r.reshape(len(f_query_arr), self.r)
        mu_i = mu_i.reshape(len(f_query_arr), self.r)
        coeffs_pred = mu_r + 1j*mu_i
        return coeffs_pred @ self.Vh


# -----------------------
# Reduced Basis multi-task GP class
# -----------------------
class ReducedBasisGPMultiTask(ReducedBasisGPBASE):
        
    def _fit_gps(self):
        "fit 2d data directly"
        self.gps_real = []
        self.gps_imag = []
        x_train = self.normalizerX.fit_transform(np.array(self.freqs)[:, None])
        y_train = self.coeffs

        gp_r, gp_i = self.train_gp(
            x_train, y_train, terms=self.terms, 
            # training_iter=1000, 
            training_iter=max(min(200*self.r, 3000), 500), 
            verbose=self.verbose, normalize_y=self.normalizeY, 
            dims = x_train.shape[-1]
            )
        self.gps_real.append(gp_r)
        self.gps_imag.append(gp_i)
        return 0
    
    def _pred_gps(self, f_min, f_max, n_grid):
        "pred 2d data directly"
        x_pred = self.sampler(f_min, f_max, n_grid)
        x_pred = self.normalizerX.transform(x_pred)
        # total_mu, total_var = np.zeros(n_grid), np.zeros(n_grid)
        
        mu_r, std_r = self.gps_real[0].predict(x_pred, return_std=True)
        mu_i, std_i = self.gps_imag[0].predict(x_pred, return_std=True)
        # data shape: [frequency, coef]
        weight = np.square(self.S)[None, :]
        weight_sum = np.sum(weight)
        total_mu = np.sum(weight * (mu_r**2  + mu_i**2 ), axis=-1)
        total_var = np.sum(weight * (std_r**2 + std_i**2), axis=-1)
        std_r /= self.gps_real[0]._y_train_std
        std_i /= self.gps_imag[0]._y_train_std
        normalized_total_var = np.sum(weight * (std_r**2 + std_i**2), axis=-1)
        
        scaled_total_mu = total_mu / (weight_sum + 1e-30)
        scaled_total_var = total_var / (weight_sum + 1e-30)
        frac_predictive = np.mean(normalized_total_var) / (weight_sum + 1e-30)
        # frac_predictive = np.quantile(total_var / (weight_sum + 1e-30), 0.9)
        return scaled_total_mu, scaled_total_var, x_pred, frac_predictive
    
    # def _pred_gps(self, f_min, f_max, n_grid, eps=1e-12):
    #     "pred 2d data directly — corrected analytic estimator"
    #     x_pred = self.sampler(f_min, f_max, n_grid)
    #     x_pred = self.normalizerX.transform(x_pred)

    #     # predict means and std (in the *normalized* target space)
    #     mu_r, std_r = self.gps_real[0].predict(x_pred, return_std=True)  # shape (n_grid, ncoef)
    #     mu_i, std_i = self.gps_imag[0].predict(x_pred, return_std=True)

    #     # Convert std back to original target units (if you trained on y / y_std)
    #     ystd_r = getattr(self.gps_real[0], "_y_train_std", 1.0)
    #     ystd_i = getattr(self.gps_imag[0], "_y_train_std", 1.0)
    #     std_r_orig = std_r * ystd_r
    #     std_i_orig = std_i * ystd_i

    #     # means may also be in normalized units — convert to original units if needed
    #     mu_r_orig = mu_r * ystd_r
    #     mu_i_orig = mu_i * ystd_i

    #     # weight: shape (1, ncoef) so broadcasting over grid rows
    #     weight = np.square(self.S)[None, :]  # (1, ncoef)
    #     weight_sum = np.sum(weight)

    #     # Expected power per grid point including variance terms:
    #     # E[|c_k|^2] = mu_r^2 + mu_i^2 + var_r + var_i
    #     var_r = std_r_orig**2
    #     var_i = std_i_orig**2
    #     expected_per_coef = mu_r_orig**2 + mu_i_orig**2 + var_r + var_i  # (n_grid, ncoef)

    #     total_expected_power = np.sum(weight * expected_per_coef, axis=-1)  # (n_grid,)

    #     # Approximate variance of |c_k|^2 for each coefficient:
    #     # A commonly used approximation (assuming Gaussian r/i and independence):
    #     # Var(|c|^2) ≈ 2*(var_r**2 + var_i**2) + 4*(mu_r^2 * var_r + mu_i^2 * var_i)
    #     var_power_per_coef = 2*(var_r**2 + var_i**2) + 4*(mu_r_orig**2 * var_r + mu_i_orig**2 * var_i)

    #     # Combine (assume coefficients independent)
    #     total_var_power = np.sum((weight**2) * var_power_per_coef, axis=-1)  # (n_grid,)

    #     # Turn these into per-grid relative uncertainty:
    #     rel_std_per_grid = np.sqrt(total_var_power) / (total_expected_power + eps)

    #     # Scalar termination metric — choose one:
    #     # - use a robust statistic (e.g., 90th percentile) instead of mean to reduce fluctuation
    #     frac_predictive = np.quantile(rel_std_per_grid, 0.9)  # or np.median(...)

    #     # Also return scaled mean/var (averaged over weights)
    #     scaled_total_mu = total_expected_power / (weight_sum + eps)
    #     scaled_total_var = total_var_power / ((weight_sum**2) + eps)  # because we used weight**2 in var

    #     return scaled_total_mu, scaled_total_var, x_pred, frac_predictive
        
    def reconstruct(self, f_query_arr):
        """Predict full angle response at new frequency"""
        Xq = self.normalizerX.transform(np.array(f_query_arr)[:, None])
        mu_r, _ = self.gps_real[0].predict(Xq, return_std=True)
        mu_i, _ = self.gps_imag[0].predict(Xq, return_std=True)
        coeffs_pred = mu_r + 1j*mu_i
        return coeffs_pred @ self.Vh