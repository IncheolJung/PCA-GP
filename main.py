import numpy as np
from numpy.random import default_rng
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process.kernels import ExpSineSquared
from numpy.linalg import svd
from scipy.stats import norm
from datareader import *

# -----------------------
# Mock solver placeholder
# Replace with your expensive simulator
# -----------------------
# def solver(frequencies, angles):
#     """
#     frequencies: list/array of frequencies to evaluate
#     angles: array of incident angles (fixed set)
#     Returns: array of shape (len(frequencies), len(angles)), complex
#     """
#     # Example: resonant Lorentz-like response depending on f and θ
#     f = np.array(frequencies)[:, None]
#     th = np.array(angles)[None, :]
#     resp = 1.0 / (1.0 - (f/5.0)**2 + 0.1j) * np.cos(th)
#     return resp
solver = fileIOdatareader("data/data-for-kenny-paper-HH.npz")

def expected_improvement(mu, std, f_best, xi=1e-6):
    # mu, std are for the scalar target (e.g., |pred|)
    z = (mu - f_best - xi) / (std + 1e-12)
    return (mu - f_best - xi) * norm.cdf(z) + std * norm.pdf(z)

class dummy_preprocesser:
    def __init__(self):
        return None
    def fit(self, data):
        return self
    def transform(self, data):
        return data
    def inverse_transform(self, data):
        return data

# -----------------------
# Pipeline 2 implementation
# -----------------------
class ReducedBasisGP:
    def __init__(self, angles, n_init=6, r=3):
        self.angles = angles
        self.n_init = n_init
        self.r = r
        self.freqs = []
        self.responses = []
        self.U = None
        self.S = None
        self.Vh = None
        self.coeffs = None
        self.gps_real = []
        self.gps_imag = []
        self.fbest = None
        
    def initialize(self, f_min, f_max):
        # log- or lin-spaced initial design
        # f_init = np.linspace(f_min, f_max, self.n_init)
        f_init = self.latin_hypercube_sampling(f_min, f_max, self.n_init)
        self.freqs = list(f_init)
        self.fbest = np.mean(self.freqs)
        X = np.array(self.freqs)[:, None]
        self.normalizerX = dummy_preprocesser().fit(X)
        # Y = solver(f_init, self.angles)  # (n_init, n_angles)
        Y = np.array([solver(f, a) for f, a in product(f_init, self.angles)])  # (n_init, n_angles)
        Y = Y.reshape(self.n_init, len(self.angles))  # (n_init, n_angles)
        self.responses = list(Y) # store each freq response (complex vector)
        self._update_basis()
        self._fit_gps()
    
    def latin_hypercube_sampling(self, val_min, val_max, n):
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
        # Keep first r modes
        self.U, self.S, self.Vh = U[:, :self.r], S[:self.r], Vh[:self.r, :]
        # Compute coefficients (project responses on modes)
        self.coeffs = self.U * self.S  # shape (n_freqs, r)
        
    def _fit_gps(self):
        self.gps_real, self.gps_imag = [], []
        X = np.array(self.freqs)[:, None]
        for i in range(self.r):
            y = self.coeffs[:, i]
            # Separate real and imaginary parts
            kernel_constant = C(1.0, (1e-3, 1e3))
            kernel_rbf = RBF(1.0, (1e-10, 1e3))
            kernel_periodic = ExpSineSquared(1.0, 1.0, (1e-4, 1e4), (1e-2, 1e2))
            kernel = kernel_constant * kernel_rbf * kernel_periodic
            gp_r = GaussianProcessRegressor(kernel=kernel, alpha=1e-10, normalize_y=True, n_restarts_optimizer=10)
            gp_i = GaussianProcessRegressor(kernel=kernel, alpha=1e-10, normalize_y=True, n_restarts_optimizer=10)
            gp_r.fit(self.normalizerX.transform(X), y.real)
            gp_i.fit(self.normalizerX.transform(X), y.imag)
            self.gps_real.append(gp_r)
            self.gps_imag.append(gp_i)
            
    def acquisition_next_frequency(self, f_min, f_max, n_grid=101):
        """Pick frequency that maximizes integrated variance across coefficients"""
        grid = self.normalizerX.transform(np.linspace(f_min, f_max, n_grid)[:, None])
        total_mu = np.zeros(n_grid)
        total_var = np.zeros(n_grid)
        for i in range(self.r):
            mu_r, var_r = self.gps_real[i].predict(grid, return_std=True)
            mu_i, var_i = self.gps_imag[i].predict(grid, return_std=True)
            # weight variance by singular value (importance of mode)
            weight = self.S[i]**2
            total_mu += weight * (mu_r**2 + mu_i**2)
            total_var += weight * (var_r**2 + var_i**2)
        idx = np.argmax(total_var)
        # print("\n\nself.fbest:", self.fbest)
        # fbest = self.normalizerX.transform([[self.fbest]])[0][0]
        # idx = np.argmax(expected_improvement(total_mu, total_var, fbest))
        return grid[idx, 0], np.sum(total_var)
    
    def update(self, f_new):
        y_new = np.array([solver(f, a) for f, a in product([f_new], self.angles)])
        y_new = y_new.reshape(1, len(self.angles))[0]
        y_pred = self.reconstruct(self.freqs)
        self.fbest = self.freqs[np.argmin(np.linalg.norm(y_new-y_pred))]
        self.freqs.append(f_new)
        self.responses.append(y_new)
        self._update_basis()
        self._fit_gps()
        
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
    

if __name__ == "__main__":
    # -----------------------
    # Example usage
    # -----------------------
    f_min, f_max = 9.5, 10.5
    angles = np.linspace(0, 181, 181)  # 181 angles
    rbgp = ReducedBasisGP(angles, n_init=5, r=5)
    rbgp.initialize(f_min=f_min, f_max=f_max)

    f_test = np.linspace(f_min, f_max, 101)

    max_iter = 101
    tol = 1e-6
    for it in range(max_iter):  # 5 adaptive iterations
        f_next, std_total = rbgp.acquisition_next_frequency(f_min, f_max, len(f_test))
        f_next_readible = rbgp.normalizerX.inverse_transform([[f_next]])[0][0]
        print(f"Iteration {it}: sampling new frequency {f_next_readible:.3f}")
        rbgp.update(f_next)
        if std_total < tol: 
            print("\n ======  Stopping criterion met.  ====== \n")
            print("  >> Final iteration:", it, "/", max_iter, sep="\t")
            print("  >> Final variance:", std_total, sep="\t")
            print("  >> total n_freq:", len(rbgp.freqs), sep="\t")
            break

    # Predict at new frequency
    pred  = rbgp.reconstruct(f_test)
    # truth = solver(f_test, rbgp.angles)
    truth = np.array([solver(f, a) for f, a in product(f_test, rbgp.angles)])  # (n_init, n_angles)
    truth = truth.reshape(len(f_test), len(rbgp.angles))  # (n_init, n_angles)
    print("Predicted response shape:", pred.shape)
    print("RMSE:", np.sqrt(np.mean(np.square(np.abs((pred - truth) / truth)))))

    from matplotlib.pyplot import subplots, show as pltshow
    from matplotlib import rcParams
    rcParams["text.usetex"] = True
    rcParams["font.size"] = 12

    ########### 1d plot ###########
    hf, hx = subplots(nrows=2, figsize=(12,8), constrained_layout=True)
    hx[0].plot(rbgp.angles, pred.real.T,  'r-', label="PRED RE")
    hx[0].plot(rbgp.angles, truth.real.T, 'b-', label="TRUTH RE")
    hx[1].plot(rbgp.angles, pred.imag.T,  'r-', label="PRED IM")
    hx[1].plot(rbgp.angles, truth.imag.T, 'b-', label="TRUTH IM")
    hf.legend()
    ######### end 1d plot #########

    ########### 2d plot ###########
    hf, hx = subplots(nrows=2, ncols=2, figsize=(12,8), constrained_layout=True)
    extent = [f_test.min(), f_test.max(), rbgp.angles.min(), rbgp.angles.max()]
    hx[0,0].imshow(pred.real,  cmap="turbo", aspect="auto", extent=extent)
    hx[0,1].imshow(truth.real, cmap="turbo", aspect="auto", extent=extent)
    hx[1,0].imshow(pred.imag,  cmap="turbo", aspect="auto", extent=extent)
    hx[1,1].imshow(truth.imag, cmap="turbo", aspect="auto", extent=extent)
    hx[0,0].set_title("PRED RE")
    hx[0,1].set_title("TRUTH RE")
    hx[1,0].set_title("PRED IM")
    hx[1,1].set_title("TRUTH IM")
    ######### end 2d plot #########

    pltshow()