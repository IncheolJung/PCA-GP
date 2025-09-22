import sys
import warnings
import numpy as np
from numpy.random import default_rng
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
from sklearn.pipeline import Pipeline
from numpy.linalg import svd

from joblib import Parallel, delayed
# import threading

from datareader import *
from acquisitionfunctions import *

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

def train_gp(X, Y, terms: int = 1):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
    from sklearn.gaussian_process.kernels import ExpSineSquared
    kernel = None
    for i in range(terms):
        kernel_constant = C(1.0, (1e-10, 1e3))
        kernel_rbf = RBF(1.0, (1e-10, 1e3))
        kernel_periodic = ExpSineSquared(1.0, 1.0, (1e-10, 1e4), (1e-10, 1e4))
        if i==0: kernel  = kernel_constant * kernel_rbf * kernel_periodic
        else:    kernel += kernel_constant * kernel_rbf * kernel_periodic
    gp_r = GaussianProcessRegressor(kernel=kernel, alpha=1e-8, normalize_y=True, n_restarts_optimizer=10)
    gp_i = GaussianProcessRegressor(kernel=kernel, alpha=1e-8, normalize_y=True, n_restarts_optimizer=10)
    gp_r.fit(X, Y.real)
    gp_i.fit(X, Y.imag)
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore")
    #     gp_r.fit(X, Y.real)
    #     gp_i.fit(X, Y.imag)
    return gp_r, gp_i

# from gp_Kenny import train_model_gp_Kenny
# train_gp = train_model_gp_Kenny

class dummy_preprocesser:
    def __init__(self):
        return None
    def fit(self, data):
        return self
    def transform(self, data):
        return np.array(data)
    def inverse_transform(self, data):
        return np.array(data)


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
# Pipeline 2 implementation
# -----------------------
class ReducedBasisGP:
    def __init__(self, solver, angles, n_init=6, r=3, adaptive_r=True, acquisition_type=0, Xnormalizer_type=0, terms=1):
        self.solver = solver
        self.angles = angles
        self.n_init = n_init
        self.r = r
        self.adaptive_r = adaptive_r
        self.acquisition_type = acquisition_type
        self.Xnormalizer_type = Xnormalizer_type
        self.terms = terms
        self.freqs = []
        self.responses = []
        self.U = None
        self.S = None
        self.Vh = None
        self.coeffs = None
        self.gps_real = []
        self.gps_imag = []
        self.fbest = None
        return None
        
    def initialize(self, f_min, f_max):

        # log- or lin-spaced initial design
        # f_init = np.linspace(f_min, f_max, self.n_init)
        f_init = self.latin_hypercube_sampling(f_min, f_max, self.n_init)
        self.freqs = list(f_init)
        # Y = self.solver(f_init, self.angles)  # (n_init, n_angles)
        Y = np.array([self.solver(f, a) for f, a in product(f_init, self.angles)])  # (n_init, n_angles)
        # self.fbest = np.mean(Y)
        Y = Y.reshape(self.n_init, len(self.angles))  # (n_init, n_angles)
        self.responses = list(Y) # store each freq response (complex vector)

        # Data normalizer
        X = np.array(self.freqs)[:, None]
        if (self.Xnormalizer_type==0):      # do nothing
            self.normalizerX = dummy_preprocesser().fit(X)
        elif (self.Xnormalizer_type==1):    # z-score
            self.normalizerX = StandardScaler().fit(X)
        elif (self.Xnormalizer_type==2):    # min-max
            self.normalizerX = MinMaxScaler().fit(X)
        elif (self.Xnormalizer_type==3):    # power-transform
            self.normalizerX = PowerTransformer().fit(X)
        elif (self.Xnormalizer_type==4):    # standardizer + power-transform
            self.normalizerX = Pipeline([("std", StandardScaler()), ("pwr", PowerTransformer())]).fit(X)
        elif (self.Xnormalizer_type==5):    # standardizer + min-max
            self.normalizerX = Pipeline([
                ("std", StandardScaler()), 
                ("scale", MinMaxScaler(feature_range=(0, 100)))
                ]).fit(X)
        else: RuntimeError(f"INVALID ACQUISITION TYPE: {self.Xnormalizer_type} must be < 6")

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
        if self.adaptive_r:
            # threshold
            eps = 1e-4
            e = 1
            r = 0
            for sr in S[1:]:
                r += 1
                e = (sr/S[0])**2
                if (e<(eps**2)): break
            print(f"number of basis updated: {self.r} -> {r}")
            print("S[0, r] = ", S[0], S[r])
            if (r+1)<len(S): print("S[r+1] =", S[r+1])
            self.r = r
        # Keep first r modes
        self.U, self.S, self.Vh = U[:, :self.r], S[:self.r], Vh[:self.r, :]
        # Compute coefficients (project responses on modes)
        self.coeffs = self.U * self.S  # shape (n_freqs, r)
        return 0
        
    # def _init_gps(self):
    #     self.gps_real, self.gps_imag = [], []
    #     for i in range(self.r):
    #         # Separate real and imaginary parts
    #         kernel_constant = C(1.0, (1e-3, 1e3))
    #         kernel_rbf = RBF(1.0, (1e-10, 1e3))
    #         kernel_periodic = ExpSineSquared(1.0, 1.0, (1e-4, 1e4), (1e-4, 1e4))
    #         kernel = kernel_constant * kernel_rbf * kernel_periodic
    #         gp_r = GaussianProcessRegressor(kernel=kernel, alpha=1e-10, normalize_y=True, n_restarts_optimizer=10)
    #         gp_i = GaussianProcessRegressor(kernel=kernel, alpha=1e-10, normalize_y=True, n_restarts_optimizer=10)
    #         self.gps_real.append(gp_r)
    #         self.gps_imag.append(gp_i)
    #     return 0
        
    # def _fit_gps(self):
    #     X = self.normalizerX.transform(np.array(self.freqs)[:, None])
    #     for i in range(self.r):
    #         y = self.coeffs[:, i]
    #         self.gps_real[i].fit(X, y.real)
    #         self.gps_imag[i].fit(X, y.imag)
    #     return 0
        
    def _fit_gps(self):
        # self.gps_real = [None] * self.r
        # self.gps_imag = [None] * self.r
        def _train_one(i, x_train, coeffs, r):
            y_train = coeffs[:, i]
            gp_r, gp_i = train_gp(x_train, y_train, terms=self.terms)
            return gp_r, gp_i
            # self.gps_real[i] = gp_r
            # self.gps_imag[i] = gp_i
            # return None
        
        x_train = self.normalizerX.transform(np.array(self.freqs)[:, None])

        # Parallel training
        results = Parallel(n_jobs=-1)(  # use all cores
            delayed(_train_one)(i, x_train, self.coeffs, self.r)
            for i in range(self.r)
        )

        # Unpack results
        self.gps_real, self.gps_imag = zip(*results)
        # Convert back to lists if needed
        self.gps_real, self.gps_imag = list(self.gps_real), list(self.gps_imag)

        # threads = [threading.Thread(target=_train_one, args=(i,x_train,)) for i in range(self.r)]
        # for t in threads: t.start()
        # for t in threads: t.join()

        # for i in range(self.r):
        #     # Separate real and imaginary parts
        #     y_train = self.coeffs[:, i]
        #     gp_r, gp_i = train_gp(x_train, y_train, terms=self.r)
        #     self.gps_real.append(gp_r)
        #     self.gps_imag.append(gp_i)
        return 0
            
    def acquisition_next_frequency(self, f_min, f_max, n_grid=101):
        """Pick frequency that maximizes integrated variance across coefficients"""
        # mu_list = [None] * self.r
        # var_list = [None] * self.r
        def _predict_one(i, gps_real, gps_imag, S, grid):
            mu_r, std_r = gps_real[i].predict(grid, return_std=True)
            mu_i, std_i = gps_imag[i].predict(grid, return_std=True)
            weight = S[i]**2
            return weight * (mu_r**2 + mu_i**2), weight * (std_r**2 + std_i**2)
            # mu_list[i]  = weight * (mu_r**2 + mu_i**2)
            # var_list[i] = weight * (std_r**2 + std_i**2)
            # return None
        
        # grid = self.normalizerX.transform(np.linspace(f_min, f_max, n_grid)[:, None])
        freq_lhs = self.latin_hypercube_sampling(f_min, f_max, n_grid)
        grid = self.normalizerX.transform(freq_lhs[:, None])

        results = Parallel(n_jobs=-1)(  # -1 = all cores
            delayed(_predict_one)(i, self.gps_real, self.gps_imag, self.S, grid)
            for i in range(self.r)
        )

        total_mu, total_var = np.sum(np.array(results), axis=0)

        # threads = [threading.Thread(target=_predict_one, args=(i, self.gps_real, self.gps_imag, self.S, grid)) for i in range(self.r)]
        # for t in threads: t.start()
        # for t in threads: t.join()

        # total_mu = np.sum(mu_list, axis=0)
        # total_var = np.sum(var_list, axis=0)

        # grid = self.normalizerX.transform(np.linspace(f_min, f_max, n_grid)[:, None])
        # total_mu = np.zeros(n_grid)
        # total_var = np.zeros(n_grid)
        # for i in range(self.r):
        #     mu_r, std_r = self.gps_real[i].predict(grid, return_std=True)
        #     mu_i, std_i = self.gps_imag[i].predict(grid, return_std=True)
        #     # weight variance by singular value (importance of mode)
        #     weight = self.S[i]**2
        #     total_mu += weight * (mu_r**2 + mu_i**2)
        #     total_var += weight * (std_r**2 + std_i**2)

        responses_pred = self.reconstruct(self.freqs)   # [freq, angle]
        loss_per_freq = np.mean(np.square(self.responses-responses_pred), axis=1)
        self.fbest = total_mu[np.argmin(loss_per_freq)]

        ac_vals = self.acquisition_function(total_mu, total_var)
        idx = np.argmax(ac_vals)
        grid = self.normalizerX.inverse_transform(grid)
        for new_idx in np.flip(np.argsort(ac_vals)):
            if grid[new_idx, 0] not in self.freqs:
                idx = new_idx
                break
        denom = self.r/len(self.freqs)

        return grid[idx, 0], np.sum(ac_vals)/denom, np.sum(total_var)/denom
    
    def update(self, f_new):
        y_new = np.array([self.solver(f, a) for f, a in product([f_new], self.angles)])
        y_new = y_new.reshape(1, len(self.angles))[0]
        self.freqs.append(f_new)
        self.responses.append(y_new)
        self._update_basis()
        self._fit_gps()
        return 0
        
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


def main():
    # -----------------------
    # CONFIG
    # -----------------------
    adaptive_basis = True
    acquisition_function = 0
    acquisition_function_candidate = ["maximum variance", "expected improvement", "upper confidence bound"]
    Xnormalizer_type = 2
    Xnormalizer_type_candidate = ["pass", "z-score", "min-max", "power transform", "standardized power transform", "scaled z-score"]
    n_init = 3
    terms  = 5
    max_iter = 20
    tol = 1e-5
    # -----------------------
    # SOLVER
    # -----------------------

    # solver = fileIOdatareader("data/data-for-kenny-paper-HH.npz")

    # solver = OnFlySolver(
    #     workingpath="./data/VWT-data/sphere",
    #     model_name="sphere",
    #     angles=np.linspace(0, 180, 181)
    #     )

    solver = OnFlySolver(
        workingpath="./data/VWT-data/prime-airplane",
        model_name="Open-Duct_PRIME_model_meshAA",
        angles=np.linspace(0, 180, 181)
        )
    # -----------------------
    # OUTPUT DRIECTORY SETUP
    # -----------------------
    dir_out = Path("out")
    simulation_number = len([d for d in dir_out.glob("*") if d.is_dir()]) + 1
    dir_out = dir_out/Path(f"GP_test_{simulation_number:04d}")
    dir_out.mkdir()
    stdout = Tee(dir_out/"GP_results.log", "w")     # Log file setup
    print(f"\n ======  Simulation {simulation_number} Initialized  ====== \n")
    print(f"  >> Adaptive basis: {adaptive_basis}")
    print(f"  >> Acquisition function: {acquisition_function_candidate[acquisition_function]}")
    print(f"  >> X normalization strategy: {Xnormalizer_type_candidate[Xnormalizer_type]}")
    print(f"  >> n_init: {n_init}")
    print(f"  >> terms: {terms}")
    print(f"  >> max_iter: {max_iter}")
    print(f"  >> tol: {tol}")
    print(f"\n ======  Simulation {simulation_number} Initialized  ====== \n")
    stdout.flush()
    # -----------------------
    # BEGIN
    # -----------------------
    f_min, f_max, f_num = 9500, 10500, 101
    # f_min, f_max, f_num = 500, 1500, 151
    # f_test = np.linspace(f_min, f_max, 101)
    f_test = np.linspace(f_min, f_max, f_num)
    angles = np.linspace(0, 180, 181)  # 181 angles
    # angles = np.linspace(0, 180, 19)
    rbgp = ReducedBasisGP(solver, angles, n_init=n_init, r=n_init, adaptive_r=adaptive_basis, 
                          acquisition_type=acquisition_function, Xnormalizer_type=Xnormalizer_type, terms=terms)
    rbgp.initialize(f_min=f_min, f_max=f_max)


    # max_iter = len(f_test) - n_init
    for it in range(max_iter):  # 5 adaptive iterations
        f_next, ac_fx, avg_var = rbgp.acquisition_next_frequency(f_min, f_max, 4*len(f_test))
        print(f"\nIteration {it+1} / {max_iter}: acquisition {ac_fx:.10f} | variance {avg_var:.10f}")
        print("number of frequency samples:", len(rbgp.freqs))
        print(f"sampling new frequency {f_next:.3f}")
        rbgp.update(f_next)
        stdout.flush()
        if avg_var < tol: 
            break
    print("\n ======  Stopping criterion met.  ====== \n")
    print("  >> Final iteration:", it+1, "/", max_iter, sep="\t")
    print("  >> Final acquisition:", ac_fx, sep="\t")
    print("  >> Final variance:", avg_var, sep="\t")
    print("  >> total n_freq:", len(rbgp.freqs), sep="\t")
    stdout.flush()

    # Predict at new frequency
    pred  = rbgp.reconstruct(f_test)
    # truth = solver(f_test, rbgp.angles)
    truth = np.array([solver(f, a) for f, a in product(f_test, rbgp.angles)])  # (n_init, n_angles)
    truth = truth.reshape(len(f_test), len(rbgp.angles))  # (n_init, n_angles)
    print("\n ======  Final Statistics  ====== \n")
    print("Predicted response shape:", pred.shape)
    print("RMSE:{:.12f}".format(np.mean(np.square(np.abs(pred - truth))) / np.mean(np.square(np.abs(truth)))) )
    stdout.flush()

    from matplotlib.pyplot import subplots, show as pltshow
    from matplotlib import rcParams
    rcParams["text.usetex"] = True
    rcParams["font.size"] = 12

    ########### 1d plot ###########
    hf, hx = subplots(nrows=2, figsize=(12,8), constrained_layout=True)
    for i in range(pred.shape[0]):
        hx[0].plot(rbgp.angles, pred.real[i],  'r-', alpha=0.5, label="PRED RE"  if i == 0 else None)
        hx[0].plot(rbgp.angles, truth.real[i], 'b-', alpha=0.5, label="TRUTH RE" if i == 0 else None)
    for i in range(pred.shape[0]):
        hx[1].plot(rbgp.angles, pred.imag[i],  'r-', alpha=0.5, label="PRED IM"  if i == 0 else None)
        hx[1].plot(rbgp.angles, truth.imag[i], 'b-', alpha=0.5, label="TRUTH IM" if i == 0 else None)
    hf.legend()
    hf.savefig(dir_out/"GP_results_1d.png")
    ######### end 1d plot #########

    ########### 2d plot ###########
    nrows, ncols = 2, 3
    hf, hx = subplots(nrows=nrows, ncols=ncols, figsize=(16,8), constrained_layout=True)
    extent = [rbgp.angles.min(), rbgp.angles.max(), f_test.min(), f_test.max()]
    im = np.empty((nrows,ncols), dtype="object")
    error_real = 20 * np.log10(np.square(np.abs(pred.real - truth.real)) / np.square(np.abs(truth.real)))
    error_imag = 20 * np.log10(np.square(np.abs(pred.imag - truth.imag)) / np.square(np.abs(truth.imag)))
    error = error_real + 1j*error_imag
    im[0,0] = hx[0,0].imshow(pred.real,  cmap="turbo", aspect="auto", extent=extent)
    im[0,1] = hx[0,1].imshow(truth.real, cmap="turbo", aspect="auto", extent=extent)
    im[1,0] = hx[1,0].imshow(pred.imag,  cmap="turbo", aspect="auto", extent=extent)
    im[1,1] = hx[1,1].imshow(truth.imag, cmap="turbo", aspect="auto", extent=extent)
    im[0,2] = hx[0,2].imshow(error.real,  cmap="turbo", aspect="auto", extent=extent)
    im[1,2] = hx[1,2].imshow(error.imag, cmap="turbo", aspect="auto", extent=extent)
    hx[0,0].set_title("PRED RE")
    hx[0,1].set_title("TRUTH RE")
    hx[1,0].set_title("PRED IM")
    hx[1,1].set_title("TRUTH IM")
    hx[0,2].set_title("RMSE RE [dB]")
    hx[1,2].set_title("RMSE IM [dB]")
    for i in range(nrows): 
        for j in range(ncols):
            hf.colorbar(im[i,j], ax=hx[i,j])
    hf.savefig(dir_out/"GP_results_2d.png")
    ######### end 2d plot #########

    # pltshow()
    return 0


if __name__ == "__main__":
    main()