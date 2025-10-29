import numpy as np
from sklearn import preprocessing
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process.kernels import ExpSineSquared
from sklearn.decomposition import TruncatedSVD
from itertools import product
from testfunctions import *
from datareader import *

# --- User-supplied expensive solver (placeholder) ---
# def expensive_solver(freq, angle):
#     # Replace this with actual far-field solver call
#     return func_non_separable(freq, angle)
expensive_solver = fileIOdatareader("data/data-for-kenny-paper-HH.npz")

# --- GP model ---
def train_gp(X, y):
    kernel_constant = C(1.0, (1e-3, 1e3))
    kernel_rbf = RBF([1.0, 1.0], (1e-6, 1e6))
    kernel_periodic = ExpSineSquared(1.0, 1.0, (1e-4, 1e4), (1e-2, 1e2))
    kernel = kernel_constant * kernel_rbf * kernel_periodic
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, n_restarts_optimizer=5)
    gp.fit(X, y)
    return gp

def train_svd_gp(X, Y, n_components=10):
    """
    X : (n_samples, d) design points
    Y : (n_samples, m) output matrix (m-dimensional outputs)
    """
    n_components = min(n_components, len(X))
    # Step 1: Reduce output dimension
    svd = TruncatedSVD(n_components=n_components)
    coeffs = svd.fit_transform(Y)   # (n_samples, r)

    # Step 2: Train GP for each mode coefficient
    gps = []
    for i in range(coeffs.shape[1]):
        gp = train_gp(X, coeffs[:, i])
        gps.append(gp)

    return gps, svd

def predict_svd_gp(gps, svd, Xnew):
    coeff_preds, std_preds = [], []
    for gp in gps:
        _mu, _std = gp.predict(Xnew, return_std=True)
        coeff_preds.append(_mu)
        std_preds.append(_std)
    coeff_preds = np.vstack(coeff_preds).T   # (n_new, r)
    std_preds = np.vstack(std_preds).T   # (n_new, r)
    Ypred = svd.inverse_transform(coeff_preds)  # back to (n_new, m)
    STDpred = svd.inverse_transform(std_preds)  # back to (n_new, m)
    return Ypred, STDpred

# # --- Acquisition function (variance-based) ---
# def select_next_point(gp, search_grid, sampled_points):
#     mean, std = gp.predict(search_grid, return_std=True)
#     # pick the point with maximum predictive variance
#     idx = np.argmax(std)
#     candidate = search_grid[idx]
#     # avoid duplicate points
#     if any(np.allclose(candidate, sp) for sp in sampled_points):
#         std[idx] = -np.inf
#         idx = np.argmax(std)
#         candidate = search_grid[idx]
#     return candidate, std[idx]

# --- Adaptive loop ---
def adaptive_sampling(freq_range, angle_range, tol=1e-2, max_iter=300):
    # Initial design: small Latin hypercube/random
    n_init = 10
    freqs = np.random.uniform(freq_range[0], freq_range[1], n_init)
    angles = np.random.uniform(angle_range[0], angle_range[1], n_init)
    X = np.vstack([freqs, angles]).T
    y = np.array([expensive_solver(f, a) for f, a in X])
    Y = np.stack([np.real(y), np.imag(y)], axis=-1)

    scaler = preprocessing.StandardScaler().fit(X)

    n_freq_search = 101
    n_phi_search = 181
    search_freqs = np.linspace(*freq_range, n_freq_search)
    search_angles = np.linspace(*angle_range, n_phi_search)
    search_grid = np.array(list(product(search_freqs, search_angles)))
    # search_grid = np.array([[f, a] for f in search_freqs for a in search_angles])

    for i in range(max_iter):

        # gp_real = train_gp(scaler.transform(X), y.real)
        # gp_imag = train_gp(scaler.transform(X), y.imag)

        # # Predict combined std (real + imag)
        # _, std_r = gp_real.predict(scaler.transform(search_grid), return_std=True)
        # _, std_i = gp_imag.predict(scaler.transform(search_grid), return_std=True)
        
        gps, svd = train_svd_gp(X, Y)
        _, std_preds = predict_svd_gp(gps, svd, search_grid)
        std_r, std_i = np.moveaxis(std_preds, -1, 0)

        std_total = np.sqrt(std_r**2 + std_i**2)

        max_std = np.max(std_total)
        print(f"Iter {i}: max predictive std = {max_std:.4e}")

        if max_std < tol:
            print("\n ======  Stopping criterion met.  ====== \n")
            break

        # pick next point
        # idx = np.argmax(std_total)
        std_2d = std_total.reshape(n_freq_search, n_phi_search)
        idx = np.argmax(np.sum(std_2d, axis=0))
        next_points = np.array(list(product(search_freqs[idx:idx+1], search_angles)))
        for next_p in next_points:
            f_new, a_new = next_p
            y_new = expensive_solver(f_new, a_new)
            X = np.vstack([X, next_p])
            Y = np.append(Y, y_new)

    return X, Y, gp_real, gp_imag


if __name__ == "__main__":

    # --- Run example ---
    freq_range = (9.5, 10.5)     # GHz
    angle_range = (0.0, 180.0)   # degrees
    tolerance = 1e-12
    X, y, gp_real, gp_imag = adaptive_sampling(freq_range, angle_range, tol=tolerance)

    freq_test = np.linspace(*freq_range, 101)
    angle_test = np.linspace(*angle_range, 181)
    X_test = np.array(list(product(freq_test, angle_test)))

    pred = gp_real.predict(X_test) + 1j * gp_imag.predict(X_test)
    truth = np.array([expensive_solver(f, a) for f, a in X_test])
    pred = pred.reshape(len(freq_test), len(angle_test))
    truth = truth.reshape(len(freq_test), len(angle_test))

    print("RMSE:", np.sqrt(np.mean(np.square(np.abs((pred - truth) / truth)))))

    from matplotlib.pyplot import subplots, show as pltshow
    from matplotlib import rcParams
    rcParams["text.usetex"] = True
    rcParams["font.size"] = 12

    ########### 2d plot ###########
    hf, hx = subplots(nrows=2, ncols=2, figsize=(12,8), constrained_layout=True)
    extent = [freq_test.min(), freq_test.max(), angle_test.min(), angle_test.max()]
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