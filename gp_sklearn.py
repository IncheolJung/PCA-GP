from numpy import array, cos
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process.kernels import ExpSineSquared
from sklearn.exceptions import ConvergenceWarning
import warnings


# -----------------------
# Mock solver placeholder
# Replace with your expensive simulator
# -----------------------
def solver(frequencies, angles):
    """
    frequencies: list/array of frequencies to evaluate
    angles: array of incident angles (fixed set)
    Returns: array of shape (len(frequencies), len(angles)), complex
    """
    # Example: resonant Lorentz-like response depending on f and θ
    f = array(frequencies)[:, None]
    th = array(angles)[None, :]
    resp = 1.0 / (1.0 - (f/5.0)**2 + 0.1j) * cos(th)
    return resp


# -----------------------
# sklearn gp
# -----------------------
def train_gp_sklearn(
    X, Y, terms: int = 1, training_iter=1000, 
    verbose=False, normalize_y=True, dims = 1
):
    kernel = None
    for i in range(terms):
        kernel_constant = C(1.0, (1e-10, 1e3))
        kernel_rbf = RBF(1.0, (1e-10, 1e3))
        kernel_periodic = ExpSineSquared(1.0, 1.0, (1e-10, 1e4), (1e-10, 1e4))
        if i==0: kernel  = kernel_constant * kernel_rbf * kernel_periodic
        else:    kernel += kernel_constant * kernel_rbf * kernel_periodic
    # gp_r = GPR(kernel=kernel, alpha=1e-8, normalize_y=normalize_y, n_restarts_optimizer=10)
    # gp_i = GPR(kernel=kernel, alpha=1e-8, normalize_y=normalize_y, n_restarts_optimizer=10)
    def scipy_optimizer(obj_func, initial_theta, bounds):
        from scipy.optimize import fmin_l_bfgs_b
        theta_opt, func_min, _ = fmin_l_bfgs_b(
            obj_func, initial_theta, bounds=bounds, 
            maxiter=training_iter
        )
        return theta_opt, func_min
    gp_r = GPR(
        kernel=kernel, alpha=1e-8, optimizer=scipy_optimizer, 
        normalize_y=normalize_y, n_restarts_optimizer=10)
    gp_i = GPR(
        kernel=kernel, alpha=1e-8, optimizer=scipy_optimizer, 
        normalize_y=normalize_y, n_restarts_optimizer=10)
    # gp_r.fit(X, Y.real)
    # gp_i.fit(X, Y.imag)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gp_r.fit(X, Y.real)
        gp_i.fit(X, Y.imag)
    return gp_r, gp_i