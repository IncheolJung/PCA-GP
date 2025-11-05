import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import cho_solve, cho_factor, solve_triangular
from jaxopt import LBFGS
import equinox as eqx

jax.config.update("jax_enable_x64", True)


# ============================================================
#   Optimized kernel functions (vectorized + JIT-friendly)
# ============================================================

def ensure_2d(X):
    X = jnp.asarray(X)
    if X.ndim == 1:
        X = X[:, None]  # treat as single feature
    return X

def rbf_kernel(X1, X2, lengthscale):
    X1, X2 = ensure_2d(X1), ensure_2d(X2)
    diff = (X1[:, None, :] - X2[None, :, :]) / lengthscale
    return jnp.exp(-0.5 * jnp.sum(diff ** 2, axis=-1))

def periodic_kernel(X1, X2, period, gamma=1.0):
    X1, X2 = ensure_2d(X1), ensure_2d(X2)
    diff = jnp.sqrt(jnp.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=-1))
    s = jnp.sin(jnp.pi * diff / period)
    return jnp.exp(-2 * (s ** 2) * gamma)

def combined_kernel(X1, X2, params, terms=1):
    """params: length 2*terms -> alternating [rbf_scale, period] per term"""
    X1, X2 = ensure_2d(X1), ensure_2d(X2)
    k = 0.0
    for i in range(terms):
        lengthscale = jnp.exp(params[2*i])
        period = jnp.exp(params[2*i+1])
        k += rbf_kernel(X1, X2, lengthscale) * periodic_kernel(X1, X2, period)
    return k


# ============================================================
#   Optimized GPR class (caches Cholesky + jitted predict)
# ============================================================

class GPR_tinygp(eqx.Module):
    X: jnp.ndarray
    ydata: jnp.ndarray
    L: jnp.ndarray
    alpha: jnp.ndarray
    params: jnp.ndarray
    terms: int
    _y_train_mean: float
    _y_train_std: float
    noise: float = 1e-6

    @staticmethod
    def from_training(X, y, params, terms=1, noise=1e-6):
        """Build trained GP with cached Cholesky."""
        K = combined_kernel(X, X, params, terms) + noise * jnp.eye(X.shape[0])
        L = jnp.linalg.cholesky(K)
        alpha = cho_solve((L, True), y)
        y_mean, y_std = jnp.mean(y), jnp.std(y)
        return GPR_tinygp(X, y, L, alpha, params, terms, y_mean, y_std, noise)

    @eqx.filter_jit
    def _predict(self, X_new, return_std=False):
        KxX = combined_kernel(X_new, self.X, self.params, self.terms)
        mu = KxX @ self.alpha
        y_mean, y_std = self._y_train_mean, self._y_train_std
        if return_std:
            v = solve_triangular(self.L, KxX.T, lower=True)
            var = jnp.clip(
                jnp.diag(combined_kernel(X_new, X_new, self.params, self.terms))
                - jnp.sum(v**2, axis=0), 0, jnp.inf
            )
            return mu * y_std + y_mean, jnp.sqrt(var) * y_std
        return mu * y_std + y_mean
    
    def predict(self, *args, **kwargs):
        if kwargs["return_std"]:
            return map(np.asarray, self._predict(*args, **kwargs))
        else:
            return np.asarray(self._predict(*args, **kwargs))


# ============================================================
#   Optimized training function (JITed loss + L-BFGS)
# ============================================================

def train_gp_tinygp(
    X, Y, terms=1, training_iter=1000, verbose=False, normalize_y=True, dims=1
):
    # ---------- Shape handling ----------
    X = jnp.asarray(X)
    if X.ndim == 1 and dims > 1:
        X = X.reshape(-1, dims)
    elif X.ndim == 2 and X.shape[1] == 1 and dims == 1:
        X = X.reshape(-1)

    Y_r = jnp.asarray(Y.real).ravel()
    Y_i = jnp.asarray(Y.imag).ravel()

    # ---------- Normalization ----------
    if normalize_y:
        y_mean_r, y_std_r = jnp.mean(Y_r), jnp.std(Y_r)
        Y_r = (Y_r - y_mean_r) / y_std_r
        y_mean_i, y_std_i = jnp.mean(Y_i), jnp.std(Y_i)
        Y_i = (Y_i - y_mean_i) / y_std_i
    else:
        y_mean_r, y_std_r = 0.0, 1.0
        y_mean_i, y_std_i = 0.0, 1.0

    # ---------- JITed negative log marginal likelihood ----------
    @jax.jit
    def nll(params, X, y, noise=1e-6):
        K = combined_kernel(X, X, params, terms) + noise * jnp.eye(X.shape[0])
        L = jnp.linalg.cholesky(K)
        alpha = cho_solve((L, True), y)
        nll_val = 0.5 * (y @ alpha) + jnp.sum(jnp.log(jnp.diag(L))) + 0.5 * X.shape[0] * jnp.log(2*jnp.pi)
        return nll_val

    def _train_single(Y_t):
        params0 = jnp.zeros(2*terms)
        lbfgs = LBFGS(fun=lambda p: nll(p, X, Y_t), maxiter=training_iter)
        opt_params, state = lbfgs.run(params0)
        if verbose:
            print(f"Final NLL: {state.value}")
        gp = GPR_tinygp.from_training(X, Y_t, opt_params, terms=terms)
        return gp, state.value

    # ---------- Train real/imag ----------
    gp_r, loss_r = _train_single(Y_r)
    gp_i, loss_i = _train_single(Y_i)

    # ---------- Store normalization info ----------
    gp_r = eqx.tree_at(lambda g: (g._y_train_mean, g._y_train_std), gp_r, (y_mean_r, y_std_r))
    gp_i = eqx.tree_at(lambda g: (g._y_train_mean, g._y_train_std), gp_i, (y_mean_i, y_std_i))
    return gp_r, gp_i
