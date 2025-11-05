import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax
from jaxopt import LBFGS
from tinygp import GaussianProcess, kernels
import warnings


jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


class GPR_tinygp(GaussianProcess):

    ydata: jnp.ndarray = eqx.field(default=None)        # non-static OK if you used tree_at earlier
    _y_train_mean: jnp.ndarray = eqx.field(default=None)
    _y_train_std: jnp.ndarray = eqx.field(default=None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def store_yinfo(self, ydata, ymean, ystd):
        # keep your chosen approach; here we mutate in-place (you used object.__setattr__)
        object.__setattr__(self, "ydata", jnp.asarray(ydata).ravel())
        object.__setattr__(self, "_y_train_mean", ymean)
        object.__setattr__(self, "_y_train_std", ystd)
        return self

    def _ensure_X_test_shape(self, X_test):
        """
        Make X_test have same tree/shape as training X stored in self.X.
        tinygp stores training inputs in self.X. We reshape X_test to
        (n_test, *self.X.shape[1:]) if needed.
        """
        X_test = jnp.asarray(X_test)
        X_train = self.X  # tinygp GaussianProcess stores the inputs here
        # get expected trailing shape (could be scalar or tuple)
        train_trail = jnp.shape(X_train)[1:]
        # if training inputs are 1D features per point, train_trail = (d,)
        if X_test.ndim == 1:
            # single test point given as 1D -> shape to (1, d) if d>1 else (n_test,1)
            if len(train_trail) == 0:
                # training X was shape (n_train,) (1D inputs), keep X_test as (n_test,)
                X_test = jnp.atleast_1d(X_test)
            else:
                X_test = X_test.reshape(1, *train_trail)
        else:
            # X_test.ndim >= 2: ensure trailing dims match
            if jnp.shape(X_test)[1:] != train_trail:
                # try to reshape if leading dimension plausible
                try:
                    n_test = X_test.shape[0]
                    X_test = X_test.reshape(n_test, *train_trail)
                except Exception:
                    raise ValueError(
                        f"X_test trailing shape {jnp.shape(X_test)[1:]} does not match "
                        f"training trailing shape {train_trail}. Provide X_test shaped (n_test, *{train_trail})."
                    )
        return X_test

    def predict(self, x_pred, return_std=False):
        # convert and align shapes
        x_pred_j = jnp.asarray(x_pred)
        x_pred_j = self._ensure_X_test_shape(x_pred_j)

        # ensure ydata is 1-D and matches training n
        y = jnp.asarray(self.ydata).ravel()

        # now call condition with matched shapes and structures
        conditioned = self.condition(y=y, X_test=x_pred_j).gp
        mu = conditioned.loc * self._y_train_std + self._y_train_mean
        if return_std:
            std = jnp.sqrt(conditioned.variance) * self._y_train_std
            return np.asarray(mu), np.asarray(std)
        return np.asarray(mu)


def train_gp_tinygp(
    X, Y, terms: int = 1, training_iter=1000, 
    verbose=False, normalize_y=True, dims = 1
):
    # Ensure X has shape (n_train, dims)
    X = jnp.asarray(X)
    if X.ndim == 1 and dims > 1:
        X = X.reshape(-1, dims)
    # If dims==1 then X could be (n,) or (n,1). Make it (n,) or (n,1) consistently:
    if X.ndim == 2 and X.shape[1] == 1 and dims == 1:
        X = X.reshape(-1)   # tinygp allows 1D inputs; decide which you want
    X_j = X   # do not .squeeze()

    # Y -> ensure training y is 1-D per GP expectation
    Y_r = jnp.asarray(Y.real).ravel()
    Y_i = jnp.asarray(Y.imag).ravel()

    # X_j = jax.device_put(X_j, device=jax.devices("gpu")[0])
    # Y_r = jax.device_put(Y_r, device=jax.devices("gpu")[0])
    # Y_i = jax.device_put(Y_i, device=jax.devices("gpu")[0])

    # Optional normalization
    if normalize_y:
        y_mean_r, y_std_r = jnp.mean(Y_r), jnp.std(Y_r)
        Y_r = (Y_r - y_mean_r) / y_std_r
        y_mean_i, y_std_i = jnp.mean(Y_i), jnp.std(Y_i)
        Y_i = (Y_i - y_mean_i) / y_std_i
    else:
        y_mean_r = y_std_r = y_mean_i = y_std_i = 0.0, 1.0

    def make_kernel(params):
        # params: array of length = 2*terms: [rbf_scales..., periodic_scales...]
        k = None
        for i in range(terms):
            scale = kernels.Constant(1.0)
            rbf = kernels.ExpSquared(scale=params[i])
            periodic = kernels.ExpSineSquared(scale=params[terms+i], gamma=1.0)
            term = scale * rbf * periodic
            k = term if k is None else k + term
        return k
    
    def _train_single(Y_t):
        init_params = jnp.ones(2*terms)

        # Define objective for L-BFGS
        def loss_fn(params):
            gp = GPR_tinygp(make_kernel(params), X_j, diag=1e-8)
            return -gp.log_probability(Y_t)

        # Create LBFGS optimizer from jaxopt
        lbfgs = LBFGS(fun=loss_fn, maxiter=training_iter)

        # Run optimization
        params_opt, _state = lbfgs.run(init_params)
        if verbose:
            print(f"Final negative log likelihood: {_state.value}")

        # Build trained GP with optimized kernel
        gp_trained = GPR_tinygp(make_kernel(params_opt), X_j, diag=1e-8)
        return gp_trained, _state.value

    # def _train_single(Y_t):
    #     # Initialize kernel parameters
    #     init_params = jnp.ones(2 * terms)  # first `terms` are RBF scales, next `terms` are periodic scales
    #     optimizer = optax.adam(learning_rate=0.05)
    #     opt_state = optimizer.init(init_params)

    #     @jax.jit
    #     def step(params, opt_state):
    #         def loss_fn(p):
    #             gp = GPR_tinygp(make_kernel(p), X_j, ydata=Y_t, diag=1e-8)
    #             return -gp.log_probability(Y_t)
    #         grads = jax.grad(loss_fn)(params)
    #         updates, opt_state = optimizer.update(grads, opt_state)
    #         params = optax.apply_updates(params, updates)
    #         return params, opt_state

    #     params = init_params
    #     for _ in range(training_iter):
    #         params, opt_state = step(params, opt_state)

    #     # Return trained GP with optimized kernel
    #     gp_trained = GPR_tinygp(make_kernel(params), X_j, ydata=Y_t, diag=1e-8)
    #     return gp_trained

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        gp_r, loss_r = _train_single(Y_r)
        gp_i, loss_i = _train_single(Y_i)
    gp_r = gp_r.store_yinfo(Y_r, y_mean_r, y_std_r)
    gp_i = gp_i.store_yinfo(Y_i, y_mean_i, y_std_i)
    return gp_r, gp_i
