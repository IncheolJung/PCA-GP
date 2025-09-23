from libgp_Kenny import *
import torch, gpytorch
import warnings
from gpytorch.utils.warnings import GPInputWarning

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class GPModel(gpytorch.models.ExactGP):
    """
    Gaussian Process model class.
    Handles Y-value normalization and prediction logic.
    """
    
    def __init__(
            self,
            train_x: torch.Tensor, train_y: torch.Tensor,
            likelihood, kernel=None, normalize_y=True):
        if normalize_y and train_y.numel() > 1:
            self.y_stat = {
                "mean": train_y.mean(), "std": train_y.std(),
                "min":  train_y.mean(), "max": train_y.std()
                }
            eps = 1e-12  # or slightly larger depending on scale
            range_y = self.y_stat["max"] - self.y_stat["min"]
            range_y = torch.clamp(range_y, min=eps)
            # self.y_normalizer   = lambda y: (y - self.y_stat["mean"]) / (self.y_stat["std"] + 1e-8) # Add epsilon for stability
            # self.y_denormalizer = lambda y: (y * self.y_stat["std"]) + self.y_stat["mean"]
            # self.y_denorm_std   = lambda y: (y * self.y_stat["std"])
            self.y_normalizer   = lambda y: (y - self.y_stat["min"]) / range_y # Add epsilon for stability
            self.y_denormalizer = lambda y: (y * range_y) + self.y_stat["min"]
            self.y_denorm_std   = lambda y: (y * range_y)
        else:
            self.y_normalizer   = lambda y: y
            self.y_denormalizer = lambda y: y
            self.y_denorm_std   = lambda y: y
        super(GPModel, self).__init__(
            train_x, self.y_normalizer(train_y), likelihood
            )
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )
        self.likelihood = likelihood
    
    def train(self, *args):
        self.mean_module.train()
        self.covar_module.train()
        self.likelihood.train()
        return super(GPModel, self).train(*args)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).view(-1,1).double().to(device)
            # x = torch.tensor(x, dtype=torch.float64).to(device)
            
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def predict(self, x, return_std=False):
        x = torch.from_numpy(x).view(-1,1).double().to(device)
        self.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var(), warnings.catch_warnings():
            # The forward pass will handle reshaping the prediction data
            warnings.simplefilter("ignore", GPInputWarning)
            observed_pred = self.likelihood(self.__call__(x))
        ypred = self.y_denormalizer(observed_pred.mean)
        prediction = [ypred.detach().cpu().numpy()]
        if return_std:
            ystd = self.y_denorm_std(observed_pred.variance.sqrt())
            prediction.append(ystd.detach().cpu().numpy())
        return prediction


def train_model_per_batch(
        train_x, train_y, terms=4, 
        training_iter: int = 1000, verbose = False, normalize_y=True
        ):
    from numpy import log10

    train_x = torch.squeeze(train_x, dim=-1).double().to(device)
    train_y = torch.squeeze(train_y, dim=-1).double().to(device)

    kern_sett = KernelSettings("Stacked_RBFP", nu=0.5, terms=terms, dims=1)
    # kern_sett = KernelSettings("Stacked_LF_NSM", nu=0.5, terms=terms, dims=1)
    # kern_sett = KernelSettings("LF_NSM", nu=0.5, terms=terms, dims=1)
    # kern_sett = KernelSettings("RBFP", nu=0.5, terms=terms, dims=1)
    # kern_sett = KernelSettings("RBF", nu=0.5, terms=terms, dims=1)
    kernel = get_kernel(kern_sett).double().to(device)
    # kernel = gpytorch.kernels.ScaleKernel(
    #     gpytorch.kernels.RBFKernel()
    #     ).double().to(device)
    if hasattr(kernel.base_kernel, "lengthscale"):
        kernel.base_kernel.lengthscale = train_x.std()  # for RBF-type kernels
    if hasattr(kernel.base_kernel, "outputscale"):
        kernel.base_kernel.outputscale = train_y.std() ** 2

    lower_bound = 1e-8
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.GreaterThan(lower_bound)
    ).double().to(device)
    likelihood.noise = max(lower_bound, 1e-2 * train_y.std() ** 2)

    model = GPModel(
        train_x, train_y, likelihood, kernel, normalize_y=normalize_y
        ).double().to(device)

    model.train()
    likelihood.train()

    # number of epochs to wait without improvement
    patience = training_iter//2
    # patience = 3 * max(10, int(log10(training_iter))) + 5

    optimizer = torch.optim.Adam(model.parameters(), lr=0.3)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model).to(device)
    # scheduler = None
    # scheduler_class = torch.optim.lr_scheduler.OneCycleLR
    # scheduler = scheduler_class(optimizer, max_lr=0.4, total_steps=training_iter, anneal_strategy="linear")
    scheduler_class = torch.optim.lr_scheduler.CosineAnnealingLR
    scheduler = scheduler_class(optimizer, T_max=training_iter)
    # scheduler_class = torch.optim.lr_scheduler.ReduceLROnPlateau
    # scheduler = scheduler_class(optimizer, patience=int(patience/4), factor=0.3)

    best_loss = float("inf")
    patience_counter = 0
    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y).sum()

        # print("\n\ntrain_x:", train_x, sep="\n", end="\n")
        # print("\n\ntrain_y_each:", train_y_each, sep="\n", end="\n")
        
        if verbose and (i % 10 == 0 or i == training_iter-1):
            it = f"({i+1}/{training_iter})"
            tab = " "*6
            lr = f"LR: {optimizer.param_groups[0]['lr']:.4f}"
            cost = f"loss: {loss:.4f}"
            print(f"Training.... {it}{tab}{lr}{tab}{cost}", end="\r", flush=True)

        loss.backward()
        optimizer.step()
        scheduler.step()
        # scheduler.step(loss.item())     # for ReduceLROnPlateau

        # ---- Early stopping check ----
        if loss.item() < best_loss - 1e-6:  # tolerance to avoid floating point noise
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            if verbose:
                best_cost = f"loss: {best_loss:.4f}"
                end_phrase = f"..EarlyStop! {it}{tab}{lr}{tab}{best_cost}{tab}"
                print(end_phrase, flush=True, end='\033[K\r')
            break
    if verbose and patience_counter < patience:
        cost = f"loss: {loss:.4f}"
        end_phrase = f".....Finish! {it}{tab}{lr}{tab}{cost}{tab}"
        print(end_phrase, flush=True, end='\033[K\r')

    # Fine-tuning
    fine_tuning_iter = 10
    # finetuner = torch.optim.Rprop(model.parameters(), lr=0.01)
    finetuner = torch.optim.LBFGS(
        model.parameters(), lr=0.01, max_iter=fine_tuning_iter,
        history_size=10, line_search_fn="strong_wolfe"
        )
    def closure():
        finetuner.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y).sum()
        loss.backward()
        if verbose:
            if loss.item()>1e3: loss_stdout = f"{loss.item():.4e}"
            else: loss_stdout = f"{loss.item():.4f}"
            print(end_phrase, f"LBFGS step loss: {loss_stdout}", 
                  sep=" "*4, end='\033[K\r', flush=True)
        return loss
    finetuner.step(closure)

    if verbose:
        print()

    return model


def train_model_gp_Kenny(
        train_x, train_y, terms=4,
        training_iter: int = 1000, verbose = False, normalize_y=True
        ):
    """
    Trains separate GP models for the real and imaginary parts of complex data.
    """

    train_x_tensor = torch.tensor(
        train_x, dtype=torch.float64, requires_grad=True
        ).view(-1,1).to(device)
    train_y_tensor_r = torch.tensor(
        train_y.real, dtype=torch.float64, requires_grad=True
        ).view(-1,1)
    train_y_tensor_i = torch.tensor(
        train_y.imag, dtype=torch.float64, requires_grad=True
        ).view(-1,1)

    train_batch = [
        (train_x_tensor, train_y_tensor_r),
        (train_x_tensor, train_y_tensor_i),
    ]

    args = (terms, training_iter, verbose, normalize_y)
    model_r, model_i = [train_model_per_batch(*b, *args) for b in train_batch]

    return model_r, model_i