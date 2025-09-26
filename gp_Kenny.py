from libgp_Kenny import *
import torch, gpytorch
import warnings
from gpytorch.utils.warnings import GPInputWarning
from torch.nn.utils import clip_grad_norm_
from copy import deepcopy

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'


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
                "mean": train_y.mean(dim=0, keepdim=True), 
                "std":  train_y.std (dim=0, keepdim=True),
                "min":  train_y.min (dim=0, keepdim=True), 
                "max":  train_y.max (dim=0, keepdim=True)
                }
            eps = 1e-12  # or slightly larger depending on scale
            ystd = torch.clamp(self.y_stat["std"], min=eps)
            self.y_normalizer   = lambda y: (y - self.y_stat["mean"]) / ystd # Add epsilon for stability
            self.y_denormalizer = lambda y: (y * self.y_stat["std"]) + self.y_stat["mean"]
            self.y_denorm_std   = lambda y: (y * self.y_stat["std"])
            # range_y = self.y_stat["max"] - self.y_stat["min"]
            # range_y = torch.clamp(range_y, min=eps)
            # self.y_normalizer   = lambda y: (y - self.y_stat["min"]) / range_y
            # self.y_denormalizer = lambda y: (y * range_y) + self.y_stat["min"]
            # self.y_denorm_std   = lambda y: (y * range_y)
        else:
            self.y_normalizer   = lambda y: y
            self.y_denormalizer = lambda y: y
            self.y_denorm_std   = lambda y: y
        super(GPModel, self).__init__(
            train_x, self.y_normalizer(train_y), likelihood
            )
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel
        # self.covar_module = gpytorch.kernels.ScaleKernel(
        #     gpytorch.kernels.RBFKernel()
        # )
        self.likelihood = likelihood
        self.distribute = gpytorch.distributions.MultivariateNormal
    
    def make_multitask(self, n_tasks):
        self.mean_module = gpytorch.means.MultitaskMean(
            self.mean_module, num_tasks=n_tasks
            # deepcopy(self.mean_module), num_tasks=n_tasks
        ).double()
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            self.covar_module, num_tasks=n_tasks, rank=1
            # deepcopy(self.covar_module), num_tasks=n_tasks, rank=1
        ).double()
        self.distribute = gpytorch.distributions.MultitaskMultivariateNormal
        return 0

    def forward(self, x):
        # print(f"Report from gp_Kenny.GPModel.forward")
        # print("Input shape:", x.shape)
        # print("active_dims:", self.covar_module.active_dims)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        # print(f"Report from gp_Kenny.GPModel.forward")
        # print("Output shape:", mean_x.shape, covar_x.shape)
        return self.distribute(mean_x, covar_x)

    def predict(self, x, return_std=False):
        # print(f"Report from gp_Kenny.GPModel.predict")
        # print("Input shape:", x.shape)
        if x.ndim == 1: x = x[:, None]
        x = torch.from_numpy(x).double().to(device)
        self.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var(), warnings.catch_warnings():
            # The forward pass will handle reshaping the prediction data
            warnings.simplefilter("ignore", GPInputWarning)
            observed_pred = self.likelihood(self.__call__(x))
            # observed_pred = self.__call__(x)
        ypred = self.y_denormalizer(observed_pred.mean)
        prediction = [ypred.detach().cpu().numpy()]
        if return_std:
            ystd = self.y_denorm_std(observed_pred.variance.sqrt())
            prediction.append(ystd.detach().cpu().numpy())
        # print(f"Report from gp_Kenny.GPModel.predict")
        # print("Output shape:", ypred.shape)
        return prediction


def train_model_per_batch(
        train_x, train_y, terms, training_iter, 
        verbose, normalize_y, dims
        ):
    from numpy import log10

    train_x = torch.squeeze(train_x, dim=-1).double().to(device)
    train_y = torch.squeeze(train_y, dim=-1).double().to(device)
    if train_x.ndim != train_y.ndim:
        if (train_x.ndim==1 and train_y.ndim==2):
            train_x = train_x.view(-1, 1)

    # print(f"Report from gp_Kenny.train_model_per_batch")
    # print("Input shapes:", train_x.shape, train_y.shape)

    # kern_sett = KernelSettings("Stacked_RBFP", nu=torch.inf, terms=terms, dims=dims)
    # kern_sett = KernelSettings("Stacked_LF_NSM", nu=0.5, terms=terms, dims=dims)
    kern_sett = KernelSettings("LF_NSM", nu=0.5, terms=terms, dims=dims)
    # kern_sett = KernelSettings("RBFP", nu=0.5, terms=terms, dims=dims)
    # kern_sett = KernelSettings("RBF", nu=0.5, terms=terms, dims=dims)
    kernel = get_kernel(kern_sett).double().to(device)
    # kernel = gpytorch.kernels.ScaleKernel(
    #     gpytorch.kernels.RBFKernel()
    #     ).double().to(device)
    # if hasattr(kernel, "base_kernel"):
    #     if hasattr(kernel.base_kernel, "lengthscale"):
    #         kernel.base_kernel.lengthscale = train_x.std()  # for RBF-type kernels
    #     if hasattr(kernel.base_kernel, "outputscale"):
    #         kernel.base_kernel.outputscale = train_y.std() ** 2
    # if hasattr(kernel, "lengthscale"):
    #     kernel.lengthscale = train_x.std()  # for RBF-type kernels
    # if hasattr(kernel, "outputscale"):
    #     kernel.outputscale = train_y.std() ** 2

    lower_bound = 1e-14
    if train_y.ndim == 2 and train_y.shape[-1] > 1:
        # likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
        #     lower_bound * torch.ones_like(train_y)
        # ).double().to(device)
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(lower_bound),
            num_tasks=train_y.shape[-1],
        ).double().to(device)
        likelihood.noise = lower_bound # for example
        tasknoise = 1e-12 * torch.ones_like(likelihood.task_noises)
        likelihood.task_noises = tasknoise
        # likelihood.raw_noise.detach_()          # Freeze noise
        # likelihood.raw_task_noises.detach_()    # Freeze noise
    else:
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(lower_bound)
        ).double().to(device)
        likelihood.noise_covar.initialize(noise=lower_bound)
        likelihood.noise_covar.raw_noise.requires_grad_(False)  # freeze
    # likelihood.noise = max(lower_bound, 1e-3 * train_y.std() ** 2)

    model = GPModel(
        train_x, train_y, likelihood, kernel, normalize_y=normalize_y
        ).double().to(device)
    
    if train_y.ndim == 2 and train_y.shape[-1] > 1:
        # print(f"Setting multitask {train_y.shape[-1]}")
        model.make_multitask(train_y.shape[-1])
    
    # model.make_multitask(train_y.shape[-1])
    
    model.train()
    likelihood.train()

    # all_params = list(model.named_parameters())
    # seen = set()
    # def unique(params): 
    #     out = []
    #     for p in params:
    #         if id(p) not in seen:
    #             out.append(p); seen.add(id(p))
    #     return out

    # kernel_params = [p for n,p in all_params if "covar" in n or "kernel" in n]
    # likelihood_params = [p for n,p in all_params if "likelihood" in n or "noise" in n]
    # coregion_params = [p for n,p in all_params if "coreg" in n or "task" in n]

    # optimizer = torch.optim.Adam([
    #         {'params': unique(kernel_params),     'lr': 5e-2},
    #         {'params': unique(coregion_params),   'lr': 2e-2},  # often sensitive
    #         {'params': unique(likelihood_params), 'lr': 1e-1},  # let noise move faster
    #     ], betas=(0.9,0.999), weight_decay=1e-8, amsgrad=False
    # )
    # optimizer = torch.optim.Adam([
    #     {'params': unique(kernel_params),     'lr': 1e-2},
    #     {'params': unique(coregion_params),   'lr': 5e-3},
    #     {'params': unique(likelihood_params), 'lr': 1e-2},
    # ])
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.05, weight_decay=0
    )
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(
        likelihood, model
    ).to(device)
    # scheduler = None
    # scheduler_class = torch.optim.lr_scheduler.OneCycleLR
    # scheduler = scheduler_class(optimizer, max_lr=0.2, total_steps=training_iter, anneal_strategy="linear")
    scheduler_class = torch.optim.lr_scheduler.CosineAnnealingLR
    scheduler = scheduler_class(
        optimizer, eta_min=1e-6, T_max=training_iter//3)
    # scheduler_class = torch.optim.lr_scheduler.ReduceLROnPlateau
    # scheduler = scheduler_class(
    #     optimizer, mode="min", factor=0.5, patience=5, cooldown=3, 
    #     threshold = 1e-4, threshold_mode = 'rel', min_lr = 1e-6
    # )

    # number of epochs to wait without improvement (Early stop cond)
    # patience = 20
    patience = int(training_iter/4)
    # patience = 10 * min(10, int(log10(training_iter))) + 5

    best_loss = float("inf")
    patience_counter = 0
    tab = " "*2
    for i in range(training_iter):
        optimizer.zero_grad()
        # with gpytorch.settings.cholesky_jitter(1e-4):
        output = model(train_x)
        # print("Shapes during Training:", train_x.shape, output.mean.shape, train_y.shape, )
        loss = -mll(output, train_y)
        
        if verbose and (i % 10 == 0 or i == training_iter-1):
            it = f"({i+1}/{training_iter})"
            lr = f"LR: {optimizer.param_groups[0]['lr']:.4f}"
            cost = f"loss: {loss:.4f}  noise: [{torch.min(likelihood.task_noises):.2e}, {torch.max(likelihood.task_noises):.2e}]"
            print(f"Training.... {it}{tab}{lr}{tab}{cost}", end="\r", flush=True)

        # ---- Early stopping check ----
        if loss.item() < best_loss - 1e-6:  # tolerance to avoid floating point noise
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            if verbose:
                it = f"({i+1}/{training_iter})"
                lr = f"LR: {scheduler.get_last_lr()[0]:.4f}"
                # lr = f"LR: {optimizer.param_groups[0]['lr']:.4f}"
                best_cost = f"loss: {best_loss:.4f}  noise: [{torch.min(likelihood.task_noises):.2e}, {torch.max(likelihood.task_noises):.2e}]"
                end_phrase = f"..EarlyStop! {it}{tab}{lr}{tab}{best_cost}{tab}"
                print(end_phrase, flush=True, end='\033[K\r')
            break
        # ---- Early stopping check ----

        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        scheduler.step()
        # scheduler.step(loss.item())     # for ReduceLROnPlateau

    if verbose and patience_counter < patience:
        it = f"({i+1}/{training_iter})"
        lr = f"LR: {scheduler.get_last_lr()[0]:.4f}"
        # lr = f"LR: {optimizer.param_groups[0]['lr']:.4f}"
        cost = f"loss: {loss:.4f}{tab}noise: [{torch.min(likelihood.task_noises):.2e}, {torch.max(likelihood.task_noises):.2e}]"
        end_phrase = f".....Finish! {it}{tab}{lr}{tab}{cost}{tab}"
        print(end_phrase, flush=True, end='\033[K\r')

    # ------- Fine-tuning ------- 
    fine_tuning_iter = 50
    # fine_tuning_iter = training_iter - i + 21
    # fine_tuning_lr   = 2*scheduler.get_last_lr()[0]    # 2*last_lr
    fine_tuning_lr   = 0.01
    # end_phrase = f"Training.... "
    # finetuner = torch.optim.Rprop(model.parameters(), lr=fine_tuning_lr)
    finetuner = torch.optim.LBFGS(
        model.parameters(), lr=fine_tuning_lr, 
        max_iter=fine_tuning_iter,
        history_size=10, line_search_fn="strong_wolfe"
        )
    def closure():
        finetuner.zero_grad()
        # with gpytorch.settings.cholesky_jitter(1e-4):
        output = model(train_x)
        loss = -mll(output, train_y).sum()
        loss.backward()
        if verbose:
            if loss.item()>1e3: loss_stdout = f"{loss.item():.4e}"
            else: loss_stdout = f"{loss.item():.4f}"
            print(end_phrase, f"LBFGS loss: {loss_stdout}", 
                  sep=tab, end='\033[K\r', flush=True)
        return loss
    finetuner.step(closure)
    # ------- Fine-tuning ------- 

    if verbose:
        print()

    return model


def train_model_gp_Kenny(
        train_x, train_y, terms=4, training_iter: int = 100, 
        verbose = False, normalize_y=True, dims = 1
        ):
    """
    Trains separate GP models for the real and imaginary parts of complex data.
    """

    if train_x.ndim == 1: train_x = train_x[:, None]
    if train_y.ndim == 1: train_y = train_y[:, None]

    train_x_tensor = torch.tensor(
        train_x, dtype=torch.float64, requires_grad=True
        ).to(device)
    train_y_tensor_r = torch.tensor(
        train_y.real, dtype=torch.float64, requires_grad=True
        ).to(device)
    train_y_tensor_i = torch.tensor(
        train_y.imag, dtype=torch.float64, requires_grad=True
        ).to(device)

    train_batch = [
        (train_x_tensor, train_y_tensor_r),
        (train_x_tensor, train_y_tensor_i),
    ]

    args = (terms, training_iter, verbose, normalize_y, dims)
    model_r, model_i = [train_model_per_batch(*b, *args) for b in train_batch]

    return model_r, model_i