from libgp_Kenny import *
import torch, gpytorch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor, likelihood, kernel, normalize_y=True):
        if normalize_y:
            self.y_stat = {"mean": train_y.mean(), "std": train_y.std()}
            self.y_normalizer = lambda y: (y - self.y_stat["mean"]) / (self.y_stat["std"])
            self.y_denormalizer = lambda y: (y * self.y_stat["std"]) + (self.y_stat["mean"])
        else:
            self.y_normalizer = lambda y: y
            self.y_denormalizer = lambda y: y
        super(GPModel, self).__init__(train_x, self.y_normalizer(train_y), likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

    def forward(self, x):
        if not isinstance(x, torch.Tensor): x = torch.tensor(x).to(device)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def predict(self, x, return_std=False):
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.forward(x)
        ypred = self.y_denormalizer(observed_pred.mean).detach().cpu().numpy()
        prediction = [ypred]
        if return_std: 
            ystd = (observed_pred.variance.sqrt() * self.y_stat["std"]).detach().cpu().numpy()
            prediction.append(ystd)
        return prediction


def train_model_gp_Kenny(train_x, train_y, terms=4, training_iter: int = 100, verbose = False):

    train_x = torch.tensor(train_x, dtype=float).to(device)
    train_y = torch.tensor(train_y).to(device)

    kern_sett = KernelSettings("LF_NSM", nu=0.5, terms=terms, dims=1)
    kernel_r = get_kernel(kern_sett)
    kernel_i = get_kernel(kern_sett)
    likelihood_r = gpytorch.likelihoods.GaussianLikelihood()
    likelihood_i = gpytorch.likelihoods.GaussianLikelihood()
    model_r = GPModel(train_x, train_y.real.float(), likelihood_r, kernel_r, normalize_y=False).to(device)
    model_i = GPModel(train_x, train_y.imag.float(), likelihood_i, kernel_i, normalize_y=False).to(device)

    model_r.train()
    model_i.train()
    likelihood_r.train()
    likelihood_i.train()

    mll_r = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_r, model_r)
    mll_i = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_i, model_i)
    optimizer_r = torch.optim.Adam(model_r.parameters(), lr=0.05)
    optimizer_i = torch.optim.Adam(model_i.parameters(), lr=0.05)
    scheduler_r = None
    scheduler_i = None
    # scheduler_class = torch.optim.lr_scheduler.OneCycleLR
    # scheduler_r = scheduler_class(optimizer_r, max_lr=0.3, total_steps=training_iter, anneal_strategy="linear")
    # scheduler_i = scheduler_class(optimizer_i, max_lr=0.3, total_steps=training_iter, anneal_strategy="linear")
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR
    # scheduler = scheduler(optimizer, T_max=training_iter//7)

    patience = training_iter//10     # number of epochs to wait without improvement

    each_gp_run = [
        [model_r, mll_r, optimizer_r, scheduler_r],
        [model_i, mll_i, optimizer_i, scheduler_i],
    ]
    for model, mll, optimizer, scheduler in each_gp_run:
        best_loss = float("inf")
        patience_counter = 0
        for i in range(training_iter):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y).sum()
            
            it = f"({i+1}/{training_iter})"
            tab = " "*6
            lr = f"LR: {optimizer.param_groups[0]['lr']:.4f}"
            cost = f"loss: {loss:.4f}"
            if verbose:
                print(f"Training.... {it}{tab}{lr}{tab}{cost}", end="\r", flush=True)

            loss.backward()
            optimizer.step()
            # scheduler.step()

            # ---- Early stopping check ----
            if loss.item() < best_loss - 1e-6:  # tolerance to avoid floating point noise
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                if verbose:
                    print(f"\nEarly stopping at iteration {i+1}, best loss={best_loss:.4f}")
                break

        if verbose:
            print()
    if verbose:
        print()
    return model_r, model_i