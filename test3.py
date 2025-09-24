import torch
import gpytorch

train_x = torch.linspace(0, 1, 3)[:, None]        # (3,1)
train_y = torch.stack([torch.sin(train_x[:, 0]),
                       torch.cos(train_x[:, 0])], dim=-1)  # (3,2)

class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, n_tasks):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=n_tasks
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=n_tasks, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)              # (n, n_tasks)
        covar_x = self.covar_module(x)            # block covariance
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

n_tasks = 2
likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=n_tasks)
model = MultitaskGPModel(train_x, train_y, likelihood, n_tasks)

mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

output = model(train_x)
loss = -mll(output, train_y).sum()   # ✅ works
