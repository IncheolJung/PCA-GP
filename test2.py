import torch
import gpytorch
from matplotlib import pyplot as plt

# Check device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ======== Define GP Model ========
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)\
    
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
            self.y_stat = {"mean": train_y.mean(), "std": train_y.std()}
            self.y_normalizer   = lambda y: (y - self.y_stat["mean"]) / (self.y_stat["std"] + 1e-8) # Add epsilon for stability
            self.y_denormalizer = lambda y: (y * self.y_stat["std"]) + self.y_stat["mean"]
            self.y_denorm_std   = lambda y: (y * self.y_stat["std"])
        else:
            self.y_normalizer   = lambda y: y
            self.y_denormalizer = lambda y: y
            self.y_denorm_std   = lambda y: y
        
        # Pass train_x directly; reshaping is handled in forward for all inputs.
        super(GPModel, self).__init__(
            train_x, self.y_normalizer(train_y), likelihood
            )
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )
        self.likelihood = likelihood

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).double().to(device)
            # x = torch.tensor(x, dtype=torch.float64).to(device)
            
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def predict(self, x, likelihood, return_std=False):
        self.eval()
        likelihood.eval()
        # self.likelihood.eval()
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).double().to(next(self.parameters()).device)

        print("Predict Input shape:", x.shape)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = likelihood(self.__call__(x))

        ypred = observed_pred.mean.detach()
        prediction = [ypred.cpu().numpy()]

        if return_std:
            ystd = observed_pred.variance.sqrt().detach()
            prediction.append(ystd.cpu().numpy())

        return prediction

# ======== Example Data ========
# Let's create a simple 1D regression dataset
train_x = torch.linspace(0, 1, 100).to(device)
train_y = torch.sin(train_x * (2 * torch.pi)) + 0.2 * torch.randn(train_x.size()).to(device)

# ======== Likelihood & Model ========
likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
model = GPModel(train_x, train_y, likelihood).to(device)

# ======== Training ========
model.train()
likelihood.train()

optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

print("Input shapes:", train_x.shape, train_y.shape)
training_iter = 100
for i in range(training_iter):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    # print(loss)
    loss.backward()
    if (i+1) % 10 == 0:
        print(f"Iter {i+1}/{training_iter} - Loss: {loss.item():.3f}")
    optimizer.step()

# ======== Prediction ========
model.eval()
likelihood.eval()

# Test points
test_x = torch.linspace(0, 1, 200).to(device)
pred_y, std_y = model.predict(test_x, likelihood, return_std=True)
# with torch.no_grad(), gpytorch.settings.fast_pred_var():
#     observed_pred = likelihood(model(test_x))
#     mean = observed_pred.mean
#     lower, upper = observed_pred.confidence_region()
lower, upper = pred_y - std_y, pred_y + std_y

# ======== Plot ========
plt.figure(figsize=(8,5))
plt.plot(train_x.cpu(), train_y.cpu(), 'k*', label='Training Data')
plt.plot(test_x.cpu(), pred_y, 'b', label='Predicted Mean')
plt.fill_between(test_x.cpu(), lower, upper, alpha=0.3, label='Confidence Interval')
plt.legend()
plt.show()
