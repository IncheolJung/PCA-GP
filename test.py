import torch
import gpytorch
# Assuming libgp_Kenny contains KernelSettings and get_kernel
# from libgp_Kenny import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel=None, normalize_y=True):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean().double().to(device)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        ).double().to(device)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).double().to(device)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def predict(self, x, return_std=False):
        self.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # The forward pass will handle reshaping the prediction data
            observed_pred = self.likelihood(self.forward(x))
        ypred = observed_pred.mean.detach()
        prediction = [ypred.cpu().numpy()]
        if return_std:
            ystd = observed_pred.variance.sqrt().detach()
            prediction.append(ystd.cpu().numpy())
        return prediction



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
    
    def train(self, *args):
        self.mean_module.train()
        self.covar_module.train()
        self.likelihood.train()
        return super(GPModel, self).train(*args)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).double().to(device)
            # x = torch.tensor(x, dtype=torch.float64).to(device)
            
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def predict(self, x, return_std=False):
        x = torch.from_numpy(x).double().to(device)
        self.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # The forward pass will handle reshaping the prediction data
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

    # ======== Likelihood & Model ========
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = GPModel(train_x, train_y, likelihood, None, normalize_y).to(device)

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

# Example usage (you can replace this with your actual data)
if __name__ == '__main__':
    # Create some sample complex data
    train_x = torch.linspace(0, 4, 100)
    train_x_np = train_x.numpy()
    # A sine wave for the real part and a cosine for the imaginary part
    train_y_real = torch.sin(train_x * (2 * torch.pi)) + torch.randn(train_x.shape) * 0.1
    train_y_imag = torch.cos(train_x * (2 * torch.pi)) + torch.randn(train_x.shape) * 0.1
    train_y_np = train_y_real.numpy() + 1j * train_y_imag.numpy()

    train_x_np = train_x_np[:, None]
    train_y_np = train_y_np[:, None]

    print(f"Training on {train_x_np.shape[0]} data points.")

    # Train the models
    model_real, model_imag = train_model_gp_Kenny(
        train_x_np, train_y_np, training_iter=500, 
        verbose=True, normalize_y=True
        )

    # Make predictions
    test_x_np = torch.linspace(-1, 5, 100).numpy()
    pred_y_real, std_y_real = model_real.predict(test_x_np, return_std=True)
    pred_y_imag, std_y_imag = model_imag.predict(test_x_np, return_std=True)

    print("\nPrediction complete.")

    # Optional: Plotting the results
    try:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Plot Real Part
        ax1.plot(train_x_np, train_y_np.real, 'k.', label='Training Data')
        ax1.plot(test_x_np, pred_y_real, 'b-', label='Prediction Mean')
        ax1.fill_between(test_x_np.flatten(), (pred_y_real - 2 * std_y_real).flatten(), (pred_y_real + 2 * std_y_real).flatten(), alpha=0.2, color='blue', label='95% Confidence')
        ax1.set_title('Real Part')
        ax1.legend()
        ax1.grid(True)

        # Plot Imaginary Part
        ax2.plot(train_x_np, train_y_np.imag, 'k.', label='Training Data')
        ax2.plot(test_x_np, pred_y_imag, 'r-', label='Prediction Mean')
        ax2.fill_between(test_x_np.flatten(), (pred_y_imag - 2 * std_y_imag).flatten(), (pred_y_imag + 2 * std_y_imag).flatten(), alpha=0.2, color='red', label='95% Confidence')
        ax2.set_title('Imaginary Part')
        ax2.set_xlabel('Input (x)')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

    except ImportError:
        print("\nMatplotlib not found. Skipping plot.")
    except Exception as e:
        print(f"An error occurred during plotting: {e}")

