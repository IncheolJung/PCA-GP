from __future__ import annotations
from abc import ABC, abstractmethod
from collections import namedtuple

import math, torch, numpy as np
import gpytorch as gpy
from gpytorch.constraints import Interval, Positive, LessThan
from gpytorch.kernels import Kernel, ScaleKernel, RBFKernel, RFFKernel, MaternKernel, PeriodicKernel

from .tools import Array, Tensor

class PhaseConstraint(Interval):

    def __init__(
        self, lower_bound=-torch.pi, upper_bound=torch.pi, 
        transform=None, inv_transform=None, initial_value=None
    ):
        if transform is None: transform = lambda x: x
        if inv_transform is None: inv_transform = lambda x: x
        self.lb, self.ub = lower_bound, upper_bound
        self.range = upper_bound - lower_bound
        assert(self.range > 0)  # upper_bound > lower_bound
        super().__init__(
            lower_bound=-math.inf,
            upper_bound=math.inf,
            transform=transform,
            inv_transform=inv_transform,
            initial_value=initial_value,
        )

    def transform(self, x):
        return (x - self.lb) % (self.range) + self.lb

    def inverse_transform(self, y):
        return y   # identity if you just want wrapping

    def check(self, y):
        return torch.isfinite(y)


KernelSettings = namedtuple("KernelSettings", "name nu dims terms", defaults=(None,))
class KFunction(ABC):
    """An abstract class that represents a function to be used for constructing kernels.
    Subclasses must override the `parameters` and `forward` method. Optionally, if there are
    necessary initialization parmeters, the `__init__` method may be overridden as well.
    """
    def __init__(self):
        pass

    @abstractmethod
    def parameters(self):
        raise NotImplementedError()

    @abstractmethod
    def forward(self):
        raise NotImplementedError()


class NSMKernel:
    """
    A factory class for creating a Nonstationary Matern Kernel classes given
    the provided functions.


    """
    def __init__(self, std_func: KFunction, covar_func: KFunction, name: str):
        """
        Args:
            std_func: The local standard deviation function
            covar_func: The local covariance function
            name: Name of the new class to be created.
        """
        d = {
            "has_lengthscale": True,
            "std_func": std_func,
            "covar_func": covar_func,
            "Q": self.Q,
            "forward": self.forward,
        }
        for param in std_func.parameters().keys():
            d[f"std_{param}"] = self.make_property(f"std_{param}")

        for param in covar_func.parameters().keys():
            d[f"covar_{param}"] = self.make_property(f"covar_{param}")

        self.created = type(name, (Kernel,), d)
        self.created.__init__ = self.__nsm_init_maker(self.created)

    @staticmethod
    def __nsm_init_maker(name: str):
        """Makes the init function a NSMKernel class.

        Args:
            name: Name of the class.
        """
        def __init__(self, nu: float, std_constraints: dict = dict(), covar_constraints: dict = dict(), **kwargs):
            """Initialization function for a generated NSMKernel class.

            Args:
                nu: the nu value for the Bessel function. Expects 0.5, 1.5, or 2.5, or Infinity.
                std_constraints: A dictionary for overriding the `std_func` constraints.
                covar_constraints: A dictionary for overriding the 'covar_func' constraints.
                **kwargs: Keyword arguments to be provided to GPyTorch's Kernel class.
            """
            if nu not in {0.5, 1.5, 2.5, float('inf')}:
                raise RuntimeError("nu expected to be 0.5, 1.5, 2.5, or Infinity")

            self.nu = nu
            super(name, self).__init__(**kwargs)

            init_std_func = 0.7
            for param, info in self.std_func.parameters().items():
                shape, constraint = info
                if param in std_constraints: constraint = std_constraints[param]
                self.register_parameter(
                    name=f"raw_std_{param}",
                    parameter=torch.nn.Parameter(init_std_func*torch.rand(*(self.batch_shape + shape), dtype=torch.float64))
                )
                self.register_constraint(f"raw_std_{param}", constraint)

            init_covar_func = 0.3
            for param, info in self.covar_func.parameters().items():
                shape, constraint = info
                if param in covar_constraints: constraint = covar_constraints[param]
                self.register_parameter(
                    name=f"raw_covar_{param}",
                    parameter=torch.nn.Parameter(init_covar_func*torch.rand(*(self.batch_shape + shape), dtype=torch.float64))
                )
                self.register_constraint(f"raw_covar_{param}", constraint)

        return __init__

    @staticmethod
    def make_property(name: str) -> property:
        """Static method which creates a property function to be added to the generated
        NSMKernel class.

        Args:
            name: The name of the property.
        """
        param = f"raw_{name}"
        constraint = f"raw_{name}_constraint"

        @property
        def prop(self) -> None:
            return getattr(self, constraint).transform(getattr(self, param))

        @prop.setter
        def prop(self, value: Tensor) -> None:
            if not torch.is_tensor(value):
                value = torch.as_tensor(value).to(getattr(self, param))

            d = dict()
            d[name] = getattr(self, constraint).inverse_transform(value)
            self.initialize(**d)

        return prop

    @staticmethod
    def Q(self, x1: Tensor, x2: Tensor, M: Tensor) -> Tensor:
        """Q function for the generic NSM class of kernels."""
        M = M.unsqueeze(-1)
        v = (x1.unsqueeze(-2) - x2.unsqueeze(-3)).unsqueeze(-1)
        Q = torch.matmul(v.transpose(-1, -2), v/M).squeeze().clamp_min_(1e-30)
        return Q

    @staticmethod
    def forward(self, x1: Tensor, x2: Tensor, diag: bool = False, **params) -> Tensor:
        """Forward function for the generic NSM class of kernels."""
        # print("Input shapes from kernel.NSMKernel.forward")
        # print(x1.shape, x2.shape, x1.dtype, x2.dtype, diag)
        if diag:
            x_ = x1 / self.lengthscale
            std = self.std_func.forward(x_, self)
            sigma = self.covar_func.forward(x_, self)
            non_stationary = std.pow(2)
            #non_stationary = std1.pow(2) / sigma.prod(dim=-1).pow(1/2)
            norm2 = torch.zeros(non_stationary.shape, device=non_stationary.device)
        else:
            x1_, x2_ = x1 / self.lengthscale, x2 / self.lengthscale
            #x1_, x2_ = x1, x2
            std1 = self.std_func.forward(x1_, self)
            std2 = self.std_func.forward(x2_, self)

            sig1 = self.covar_func.forward(x1_, self)
            sig2 = self.covar_func.forward(x2_, self)
            #std1_, std2_ = std1, std2
            sig1det, sig2det = sig1.prod(dim=-1).pow(1/4), sig2.prod(dim=-1).pow(1/4)
            std1_, std2_ = std1 * sig1det, std2*sig2det

            sigma = (sig1.unsqueeze(-2) + sig2.unsqueeze(-3)) / 2

            sigmadet = sigma.prod(dim=-1).pow(1/2)
            std12 = torch.matmul(std1_.unsqueeze(-1), std2_.unsqueeze(-2))
            non_stationary = std12 / sigmadet

            norm2 = self.Q(x1_, x2_, sigma)
        # Matern
        #distance = self.covar_dist(x1, x2, **params)

        if self.nu == float('inf'):
            matern = torch.exp(-norm2/2)
        else:
            norm = norm2.sqrt_()
            exp_component = torch.exp(-math.sqrt(self.nu * 2) * norm)
            if self.nu == 0.5:
               constant_component = 1
            elif self.nu == 1.5:
                constant_component = (math.sqrt(3) * norm).add(1)
            elif self.nu == 2.5:
                constant_component = (math.sqrt(5) * norm).add(1).add(5.0 / 3.0 * norm2)

            matern = constant_component * exp_component
        return non_stationary * matern

    def __call__(self, *args, **kwargs):
        return self.created(*args, **kwargs)

class Identity(KFunction):
    "A identity function which always returns 1."
    def __init__(self, ard_num_dims: int) -> None:
        self.ard_num_dims = ard_num_dims

    def parameters(self):
        return {}

    def forward(self, x: Tensor, kernel: Kernel):
        return torch.ones((x.shape[0], self.ard_num_dims), dtype=x.dtype, device=x.device)


class LogLinearRegressor(KFunction):
    """A log linear regressor as proposed in (Risser, M. 2015), being used for a standard
    deviation estimator.
    """
    def __init__(self, ard_num_dims: int) -> None:
        self.ard_num_dims = ard_num_dims

    def parameters(self):
        return {
            "weights": ((self.ard_num_dims + 1,), Interval(-100, 101))
        }

    def forward(self, x: Tensor, kernel: Kernel) -> Tensor:
        xp = torch.nn.ConstantPad1d((1, 0), 1.0)(x)
        return torch.exp(torch.matmul(xp, kernel.std_weights)/2)


class DiagonalCovarianceRegressor(KFunction):
    """A local covariance regressor used in (Risser, M. 2015), whose output is strictly
    diagonal, allowing easy inversion for backpropogation.
    """
    def __init__(self, ard_num_dims: int) -> None:
        self.ard_num_dims = ard_num_dims

    def parameters(self):
        return {
            "phi": ((1, self.ard_num_dims), Positive()),
            "gamma": ((self.ard_num_dims, self.ard_num_dims+1), Interval(-100, 100))
        }

    def forward(self, x: Tensor, kernel: Kernel) -> Tensor:
        phi, gamma = kernel.covar_phi, kernel.covar_gamma
        x_ = torch.nn.ConstantPad1d((1, 0), 1.0)(x)
        sig = phi + (x_ @ gamma.T)**2
        return sig

class LocalFourier(KFunction):
    """A standard deviation estimator based on squared cosine terms."""
    def __init__(self, ard_num_dims: int, freqs: int):
        self.ard_num_dims = ard_num_dims
        self.freqs = freqs + 1
        self.normalize = self.freqs**self.ard_num_dims
        # print("report from LocalFourier.__init__")
        # print(self.ard_num_dims, self.freqs, self.normalize)

    def parameters(self):
        return {
            # "amps": ((2, self.normalize - 1,), Positive()),
            "amps": ((2, self.normalize - 1,), Interval(0.0, 50.0)),
            # "phases": ((2, self.normalize - 1,), Interval(-1.5*torch.pi, 1.5*torch.pi)),  # org
            # "phases": ((2, self.normalize - 1,), Interval(-744, 709)),    # roughly numerical limits for exp
            "phases": ((2, self.normalize - 1,), PhaseConstraint(-torch.pi, torch.pi)),  # custom phase constraint
            "period": ((1,), Positive(transform=torch.exp, inv_transform=torch.log)),
            "scale": ((1,), Positive(transform=torch.exp, inv_transform=torch.log)),
            "variance": ((1,), Positive(transform=torch.exp, inv_transform=torch.log)),
        }

    def forward(self, x: Tensor, kernel: Kernel) -> Tensor:
        amps, phases, period, scale = kernel.std_amps, kernel.std_phases, \
                                        kernel.std_period, kernel.std_scale
        variance = kernel.std_variance

        phases /= 100    # make phase less sensitive
        # phases = torch.angle(torch.exp(1j*phases))  # wrap phase
        dims = (torch.arange(self.freqs, dtype=x.dtype,  device=x.device),)*x.shape[-1]
        freqs = torch.cartesian_prod(*dims)[1:].view(-1, x.shape[-1]).transpose(-1, -2) # Skip zero.
        # normalize = self.freqs**self.ard_num_dims
        normalize = self.normalize
        # print("shape from LocalFourier.forward:")
        # print(x.shape, freqs.shape, period.shape, phases.shape)
        # print(dims, freqs)
        sin = torch.sin(torch.pi * torch.matmul(x, freqs)/period + phases[0])/normalize
        cos = torch.cos(torch.pi * torch.matmul(x, freqs)/period + phases[1])/normalize
        summed = torch.matmul(sin, amps[0]) + torch.matmul(cos, amps[1])
        # summed /= 2*len(freqs)
        # kernel.covar_phases = torch.angle(torch.exp(1j*kernel.covar_phases))  # wrap phase
        arg = -scale * torch.relu(summed)
        arg = torch.clamp(arg, min=-30.0, max=30.0)  # prevents overflow
        return variance * torch.exp(arg)

class DiagonalLocalFourier(KFunction):
    """A standard deviation estimator based on squared cosine terms."""
    def __init__(self, ard_num_dims: int, freqs: int):
        self.ard_num_dims = ard_num_dims
        self.freqs = freqs

    def parameters(self):
        return {
            # "amps": ((2, 1, self.ard_num_dims, self.freqs,), Positive()),
            "amps": ((2, 1, self.ard_num_dims, self.freqs,), Interval(0.0, 5.0)),
            # "phases": ((2, 1, self.ard_num_dims, self.freqs,), Interval(-1.5*torch.pi, 1.5*torch.pi)),    # org
            # "phases": ((2, 1, self.ard_num_dims, self.freqs,), Interval(-744, 709)),  # roughly numerical limits for exp
            "phases": ((2, 1, self.ard_num_dims, self.freqs,), PhaseConstraint(-10*torch.pi, 10*torch.pi)),  # custom phase constraint
            "period": ((self.ard_num_dims, 1), Positive(transform=torch.exp, inv_transform=torch.log)),
            "scale": ((1,), Positive(transform=torch.exp, inv_transform=torch.log)),
            "variance": ((1,), Positive(transform=torch.exp, inv_transform=torch.log)),
        }

    def forward(self, x: Tensor, kernel: Kernel) -> Tensor:
        amps, phases, period, scale = kernel.covar_amps, kernel.covar_phases, \
                                        kernel.covar_period, kernel.covar_scale
        variance = kernel.std_variance
        
        phases /= 100    # make phase less sensitive
        # phases = torch.angle(torch.exp(1j*phases))  # wrap phase
        freqs = torch.arange(1, self.freqs+1, device=x.device, dtype=x.dtype).unsqueeze(-2)
        sin = torch.sin((torch.pi/period) * (x.unsqueeze(-1) * freqs) + phases[0])/self.freqs
        cos = torch.cos((torch.pi/period) * (x.unsqueeze(-1) * freqs) + phases[1])/self.freqs
        summed = (sin*amps[0] + cos*amps[1]).sum(dim=-1)
        # kernel.covar_phases = torch.angle(torch.exp(1j*kernel.covar_phases))  # wrap phase
        arg = -scale * torch.relu(summed)
        arg = torch.clamp(arg, min=-30.0, max=30.0)  # prevents overflow
        return variance * torch.exp(arg)

# \Sigma defined in Noack, 2022, Advanced Stationary...
# def forward(self, x1: Tensor, x2: Tensor, kernel: Kernel, diag: bool = False) -> Tensor:
#     s, v = pos_params, params
#     if self.constant: s = torch.expand(3, self.ard_num_dims)

#     p1, p2 = (x1 - v[0]).pow(2).sum(dim=-1), (x2 - v[1]).pow(2).sum(dim=-1) # B x N x 1
#     # B x N x D
#     p1 = torch.matmul(p1.unsqueeze(-1), s[2].unsqueeze(0))
#     p2 = torch.matmul(p2.unsqueeze(-1), s[2].unsqueeze(0))
#     p1, p2 = torch.exp(p1), torch.exp(p2)

#     if diag:
#         p = p1 + p2
#     else:
#         p = p1.unsqueeze(-2) + p2.unsqueeze(-3) # B x N x N x D

#     sigma = s[0] + s[1]*p

#     return sigma

class MultipliedKernels(Kernel):

    has_lengthscale = True

    def __init__(self, kernel1, kernel2, *args, **kwargs) -> None:
        super(MultipliedKernels, self).__init__(*args, **kwargs)
        self.kernel1 = kernel1
        self.kernel2 = kernel2

    def forward(self, x1, x2, diag=False, **params):
        x1_1, x1_2 = x1.T
        x2_1, x2_2 = x2.T
        out1 = self.kernel1(x1_1, x2_1, diag=diag, **params)
        out2 = self.kernel2(x1_2, x2_2, diag=diag, **params)
        return out1 * out2
        # return RBFCovariance.apply(
        #     x1,
        #     x2,
        #     self.lengthscale,
        #     lambda x1, x2: self.covar_dist(x1, x2, square_dist=True, diag=False, **params),
        # )

class StackedKernel(Kernel):

    has_lengthscale = True
    lengthscale_ = 1.0

    def __init__(self, kern_sett: KernelSettings, *args, **kwargs) -> None:
        name, nu, dims, terms = kern_sett
        # active_dims = list(range(dims))
        # print("active_dims given to super():", active_dims)
        super(StackedKernel, self).__init__(*args, active_dims=None, **kwargs)
        kern_sett = name.replace("Stacked_",""), nu, dims, 1
        self.kernels = torch.nn.ModuleList([get_kernel(kern_sett) for _ in range(terms)])
        return None
    
    @property
    def lengthscale(self):
        return self.lengthscale_
    
    @lengthscale.setter
    def lengthscale(self, value):
        # propagate to sub-kernels
        for k in self.kernels:
            if hasattr(k, "lengthscale") and k.lengthscale is not None:
                k.lengthscale = value
        self.lengthscale_ = value
        return 0

    def forward(self, x1, x2, diag=False, **params):
        return sum(k(x1, x2, diag=False, **params) for k in self.kernels) / len(self.kernels)

def get_kernel(sett: KernelSettings) -> _Kernel:
    name, nu, dims, terms = sett
    # active_dims = list(range(dims))
    match name:
        case "RBF": return ScaleKernel(RBFKernel(ard_num_dims=dims))
        case "RBFP": return ScaleKernel(RBFKernel(ard_num_dims=dims)) * PeriodicKernel(ard_num_dims=dims)
        case "Matern": return ScaleKernel(MaternKernel(nu, ard_num_dims=dims))
        case "MaternPer": return ScaleKernel(MaternKernel(nu, ard_num_dims=dims)) * PeriodicKernel(ard_num_dims=dims)
        case "REG_NSM": return ScaleKernel(NSMKernel(LogLinearRegressor(dims), DiagonalCovarianceRegressor(dims), name)(nu, ard_num_dims=dims))
        # case "LF_NSM": return ScaleKernel(
        #     NSMKernel(
        #         LocalFourier(dims, terms), DiagonalLocalFourier(dims, terms), name)(nu, ard_num_dims=dims)
        #     ) * PeriodicKernel(ard_num_dims=dims)
        case "LF_NSM": return ScaleKernel(NSMKernel(LocalFourier(dims, terms), DiagonalLocalFourier(dims, terms), name)(nu, ard_num_dims=dims)) * PeriodicKernel(ard_num_dims=dims)
        case "LF_NSRBF": return ScaleKernel(NSMKernel(LocalFourier(dims, terms), DiagonalLocalFourier(dims, terms), name)(np.inf, ard_num_dims=dims)) * PeriodicKernel(ard_num_dims=dims)
        case "LF_NSM_noPer": return ScaleKernel(NSMKernel(LocalFourier(dims, terms), DiagonalLocalFourier(dims, terms), name)(nu, ard_num_dims=dims))
        case "RBFP_Matern": return ScaleKernel(MultipliedKernels(
            ScaleKernel(RBFKernel(ard_num_dims=dims-1)) * PeriodicKernel(ard_num_dims=dims-1), 
            ScaleKernel(MaternKernel(nu, ard_num_dims=dims-1))
            ))
        case "RBFP_MaternPer": return ScaleKernel(MultipliedKernels(
            ScaleKernel(RBFKernel(ard_num_dims=dims-1)) * PeriodicKernel(ard_num_dims=dims-1), 
            ScaleKernel(MaternKernel(nu, ard_num_dims=dims-1)) * PeriodicKernel(ard_num_dims=dims-1)
            ))
        case "RBFP_RBFP": return ScaleKernel(MultipliedKernels(
            ScaleKernel(RBFKernel(ard_num_dims=dims-1)) * PeriodicKernel(ard_num_dims=dims-1), 
            ScaleKernel(RBFKernel(ard_num_dims=dims-1)) * PeriodicKernel(ard_num_dims=dims-1)
            ))
        case "RBFP_LF_NSM": return ScaleKernel(MultipliedKernels(
            ScaleKernel(RBFKernel(ard_num_dims=dims-1)) * PeriodicKernel(ard_num_dims=dims-1), 
            ScaleKernel(NSMKernel(LocalFourier(dims-1, terms), DiagonalLocalFourier(dims-1, terms), name)(nu, ard_num_dims=dims-1)) * PeriodicKernel(ard_num_dims=dims-1)
            ))
        case "RBFP_LF_NSRBF": return ScaleKernel(MultipliedKernels(
            ScaleKernel(RBFKernel(ard_num_dims=dims-1)) * PeriodicKernel(ard_num_dims=dims-1), 
            ScaleKernel(NSMKernel(LocalFourier(dims-1, terms), DiagonalLocalFourier(dims-1, terms), name)(np.inf, ard_num_dims=dims-1)) * PeriodicKernel(ard_num_dims=dims-1)
            ))
        case "RBF_LF_NSM": return ScaleKernel(MultipliedKernels(
            ScaleKernel(RBFKernel(ard_num_dims=dims-1)), 
            ScaleKernel(NSMKernel(LocalFourier(dims-1, terms), DiagonalLocalFourier(dims-1, terms), name)(nu, ard_num_dims=dims-1)) * PeriodicKernel(ard_num_dims=dims-1)
            ))
        case "RBFPP": return ScaleKernel(RBFKernel(ard_num_dims=dims)) * (ScaleKernel(PeriodicKernel(ard_num_dims=dims)) + ScaleKernel(PeriodicKernel(ard_num_dims=dims)))
        case "Stacked_RBF": return ScaleKernel(StackedKernel(sett))
        case "Stacked_RBFP": return ScaleKernel(StackedKernel(sett))
        case "Stacked_LF_NSM": return ScaleKernel(StackedKernel(sett))
        case _: raise RuntimeError(f"Kernel name '{name}' doesn't match anything!")
