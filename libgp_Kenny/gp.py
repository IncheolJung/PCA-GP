from __future__ import annotations
import math, gc, sys
from abc import ABC, abstractmethod
from collections import namedtuple, OrderedDict
from functools import cached_property
from itertools import product
from typing import Union
from pathlib import Path
from timeit import default_timer as timer

import torch, numpy as np
import gpytorch as gpy

from .tools import Array, Tensor, Domain, DataPair, Prediction, color, pred_chunk_size
from .kernel import KernelSettings, get_kernel
from .sampling import Sampler, SamplerSettings


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print("Now working on {}".format(device))

class ExactGPModel(gpy.models.ExactGP):
    def __init__(self,
                 train_x: Union[Array, Tensor],
                 train_y: Union[Array, Tensor],
                 likelihood: Likelihood,
                 kernel: Kernel
    ):
        if isinstance(train_x, torch.Tensor) and torch.is_complex(train_x): 
            raise RuntimeError("Complex train x")
        if isinstance(train_y, torch.Tensor) and torch.is_complex(train_y): 
            raise RuntimeError("Complex train y")
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.kernel_name = kernel
        self.mean_module = gpy.means.ConstantMean()
        self.covar_module = kernel

    def forward(self, x: Tensor):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpy.distributions.MultivariateNormal(mean_x, covar_x)


def root_divisor(n):
    if n < 4: return 1
    div = 1
    for i in range(2, n):
        j = i**2
        if j == n: return i
        if j < n and n % i == 0: div = i
        if j > n: return div

class GaussianProcess:
    domain: Domain
    kern_sett: KernelSettings
    samp_sett: SamplerSettings
    chunk_shape: Tuple[int]

    samplers: Array[Samplers]
    gp_chunks: Array[GaussianProcessChunk]

    loaded: bool
    iter: int

    def __init__(self,
                 domain: Domain,
                 kern_sett: KernelSettings,
                 samp_sett: SamplerSettings,
                 chunk_shape: Optional[Tuple[int]] = None,
                 verbose: bool = False
    ) -> None:
        self.domain = domain
        self.kern_sett = kern_sett
        self.samp_sett = samp_sett
        self.__chunks = True
        self.iter = 1
        self.verbose = verbose

        if chunk_shape is None:
            self.__chunks = False
            chunk_shape = (1,)*self.domain.n
        self.chunk_shape = chunk_shape

        self.__chunk_iter = [range(x) for x in self.chunk_shape]

        # Make each GP chunk.
        self.samplers = np.empty(chunk_shape, dtype=object)
        self.gps = np.empty(chunk_shape, dtype=object)
        domains = self.domain.chunk(chunk_shape)
        num_chunks = self.samplers.size

        for i in product(*self.__chunk_iter):
            dom = domains[i]

            self.samplers[i] = Sampler.from_settings(samp_sett, dom, num_chunks)
            self.samplers[i].initial() # Set sampler to initial state.
            self.gps[i] = GaussianProcessChunk(
                dom, get_kernel(self.kern_sett), self.samplers[i], self.verbose
                )
            self.gps[i].iter = 0
    
    def update_domain(self, new_domain: Domain):
        self.domain = new_domain
        domains = self.domain.chunk(self.chunk_shape)
        for i in product(*self.__chunk_iter):
            # self.gps[i].update_domain(domains[i])
            self.gps[i] = GaussianProcessChunk(
                domains[i], get_kernel(self.kern_sett), self.samplers[i], self.verbose
                )
            self.gps[i].iter = self.iter
        # num_chunks = self.samplers.size
        # for i in product(*self.__chunk_iter):
        #     dom = domains[i]
        #     self.samplers[i] = Sampler.from_settings(self.samp_sett, dom, num_chunks)
        #     self.samplers[i].initial() # Set sampler to initial state.
        #     self.gps[i] = GaussianProcessChunk(dom, get_kernel(self.kern_sett), self.samplers[i])
        #     self.gps[i].iter = 0
        return

    def train(self, iterations: int, rate: float) -> None:
        t_start = timer()
        for i in product(*self.__chunk_iter):
            self.gps[i].train(iterations, rate, f"Chunk {i} | " if self.__chunks else "")

        t_end = timer()
        time = (t_end - t_start)
        if self.verbose: print(f"[{color('OK', '*G')}] Training: {time:.2f}s", flush=True)

    def prediction(self, return_raw: bool = False) -> Prediction:
        if self.verbose: print(f"[{color('—', 'Y')}] Predicting...", flush=True)
        t_start = timer()
        means, vars = np.empty(self.chunk_shape, dtype=object), np.empty(self.chunk_shape, dtype=object)
        for i in product(*self.__chunk_iter):
            means[i], vars[i] = self.gps[i].prediction

        t_end = timer()
        time = round(1000*(t_end - t_start))
        if self.verbose: sys.stdout.write("\033[F\033[K")
        if self.verbose: print(f"[{color('OK', '*G')}] Predicting Done: {time}ms", flush=True)
        
        if return_raw: 
            return Prediction(means, vars)
        else: 
            return Prediction(self.domain.unchunk(means), self.domain.unchunk(vars))

    def samples(self) -> Tuple[Array, Array]:
        all, new = [], []
        for i in product(*self.__chunk_iter):
            all.append(self.gps[i].train_data.x)
            new.append(self.gps[i].train_data.x[-self.samplers[i].add:])

        return np.concatenate(all), np.concatenate(new)

    def next(self) -> None:
        thres = [self.gps[i].train_data.x.shape[0] >= self.samplers[i].end for i in product(*self.__chunk_iter)]
        if np.all(thres): return False
        self.iter += 1

        if self.verbose: print(f"[{color('—', 'Y')}] Calculating Next Samples...", flush=True)
        t_start = timer()
        for i in product(*self.__chunk_iter):
            dom, kern = self.gps[i].domain, get_kernel(self.kern_sett)
            self.samplers[i].next(self.gps[i]) # Set sampler to next active sample.
            self.gps[i] = GaussianProcessChunk(
                dom, kern, self.samplers[i], self.verbose
                )
            self.gps[i].iter = self.iter
            gc.collect()

        t_end = timer()
        time = round(1000*(t_end - t_start))
        if self.verbose: sys.stdout.write("\033[F\033[K")
        if self.verbose: print(f"[{color('OK', '*G')}] Next Samples Calculated: {time}ms", flush=True)
        return True

    def save(self, path: Path) -> None:
        for i in product(*self.__chunk_iter):
            self.gps[i].save(path.with_name(path.name + f"-{i}"))

    def load(self, path: Path) -> None:
        for i in product(*self.__chunk_iter):
            self.gps[i].load(path.with_name(path.name + f"-{i}"))

    def print_params(self) -> None:
        for i in product(*self.__chunk_iter):
            for param_name, param_t in self.gps[i].model.named_parameters():
                with torch.no_grad():
                    param = param_t.cpu().numpy()
                    print(f'GP {i} | Parameter name: {param_name:42} value = {param}')


class GaussianProcessChunk:
    domain: Domain
    sampler: Sampler

    train_data: DataPair
    likelihood: Likelihood
    model: gpy.models.ExactGP


    def __init__(self, 
                 domain: Domain, 
                 kernel: Kernel, 
                 sampler: Sampler,
                 verbose: bool
                 ) -> None:
        self.domain = domain
        self.sampler = sampler
        self.verbose = verbose

        # Initialize sample data for GP to train on through Sampler.
        self.train_data = self.sampler.sample()
        # print("report from GaussianProcessChunk.__init__")
        # print("x.dtype:", self.train_data.x.dtype)
        # print("y.dtype:", self.train_data.y.dtype)
        self.__train_cuda = DataPair(self.domain.normalize(self.train_data.x),
                                     self.train_data.y
                                     ).to(device)

        # Setup the gpytorch model and likelihood with negligible noise.
        self.likelihood = gpy.likelihoods.GaussianLikelihood().to(device)
        self.likelihood.noise = 1e-4
        self.likelihood.noise_covar.raw_noise.requires_grad_(False)
        self.likelihood.double()

        self.model = ExactGPModel(self.__train_cuda.x, self.__train_cuda.y, self.likelihood, kernel).to(device)
        self.model.double()

    def update_domain(self, new_domain: Domain):
        self.domain = new_domain

    def __del__(self) -> None:
        del self.model
        del self.likelihood
        gc.collect()

    def train(self, iterations: int, lr: float, chunk: str = "", batch_size: int = 5) -> None:
        self.model.train()
        self.likelihood.train()

        iter = iterations//batch_size
        if iterations%batch_size: iter += 1

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        mll = gpy.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iter//5, eta_min=1e-4)
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=iter)

        #num_down = 0
        #last_loss = 1e10
        i_cur = 0
        for ii in range(iter):
            for jj in range(batch_size):
                optimizer.zero_grad()
                output = self.model(self.__train_cuda.x)
                loss = -mll(output, self.__train_cuda.y)
                loss.backward()

                # diff = loss.item() - last_loss
                # if diff >= 0 and diff <= 0.001:
                #     num_down += 1
                # else:
                #     num_down = 0
                # last_loss = loss.item()

                # if num_down == 1:
                #     for g in optimizer.param_groups:
                #         g['lr'] *= 0.8
                
                hashes = int(21 * i_cur / iterations)
                if self.verbose: sys.stdout.write("\033[K")
                if self.verbose:
                    print(
                        f'{chunk}Training {i_cur+1:04}/{iterations:04} |{"#"*hashes}{" "*(20-hashes)}|'
                        + f" Loss: {loss.item():.3f} Rate: {optimizer.param_groups[0]['lr']:.3f}",
                        flush=True,
                        end = "\r",
                        )
                i_cur += 1
                optimizer.step()
            # scheduler.step()
        if self.verbose: print()

    @property
    # @cached_property
    def prediction(self) -> Prediction:
        # Get into evaluation (predictive posterior) mode
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpy.settings.fast_pred_var():
            CHUNK = pred_chunk_size.value()
            x_flat = self.domain.normalize(torch.from_numpy(self.domain.raveled)).to(device)
            size = x_flat.shape[0]

            for i in range(math.ceil(size/CHUNK)):
                xi = x_flat[i*CHUNK:(i+1)*CHUNK]
                # print("xi.dtype from gp.GaussianProcessChunk.predictio:n", xi.dtype)
                validate = self.model(xi.double()) # We have no noise, so likelihood is unnecessary.

                if i == 0:
                    mean, variance = validate.mean, validate.variance
                    #lower, upper = validate.confidence_region()
                    self.train_test_cache = self.model.prediction_strategy._last_test_train_covar.T.to_dense()
                    #self.test_test_diag_cache = self.model.covar_module.forward(xi, xi, diag=True)
                else:
                    mean = torch.cat((mean, validate.mean))
                    variance = torch.cat((variance, validate.variance))
                    self.train_test_cache = torch.cat(
                        (self.train_test_cache, self.model.prediction_strategy._last_test_train_covar.T.to_dense()), 1
                    )
                    #self.test_test_diag_cache = torch.cat(
                    #    (self.test_test_diag_cache, self.model.covar_module.forward(xi, xi, diag=True))
                    #)
                    #l, u = validate.confidence_region()
                    #lower, upper = torch.cat((lower, l)), torch.cat((upper, u))

            R = self.model.prediction_strategy.covar_cache
            self.train_train_inv_cache = R @ R.T

        mean = mean.cpu().numpy()
        variance = variance.cpu().numpy()

        # return Prediction(mean, variance)
        return Prediction(self.domain.unravel(mean), self.domain.unravel(variance))

    def save(self, path: Path) -> None:
        if path.suffix != ".pth": path = path.with_name(path.name + ".pth")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path: Path) -> None:
        if path.suffix != ".pth": path = path.with_name(path.name + ".pth")
        self.model.load_state_dict(torch.load(path, weights_only=False))
