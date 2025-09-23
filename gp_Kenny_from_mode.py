from libgp_Kenny import *

import shutil, sys
from collections import namedtuple
from itertools import product
from abc import ABC, abstractmethod
from pathlib import Path

import torch, numpy as np

from libgp_Kenny.tools import Array, Tensor, Domain, DataPair, PlotData, color, pred_chunk_size, vm_chunk_size
from libgp_Kenny.gp import GaussianProcess as GP, KernelSettings, Prediction
from libgp_Kenny.sampling import SamplerSettings, Sampler, Random, Grid, VarianceMinimizer, GradientWeightedVM, RandWeightedVM
from gpytorch.settings import cholesky_jitter, max_cholesky_size

# from libgp_Kenny.figure import Figure
from scipy.spatial import cKDTree


from typing import List, Union
RunnerSettings = namedtuple("RunnerSettings", "iterations rate chunk")

import warnings
from gpytorch.utils.warnings import GPInputWarning


def make_sampler(train_x: Array, train_y: Array):
    # print("Reports from gp_Kenny_from_mode.make_sampler")
    # print("shape:", train_x.shape, train_y.shape)
    # print("dtype:", train_x.dtype, train_y.dtype)
    class FakeSampler(Sampler):
        _tree = cKDTree(train_x)
        y     = train_y

        def truth(self, x: Array) -> DataPair:
            # def _get(x: Array):
            #     _, idx = self._tree.query(x)
            #     return self.y[idx]
            # return torch.tensor([_get(xi) for xi in x])
            if x.ndim == 1: x = x[:, None]
            _, idx = self._tree.query(x[:, :1])
            return self.y[idx]

    return FakeSampler


class GaussianProcessMode(ABC):
    """An abstract class that represents a way of running the Gaussian process.
    Different modes should subclass this to reduce similar code for any Gaussian process based
    method.
    """
    name: str
    tasks: List[str]
    domain: Domain
    run_sett: RunnerSettings
    kern_sett: Union[KernelSettings, List[KernelSettings]]
    samp_sett: SamplerSettings
    overwrite: bool

    def __init__(self,
                 name: str,
                 tasks: List[str],
                 domain: Domain,
                 run_sett: RunnerSettings,
                 kern_sett: KernelSettings,
                 samp_sett: SamplerSettings,
                 overwrite: bool = False,
                 verbose: bool = False) -> None:
        self.name = name
        self.tasks = tasks
        self.domain = domain
        self.run_sett = run_sett
        self.kern_sett = kern_sett
        self.samp_sett = samp_sett
        self.overwrite = overwrite
        self.verbose = verbose

    def get_dir(self) -> None:
        self.path = Path(f"data/gp_kenny_data/{self.name}")
        if self.overwrite:
            shutil.rmtree(self.path, ignore_errors=True)
            return

        counter = 1
        name_i = f"{self.name}--{counter:02}"
        self.path = Path(f"data/gp_kenny_data/{name_i}")
        while self.path.is_dir():
            name_i = f"{self.name}--{counter:02}"
            self.path = Path(f"data/gp_kenny_data/{name_i}")
            counter += 1
        self.name = name_i

    def run(self, save: bool = True) -> None:
        if self.verbose: print("\n")
        self.get_dir()
        if self.verbose: print(f"[{color('START', '*G')}] {self.name}")
        self.run_(save)
        if self.verbose: print(f"[{color('END', '*G')}] {self.name}")
        if self.verbose: print("\n")
        return self.rmse

    @abstractmethod
    def run_(self) -> None:
        raise NotImplementedError()


class Adaptive(GaussianProcessMode):
    """This mode adaptively trains a Gaussian process, up to a given number of samples."""
    mode_name: str = "Adaptive"
    iter = 0
    def run_(self, save: bool = True) -> None:
        gp = GP(
            self.domain, self.kern_sett[0], self.samp_sett, 
            self.run_sett.chunk, self.verbose
            )
        while True:
            if self.verbose:
                print(f"[{color('INFO', '*Y')}] Start Sample Step: {gp.iter:02}, Samples: {gp.samples()[0].shape[0]}")
            gp.train(self.run_sett.iterations, self.run_sett.rate)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", GPInputWarning)
                pred = gp.prediction()
            pd = PlotData(
                DataPair(self.domain.x, self.samp_sett.sampler.y),
                gp.samples(), pred
            )
            if save:
                pd.save(self.path / f"{gp.iter:02}")
                gp.save(self.path / f"{gp.iter:02}")

            rmse = np.sqrt(np.mean(np.square(pd.ground.y - pd.pred.mean)))
            rel_l2 = np.linalg.norm(pd.ground.y - pd.pred.mean) / np.linalg.norm(pd.pred.mean)
            out_range = np.max(pd.ground.y) - np.min(pd.ground.y)
            if self.verbose:
                print(f"[{color('INFO', '*Y')}] Finish Sample Step: {gp.iter:02}, RMSE: {rmse/out_range:.2%} Rel. L2: {rel_l2: .2%}", flush=True)
            if not gp.next(): break
            self.iter = gp.iter
            # print(f"Loading GP model from", self.path / f"{self.iter-1:02}")
            gp.load(self.path / f"{self.iter+1:02}")
        self.rmse = np.sqrt(np.mean(np.square(pd.ground.y - pd.pred.mean))) / (np.max(pd.ground.y) - np.min(pd.ground.y))
    
    def predict(self, x, return_std=False) -> Array:
        sampler = make_sampler(x, x)    # dummy output just to expand sampler domain
        samp_sett = SamplerSettings(
            sampler, len(x), 1, len(x)+1, Random(), spread=1
            )
        gp = GP(Domain([x]), self.kern_sett[0], samp_sett, self.run_sett.chunk)
        gp.load(self.path / f"{self.iter+1:02}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", GPInputWarning)
            pred = gp.prediction()
        # pred = gp.prediction()
        prediction = [pred.mean]
        if return_std: 
            ystd = np.sqrt(pred.variance)
            prediction.append(ystd)
        return prediction


def train_model_gp_Kenny_from_mode(
        train_x, train_y, terms=4, training_iter: int = 100, 
        verbose = False, normalize_y=True
        ):
    # print("Reports from gp_Kenny_from_mode.train_model_gp_Kenny_from_mode:")
    # print("train_x.dtype:", train_x.dtype)
    # print("train_y.dtype:", train_y.dtype)
    overwrite = False
    task_batches = [
        ["Cpol(re)(0.0)"], ["Cpol(im)(0.0)"], 
        # ["Cpol(re)(90.0)"], ["Cpol(im)(90.0)"], 
    ]
    kern_setts = [
        KernelSettings("LF_NSM", nu=0.5, terms=terms, dims=1),
        KernelSettings("LF_NSM", nu=0.5, terms=terms, dims=1),
    ]
    strategies = [
        Random(),
        Random(),
    ]
    run_setts = [
        RunnerSettings(iterations=training_iter, rate=0.1, chunk=None),
        RunnerSettings(iterations=training_iter, rate=0.1, chunk=None),
    ]
    train_y_batch = [
        train_y.real, train_y.imag
    ]
    gp_list = []
    mhz = np.linspace(9500, 10500, 101)
    phi = np.linspace(0, 180, 181)
    n_data = len(train_x)
    for i, tup in enumerate(zip(task_batches, kern_setts, strategies, run_setts, train_y_batch)):
        # print(f"Training {i}....")
        tasks, kern_sett, strat, run_sett, train_y_batch = tup
        kerns = [kern_sett]
        sampler = make_sampler(train_x, train_y_batch)
        samp_sett = SamplerSettings(sampler, n_data, 1, n_data+2, strat, spread=1)
        name = [Adaptive.mode_name, ','.join(tasks), kern_sett.name]
        name = "-".join(name)
        domain = Domain([mhz])
        mode = Adaptive(
            name, tasks, domain, run_sett, kerns, samp_sett, 
            overwrite=overwrite, verbose=verbose
            )
        mode.run()
        gp_list.append(mode)
        # Use mode.name, since the name could be updated
        # make_figures(mode.name, tasks, False)
    gp_re, gp_im = gp_list
    return gp_re, gp_im