from __future__ import annotations
from collections import namedtuple
from itertools import product
import math
from typing import List, Union, Literal

import torch, numpy as np
from scipy.stats import qmc

from pathlib import Path
from os import listdir

Array = np.ndarray
Tensor = torch.Tensor

# Colors:
# Bl, R, G, Y, B, M, C, lG, G, lR, lG, lY, lB, lM, lC, W
FG_COL = dict(zip(
    ["Bl", "R", "G", "Y", "B", "M", "C", "lGr", "Gr", "lR", "lG", "lY", "lB", "lM", "lC", "W"],
    list(range(30, 38)) + list(range(90, 98))
))

BG_COL = dict(zip(
    ["Bl", "R", "G", "Y", "B", "M", "C", "lGr", "Gr", "lR", "lG", "lY", "lB", "lM", "lC", "W"],
    list(range(40, 48)) + list(range(100, 108))
))
# Bold, Faint, Italics, Underscore
FORMAT = dict(zip(
    ["*", "#", "^", "_"], range(1, 5)
))

# BFG_COl.BG_COL
def color(s: str, format: str) -> str:
    beg = "\033["
    if format[0] in FORMAT:
        beg += f"{FORMAT[format[0]]};"
        format = format[1:]
    format = format.split(".")
    if format[0] in FG_COL: beg += f"{FG_COL[format[0]]};"
    if len(format) > 1 and format[1] in BG_COL: beg += f"{BG_COL[format[1]]};"
    return beg[:-1] + "m" + s + "\033[0m"

class DataPair:
    x: Union[Array, Tensor]
    y: Union[Array, Tensor]

    def __init__(self, x: Union[Array, Tensor], y: Union[Array, Tensor]) -> None:
        self.x, self.y = x, y

    def numpy(self) -> DataPair:
        if type(self.x) is Tensor: self.x = self.x.numpy()
        if type(self.y) is Tensor: self.y = self.y.numpy()
        return self

    def torch(self) -> DataPair:
        if type(self.x) is Array: self.x = torch.from_numpy(self.x)
        if type(self.y) is Array: self.y = torch.from_numpy(self.y)
        return self
    
    def to(self, device: Literal["cuda", "cpu"]):
        if device == 'cuda': return self.cuda()
        elif device == 'cpu': return self.cpu()
        else: raise ValueError(f"Invalid device name: {device}")

    def cuda(self) -> DataPair:
        self.torch()
        self.x = self.x.cuda()
        self.y = self.y.cuda()
        return self

    def cpu(self) -> DataPair:
        self.torch()
        self.x = self.x.cpu()
        self.y = self.y.cpu()
        return self


class Domain:
    """This class represents the input domain. It handles the wrapping and unwrapping
    details necessary for inputs in and out of Gaussian processes, as they need to be
    fully unraveled in a GP.
    """
    x: List[Array]
    #y: Array
    n: int
    raveled: DataPair

    def __init__(self, x: List[Array]) -> None:
        self.x = x
        self.n = len(self.x)

        self.range_shape = tuple(len(xi) for xi in self.x)
        self.indices = Domain.index_permutations([np.arange(len(xi)) for xi in self.x])
        self.raveled = Domain.index_permutations(self.x)

        # self.raveled = DataPair(
        #     Domain.index_permutations(self.x), # All index pairs.
        #     self.y.ravel()
        # )

    def unravel(self, new_y: Array) -> Array:
        return new_y.reshape(self.range_shape)

    @staticmethod
    def index_permutations(axes: List[Array], flatten=True) -> Array:
        # For axes with lengths x1, x2, ..., xn
        # Returns array with shape prod(x1 ... xn, n) if flatten, otherwise
        # returns array with shape (x1 ..., xn, n).
        m = len(axes)
        arr = np.array(np.meshgrid(*axes, indexing='ij')) # Shape(N, x1, x2, ... ,xn)
        arr = np.moveaxis(arr, np.arange(m+1), np.roll(np.arange(m+1), 1)) # Move first axis to back.
        return arr.reshape(-1, m) if flatten else arr

    def __check_axes(self, axes: Tuple[int]) -> None:
        for axis in axes:
            if axis >= self.n: raise ValueError("Invalid dimension.")

    def expand_indices(self, axes: Tuple[int], indices: Array) -> Array:
        axes = tuple(set(axes))
        self.__check_axes(axes)

        outer_shape, cols, unaxes = [], [], []
        for i in range(self.n):
            if i not in axes:
                l = len(self.x[i])
                outer_shape.append(l)
                cols.append(np.arange(l))
                unaxes.append(i)

        num_inds = indices.shape[0]
        cols.append(np.arange(num_inds))
        # Each outer_shape group needs to hold an "indices" copy, along with some fixed
        # permutation term of the other dimensions.
        arr = np.empty(tuple(outer_shape) + (num_inds, self.n), dtype=indices.dtype)
        arr[..., axes] = indices # Apply indices to every outer_shape group.

        # Permutation shape is arr.shape with last axis one more than unaxes:
        # (outer_shape, num_inds, len(unaxes) + 1)
        # For every group in outer shape, we have a num_inds x len(unaxes) + 1 block.
        # The first len(unaxes) dimensions are fixed for each group, and the last dim
        # goes through 0...num_inds-1.
        permutations = Domain.index_permutations(cols, False)
        arr[..., unaxes] = permutations[...,:-1]
        return arr.reshape(-1, self.n)

    def lhs_like(self, num_samples: int, axes: Tuple[int], seed: int = None) -> Array:
        # Axes specify the dimensions which will be used for LHS.
        # Unspecified dimensions are then completely sampled.
        axes = tuple(set(axes))
        self.__check_axes(axes)
        m = len(axes)
        sampler = qmc.LatinHypercube(d=m, seed=seed)
        indices = qmc.scale(sampler.random(n=num_samples),
                            [0]*m,
                            [len(self.x[i]) - 1 for i in axes]
                            )
        indices = np.round(indices).astype(int) # Quantize to selectable indices.
        return self.expand_indices(axes, indices)

    def normalize(self, x: Union[Array, Tensor]) -> Union[Array, Tensor]:
        start = self.raveled[0]
        end   = self.raveled[-1]
        denom = end - start

        if isinstance(x, np.ndarray):
            # Handle numpy
            numer = x - start
            with np.errstate(divide="ignore", invalid="ignore"):
                out = np.true_divide(numer, denom, where=denom != 0)
                out = np.where(denom == 0, numer, out)
            return out

        elif isinstance(x, torch.Tensor):
            # Handle torch
            denom = torch.from_numpy(denom)
            numer = x - torch.from_numpy(start)
            denom_iszero = denom == 0
            out = torch.empty_like(numer)
            out = torch.where(denom_iszero, numer, numer / denom)
            return out

        else:
            raise TypeError(f"Unsupported type {type(x)}")
        # return (x - self.raveled[0]) / (self.raveled[-1] - self.raveled[0])

    def unnormalize(self, x: Union[Array, Tensor]) -> Union[Array, Tensor]:
        denom = self.raveled[-1] - self.raveled[0]
        return x * denom + self.raveled[0]

    def chunk(self, shape: Tuple[int]) -> Array[Domain]:
        iterables = [range(x) for x in shape]
        divs = [math.ceil(len(self.x[i]) / shape[i]) for i in range(len(shape))]
        funcs = np.empty(shape, dtype=object)
        for chunk_index in product(*iterables):
            chunk_slices = []
            for i, xi in enumerate(self.x):
                sec_i = chunk_index[i]
                chunk_slices.append(slice(divs[i]*sec_i, divs[i]*(sec_i+1)))

            chunk_x = [self.x[i][s] for i, s in enumerate(chunk_slices)]
            funcs[chunk_index] = Domain(chunk_x)
        return funcs

    def unchunk(self, domains: Array[Domain]) -> Array:
        out = np.empty(self.range_shape)
        shape = domains.shape
        iterables = [range(x) for x in shape]
        divs = [math.ceil(len(self.x[i]) / shape[i]) for i in range(len(shape))]
        for chunk_index in product(*iterables):
            chunk_slices = []
            for i, xi in enumerate(self.x):
                sec_i = chunk_index[i]
                chunk_slices.append(slice(divs[i]*sec_i, divs[i]*(sec_i+1)))

            out[tuple(chunk_slices)] = domains[chunk_index]
        return out

Prediction = namedtuple("Prediction", "mean variance")

class PlotData:
    ground: DataPair
    samples: Tuple[Array, Array]
    pred: Prediction
    path: Path

    def __init__(self, ground, samples, pred) -> None:
        self.ground, self.samples, self.pred = ground, samples, pred

    def save(self, path: Path) -> None:
        if path.suffix != ".npz": path = path.with_name(path.name + ".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        np.savez_compressed(path, *self.ground.x, y=self.ground.y,
                 samp0=self.samples[0], samp1=self.samples[1],
                 mean=self.pred.mean, var=self.pred.variance)

    @staticmethod
    def load(path: Path) -> PlotData:
        if path.suffix != ".npz": path = path.with_name(path.name + ".npz")
        arr = np.load(path)
        xs = [x for x in arr.files if "_" in x]
        return PlotData(
            DataPair([arr[x] for x in xs], arr['y']),
            (arr['samp0'], arr['samp1']),
            Prediction(arr['mean'], arr['var'])
        )



class _value_context:
    _global_value = None

    @classmethod
    def value(cls):
        return cls._global_value

    @classmethod
    def _set_value(cls, value):
        cls._global_value = value

    def __init__(self, value):
        self._orig_value = self.__class__.value()
        self._instance_value = value

    def __enter__(self):
        self.__class__._set_value(self._instance_value)

    def __exit__(self, *args):
        self.__class__._set_value(self._orig_value)
        return False


class pred_chunk_size(_value_context):
    _global_value = 5000

class vm_chunk_size(_value_context):
    _global_value = 4000


def move_files(from_dir: Path, to_dir: Path,
                name_includes: str = None, suffixes: list = None):
    files_in_from_dir = listdir(from_dir.resolve())
    if name_includes is not None:
        files_in_from_dir = [from_dir / Path(f) for f in files_in_from_dir if name_includes in f]
    if suffixes is not None:
        files_in_from_dir = [f for f in files_in_from_dir if f.suffix in suffixes]
    for from_file in files_in_from_dir:
        try: from_file.rename(to_dir / from_file.name)
        except FileNotFoundError:
            print(f"File not found: {from_file.resolve().absolute()}")
    return files_in_from_dir