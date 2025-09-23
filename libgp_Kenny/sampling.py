from __future__ import annotations
from collections import namedtuple
from abc import ABC, abstractmethod

import math, torch, numpy as np

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .gp import GaussianProcess as GP
from .tools import Array, Tensor, Domain, DataPair, vm_chunk_size

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class SampleStrategy(ABC):
    """An abstract class that represents a sample strategy, for adaptive sampling. All
    sample strategies should be a subclass of this. They need to implement metric() to
    return values for given candidates. This is known as the utility function U(x), in the paper.
    """
    name: str = "Unknown"

    def __init__(self) -> None:
        pass

    @abstractmethod
    def metric(self, gp: GP, candidates: Array) -> Array:
        raise NotImplementedError()

    @staticmethod
    def argsort(arr: Array) -> Array:
        return np.array(np.unravel_index(np.argsort(arr, axis=None), arr.shape)).T

    @staticmethod
    def arg_n_largest(arr: Array, n: int) -> Array:
        return np.array(np.unravel_index(np.argpartition(arr, -n, axis=None)[-n:], arr.shape)).T

    @staticmethod
    def spread_n_largest(arr: Array, n: int, spread: Array) -> Array:
        masked = arr[1:]
        found = [arr[0]] # First is always 0.
        # print("Reports from sampling.SampleStrategy.spread_n_largest:")
        # print("Shapes:", masked.shape, np.array(found).shape, spread.shape)
        for i in range(n-1):
            diff = (masked - found[-1]) / spread
            mask = np.where(np.linalg.norm(diff, axis=-1) >= 1)[0]
            found.append(masked[mask[0]]) # Append the first found one.
            masked = masked[mask[1:]] # New valid candidates

        return np.array(found)

    def new_sample_indices(
            self,
            add: int,
            gp: GP,
            old_indices: Array,
            spread: Union[int, Tuple[int]],
            marked_axes: Tuple[int],
    ) -> Array:
        if type(spread) is int:
            spread = (spread,)*old_indices.shape[-1]

        if len(spread) != old_indices.shape[-1]:
            raise ValueError("Given spread is invalid, must either be uniform, or match index dimensions.")

        spread = np.array(spread)
        diff = np.swapaxes(np.expand_dims(gp.domain.indices, -1) - old_indices.T, -2, -1) / spread
        mask = np.where(np.all(np.linalg.norm(diff, axis=-1) >= 1, axis=-1)) # Select all points out of the spread bound.
        candidates = gp.domain.indices[np.unique(mask)]

        # comb = np.vstack((gp.domain.indices, old_indices))
        # unique, counts = np.unique(comb, axis=0, return_counts=True)
        # candidates = unique[np.where(counts == 1)[0]]

        # select = np.all((candidates[..., -2:] % spread) == np.random.randint(spread), axis=-1)
        # self.__counter += 1
        # spread_mask = np.where(select)[0]
        # candidates = candidates[spread_mask]

        metric = self.metric(gp, candidates, marked_axes) # Compute metric at candidate points

        # print("Reports from sampling.SampleStrategy.new_sample_indices:")
        # print("candidates.shape:", candidates.shape)
        # print("metric.shape:", metric.shape)
        # print("SampleStrategy.argsort(metric).shape:", SampleStrategy.argsort(metric).shape)

        # Sort candidates by metric.
        sorted_idx = SampleStrategy.argsort(metric)[::-1]
        candidates = np.squeeze(candidates[sorted_idx], axis=-1)
        new_indices = SampleStrategy.spread_n_largest(candidates, add, spread)
        
        # print("candidates.shape:", candidates.shape)
        # print("new_indices.shape:", new_indices.shape)

        # Cancatenate old and new
        return np.concatenate((old_indices, new_indices))

class Random(SampleStrategy):
    name = "Rand"
    def metric(self, gp: GP, candidates: Array, marked_axes: Tuple[int]) -> Array:
        if self.seed: np.random.seed(self.seed)
        return np.random.random((candidates.shape[0],))

class Grid(SampleStrategy):
    name = "Grid"
    def __init__(self, gridsize: int):
        self.gridsize = gridsize

    def metric(self, gp: GP, candidates: Array, marked_axes: Tuple[int]) -> Array:
        gridsize = self.gridsize
        out = np.zeros((candidates.shape[0],))
        out[::gridsize] = np.ones((candidates.shape[0]/gridsize,))
        return out

class MaximumVariance(SampleStrategy):
    name = "MaxVar"
    def metric(self, gp: GP, candidates: Array, marked_axes: Tuple[int]) -> Array:
        flat_indices = np.ravel_multi_index(candidates.T, gp.domain.range_shape)
        return gp.prediction.variance.ravel()[flat_indices]

class MaximumGradient(SampleStrategy):
    name = "MaxGrad"
    def metric(self, gp: GP, candidates: Array, marked_axes: Tuple[int]) -> Array:
        deriv = np.zeros(gp.prediction.mean.shape) # Same as go.func.y.shape
        for i in range(gp.domain.n):
            slices = [slice(0,None)]*gp.domain.n
            slices[i] = slice(-1)
            deriv[tuple(slices)] += np.diff(gp.prediction.mean, axis=i)**2

        deriv = np.sqrt(deriv)
        flat_indices = np.ravel_multi_index(candidates.T, gp.domain.range_shape)
        return deriv.ravel()[flat_indices]

def chunked_K(gp, x1, x2):
    CHUNK = vm_chunk_size.value()
    k = gp.model.covar_module.forward
    with torch.no_grad():
        out = torch.zeros(x1.shape[0], x2.shape[0], device=x1.device, dtype=x1.dtype)
        for i in range(math.ceil(x1.shape[0]/CHUNK)):
            i1, i2 = i*CHUNK, (i+1)*CHUNK
            x1i = x1[i1:i2]
            for j in range(math.ceil(x2.shape[0]/CHUNK)):
                j1, j2 = j*CHUNK, (j+1)*CHUNK
                x2j = x2[j1:j2]
                out[i1:i2, j1:j2] = k(x1i, x2j).to_dense()
    return out

class VarianceMinimizer(SampleStrategy):
    name = "VM"
    def __init__(self, weight: float = 0.0, contrib_spacing: int = 2, schedule = False) -> None:
        self.weight = weight
        self.contrib_spacing = contrib_spacing
        self.schedule = schedule

    def metric(self, gp: GP, candidates: Array, marked_axes: Tuple[int]) -> Array:
        # Inputs are normalized into the GP.
        flat_indices = np.ravel_multi_index(candidates.T, gp.domain.range_shape)
        x_cand = gp.domain.normalize(gp.domain.raveled[flat_indices])
        x_cand = torch.from_numpy(x_cand).to(device)
        x_contrib = gp.domain.normalize(gp.domain.raveled[::self.contrib_spacing])
        x_contrib = torch.from_numpy(x_contrib).to(device)

        #contrib_ind = np.arange(0, gp.domain.raveled.shape[0], self.contrib_spacing)
        #contrib_ind = np.array(np.unravel_index(contrib_ind, gp.domain.range_shape)).T

        weights = self.weights(gp, candidates, marked_axes)

        # Contrib is the prediction set we minimize variance over.
        with torch.no_grad():
            Ainv = gp.train_train_inv_cache
            Kos = gp.train_test_cache
            Koca = gp.train_test_cache[:, flat_indices] # K train, candidates
            Koco = gp.train_test_cache[:, ::self.contrib_spacing] # K train, contrib
            #Koco *= weights
            W = Ainv @ Koca # N x K

            Kcaco = chunked_K(gp, x_cand, x_contrib) # K x M

            # We assume x \in X_*, and so
            # k(x,x) - b^T A^-1 b w is just the old variance.
            K = flat_indices.shape[0]

            c = 1/gp.prediction.variance.flatten()[flat_indices]
            CHUNK = vm_chunk_size.value()
            #bot = Kcaco.pow(2).sum(dim=1) # k(X_ca, X_co) dot k(X_ca, X_co) =
            bot = torch.empty(flat_indices.shape[0], device=x_cand.device, dtype=x_cand.dtype)
            for i in range(math.ceil(K/CHUNK)):
                i1, i2 = i*CHUNK, (i+1)*CHUNK
                bot[i1:i2] = Kcaco[i1:i2, :].pow(2).sum(dim=1)

            # Element wise multiply, still N x K then sum over rows
            # diag = -2*(W * (Koco @ Kcaco.T)).sum(dim=0)
            diag = torch.empty(flat_indices.shape[0], device=x_cand.device, dtype=x_cand.dtype)
            for i in range(math.ceil(K/CHUNK)):
                i1, i2 = i*CHUNK, (i+1)*CHUNK
                temp = W[:, i1:i2] * (Koco @ Kcaco.T[:, i1:i2])
                diag[i1:i2] = -2*( W[:, i1:i2] * (Koco @ Kcaco.T[:, i1:i2])).sum(dim=0)

            # Clear unneeded.
            del Kcaco
            torch.cuda.empty_cache()

            top_k = Koco @ Koco.T # N x N
            # top_v = torch.matmul(Wp.transpose(-1, -2), Wp) # K x N x N
            # top = top_v.flatten(start_dim=1) @ top_k

            top = torch.empty(flat_indices.shape[0], device=x_cand.device, dtype=x_cand.dtype)

            CHUNKR = int(math.sqrt(CHUNK))
            for i in range(math.ceil(K/CHUNK)):
                i1, i2 = i*CHUNK, (i+1)*CHUNK
                for j in range(math.ceil(W.shape[0]/CHUNKR)):
                    for k in range(math.ceil(W.shape[0]/CHUNKR)):
                        j1, j2 = j*CHUNKR, (j+1)*CHUNKR
                        k1, k2 = k*CHUNKR, (k+1)*CHUNKR
                        W_j, W_k = W[j1:j2, i1:i2], W[k1:k2, i1:i2]
                top[i1:i2] = torch.einsum("ik,jk,ij->k", W_j, W_k, top_k[j1:j2, k1:k2])

            bigV = (bot + diag + top).cpu().numpy()
            bigV *= c
            # a = v(gp, x_cand, x_cand, x_contrib, equals=True)
            # b = v(gp, x_cand, x_prev, x_contrib)

            # T = a + 2*b.sum(axis=-1)
            bigV /= np.linalg.norm(bigV)
        #return bigV
        if self.schedule:
            factor = np.exp(-2/self.weight)
            w = np.exp(-factor*gp.iter)
        else:
            w = self.weight
        return (1-w)*bigV + w*weights

    def weights(self, gp: GP, indices: Array, marked_axes: Tuple[int]) -> Tensor:
        return np.ones(indices.shape[0])

class VarianceMinimizer_1d(VarianceMinimizer):
    name = "GW_VM"

    def metric(self, gp: GP, candidates: Array, marked_axes: Tuple[int]) -> Array:
        return super().metric(gp, candidates, marked_axes)

class GradientWeightedVM(VarianceMinimizer):
    name = "GW_VM"

    def weights(self, gp: GP, indices: Array, marked_axes: Tuple[int]) -> Array:
        strat = MaximumGradient()
        grad = strat.metric(gp, indices, marked_axes) # Use gradient as weights.
        return grad / np.linalg.norm(grad)

class MaxVarWeightedVM(VarianceMinimizer):
    name = "MV_VM"

    def weights(self, gp: GP, indices: Array, marked_axes: Tuple[int]) -> Array:
        strat = MaximumVariance()
        vars = strat.metric(gp, indices, marked_axes) # Use gradient as weights.
        return vars / np.linalg.norm(vars)

class RandWeightedVM(VarianceMinimizer):
    name = "VMAS" # Name in paper.
    # name = "RW_VM"

    def weights(self, gp: GP, indices: Array, marked_axes: Tuple[int]) -> Array:
        strat = Random()
        strat.seed = self.seed
        rand = strat.metric(gp, indices, marked_axes) # Use gradient as weights.
        return rand

VMAS = RandWeightedVM # Name in paper.

SamplerSettings = namedtuple("SamplerSettings", "sampler start add end strategy spread marked_axes seed",
                             defaults=((None,)*8))

class Sampler(ABC):
    """A abstract class that represents a sampler. All samplers should be a subclass
    of this class, and implement truth() to obtain new samples.
    """
    domain: Domain
    start: int
    add: int
    end: int
    strategy: SampleStrategy
    spread: int
    axes: Tuple[int]

    def __init__(self, domain: Domain, start: int, add: int, end: int, strategy: SampleStrategy,
                 spread: int = 1, marked_axes: Tuple[int] = tuple(), seed = None) -> None:
        self.domain = domain
        self.start, self.add, self.end = start, add, end
        self.marked_axes = tuple() if type(marked_axes) is not tuple else marked_axes
        self.strategy = strategy
        self.spread = spread
        self.seed = seed
        self.strategy.seed = seed

    def initial(self) -> None:
        axes = tuple(sorted(set(range(self.domain.n)).difference(set(self.marked_axes))))
        self.indices = self.domain.indices
        # self.indices = self.domain.lhs_like(self.start, axes, self.seed)

    def next(self, gp: GP) -> None:
        self.indices = self.strategy.new_sample_indices(self.add, gp, self.indices, self.spread, self.marked_axes)

    def sample(self) -> DataPair:
        flat_indices = np.ravel_multi_index(self.indices.T, self.domain.range_shape)
        x = self.domain.raveled[flat_indices]
        # print("Report from sampling.Sampler.sample:")
        # print("self.indices.T.shape =", self.indices.T.shape)
        # print("self.domain.range_shape =", self.domain.range_shape)
        # print("flat_indices.shape =", flat_indices.shape)
        # print("x.shape =", x.shape)
        return DataPair(
            np.squeeze(x, axis=-1), np.squeeze(self.truth(x), axis=-1)
        )

    @abstractmethod
    def truth(self) -> Array:
        raise NotImplementedError()

    @staticmethod
    def from_settings(samp_sett: SampleSettings, domain: Domain, num_chunks: int = 1) -> Sampler:
        start = round(samp_sett.start / num_chunks)
        add = round(samp_sett.add / num_chunks)
        end = round(samp_sett.end / num_chunks)

        return samp_sett.sampler(domain, start, add, end, *samp_sett[4:])
