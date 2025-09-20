import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
from typing import Iterable
from itertools import product


class fileIOdatareader:

    def __init__(self, datapath: Path):
        self._read_data_kenny_format(datapath)
        print(f"Data loaded from {datapath}")
        print("x.shape:", self.x.shape)
        x_mins = np.min(self.x, axis=0)
        x_maxs = np.max(self.x, axis=0)
        y_mins = np.min(self.y, axis=0)
        y_maxs = np.max(self.y, axis=0)
        print("x.range:", f"[({x_mins}), ({x_maxs})]")
        print("y.shape:", self.y.shape)
        print("y.range:", f"[({y_mins}), ({y_maxs})]")
        self._tree = cKDTree(self.x)
        return None
    
    def __call__(self, freq, angle):
        _, idx = self._tree.query((freq, angle))
        return self.y[idx]
    
    @property
    def x(self):
        return self._x
    
    @property
    def y(self):
        return self._y

    def _read_data_kenny_format(self, file: str):
        from copy import deepcopy
        x_names, y_names, mhz, phi, data = np.load(file).values()
        ghz = mhz / 1000.0
        data_dict = {y:d for y,d in zip(y_names,data)}
        dreal, dimag = data_dict['Cpol(re)(90.0)'], data_dict['Cpol(im)(90.0)']
        complex_data = (dreal + 1j * dimag).T
        _M, _P = np.meshgrid(ghz, phi, indexing="xy")  # shapes: (181, 101)
        coords = np.stack([_M, _P], axis=-1)           # shape: (181, 101, 2)
        self._x = coords.reshape(-1, 2)
        self._y = complex_data.reshape(-1)
        return 0