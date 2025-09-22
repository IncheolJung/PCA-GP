import glob, os, sys
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
from typing import Iterable
from itertools import product


class fileIOdatareader:

    def __init__(self, datapath: Path, print_info=False):
        self._read_data_kenny_format(datapath)
        if print_info:
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
        data_dict = {y:d for y,d in zip(y_names,data)}
        dreal, dimag = data_dict['Cpol(re)(90.0)'], data_dict['Cpol(im)(90.0)']
        complex_data = (dreal + 1j * dimag).T
        _M, _P = np.meshgrid(mhz, phi, indexing="xy")  # shapes: (101, 181)
        coords = np.stack([_M, _P], axis=-1)           # shape: (101, 181, 2)
        self._x = coords.reshape(-1, 2)
        self._y = complex_data.reshape(-1)
        return 0


class OnFlySolver:

    def __init__(
            self, 
            workingpath: str, 
            model_name: str, 
            init_freqs: list = [], 
            angles = np.linspace(0, 180, 181),
            precision: int = 3
            ):
        self.workingpath = Path(workingpath).absolute().__str__()
        self.model_name = model_name
        self.in_file = Path(f"{model_name}.in")
        self.solver = "../VWT-IE-Solver-Unified-DG.x"
        self.freqs = init_freqs
        self.phi = np.array(angles)
        self.n_angles = len(angles)
        self.encoder = lambda x: round(float(x), precision)
        self.farfields = {}         # keys: (freq, angle), values: (cpol, xpol)
        self._init_in_file()        # init .in with dummy freq
        if len(self.freqs) > 0:
            # self._tree = cKDTree(list(product(self.freqs, self.phi)))
            for freq in self.freqs:
                self.__call__(freq, self.phi[0])
        freqs_computed_before = [float(d.name) for d in Path(workingpath).glob("*/") if d.is_dir()]
        if len(freqs_computed_before) > 0:
            # self._tree = cKDTree(list(product(self.freqs, self.phi)))
            for freq in freqs_computed_before:
                # self.__call__(freq, self.phi[0])
                self._load_farfield_data(f"{self.workingpath}/{freq}/{self.model_name}.efar")
            self.freqs.extend(list(sorted(freqs_computed_before)))
        return None
    
    def __call__(self, freq, angle):
        freq, angle = self.encoder(freq), self.encoder(angle)
        if (self._is_data_exist(freq)): pass
        else: self.run(freq)
        # if ((freq, angle) not in self.farfields.keys()) :
        #     print(f"warning: (freq, angle) = ({freq}, {angle}) not found in self.farfields.keys()")
        #     print("type(freq):", type(freq))
        #     print("type(angle):", type(angle))
        #     freq_key, angle_key = list(self.farfields.keys())[0]
        #     print(f"(freq_key, angle_key) = ({freq_key}, {angle_key})")
        #     print("type(freq_key):", type(freq_key))
        #     print("type(angle_key):", type(angle_key))
        #     self.run(freq)
        return self.farfields[(freq, angle)][0]     # self.farfields[(freq, angle)] = (cpol, xpol)
    
    def _is_data_exist(self, freq_query):
        return freq_query in self.freqs
    
    def _add_freq(self, new_freq):
        return self.freqs.append(new_freq)
    
    def _init_in_file(self, freq_query=2580):
        org_path = os.getcwd()
        os.chdir(self.workingpath)
        in_data = self.in_file.open('r').readlines()[:2]
        n_angles, freq, loss_c, precond_mode, skeleton_c = in_data[1].split()
        in_data[1] = " ".join([str(self.n_angles), str(freq_query), loss_c, precond_mode, skeleton_c, "\n"])
        theta, Einc_mag, Einc_phase, Pol_angle, outCurJ = "90", "1", "0", "0", "0"
        inc_E_data = [Einc_mag, Einc_phase, Pol_angle, outCurJ, "\n"]
        for i, phi in enumerate(self.phi):
            in_data.append(" ".join([str(i+1), theta, str(phi), *inc_E_data]))
        self.in_file.open('w').write("".join(in_data))
        os.chdir(org_path)
        return 0
    
    def _set_in_file(self, freq_query):
        org_path = os.getcwd()
        os.chdir(self.workingpath)
        in_data = self.in_file.open('r').readlines()
        n_angles, freq, loss_c, precond_mode, skeleton_c = in_data[1].split()
        in_data[1] = " ".join([str(self.n_angles), str(freq_query), loss_c, precond_mode, skeleton_c, "\n"])
        self.in_file.open('w').write("".join(in_data))
        os.chdir(org_path)
        return 0
    
    def _cp_all_files(self, from_dir=".", to_dir=".", new_name=False):
        import shutil
        Path(to_dir).mkdir(exist_ok=True)
        for f in glob.glob(f"{from_dir}/{self.model_name}.*"):
            if new_name:
                new_name_ = f.replace(f"{self.model_name}.", f"{new_name}.")
            else:
                new_name_ = f
            shutil.copy2(f, f"{to_dir}/{new_name_}")
        return 0
    
    def _load_farfield_data(self, farfield_file: str, freq = None):
        if freq is None: freq = float(Path(farfield_file).parent.name)
        freq = self.encoder(freq)
        farfield_data = Path(farfield_file).open('r').readlines()[1:]
        if len(farfield_data) != len(self.phi): 
            raise RuntimeError("data unmatched with queried angle: \n\t"
                               f"datasize {len(farfield_data)} != request {len(self.phi)}")
        for d in farfield_data:
            theta, phi, cpol_re, cpol_im, xpol_re, xpol_im = map(float, d.strip("\n").split())
            self.farfields[(freq, self.encoder(phi))] = (cpol_re+1j*cpol_im, xpol_re+1j*xpol_im)
        self.freqs.append(freq)
        return 0
    
    def run(self, freq_query):
        from subprocess import Popen, STDOUT, PIPE
        print(f"\n ======  Adding Frequency Sample: {freq_query}  ====== \n")
        freq_query = self.encoder(freq_query)
        org_path = os.getcwd()
        os.chdir(self.workingpath)
        self._set_in_file(freq_query)
        cmd = [self.solver, self.model_name, "|", "tee", f"{self.model_name}.log"]
        log_file_path = f"{self.model_name}.log"
        with open(log_file_path, "w") as log_file:
            # Start the process
            prog = Popen(
                [self.solver, self.model_name],
                stdout=PIPE,
                stderr=STDOUT,
                text=True,   # makes stdout/stderr strings instead of bytes
            )

            # Read stdout line by line, write to file and print to console
            for line in prog.stdout:
                # print(line, end='')      # print to console
                log_file.write(line)      # write to log file
                log_file.flush()          # flush after every line
                sys.stdout.flush()        # optional: ensure console shows immediately

            prog.wait()
        self._cp_all_files(to_dir=f"./{freq_query}")
        self._load_farfield_data(f"./{freq_query}/{self.model_name}.efar")
        os.chdir(org_path)
        return 0