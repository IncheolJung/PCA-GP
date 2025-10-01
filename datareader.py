import os, sys
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
from typing import Iterable
from itertools import product
from joblib import Parallel, delayed
import time


class fileIOdatareader:

    def __init__(self, datapath: Path, print_info=True):
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
        if isinstance(freq, Iterable):
            return np.array([self.__call__(f, angle) for f in freq])
            # results = Parallel(n_jobs=4, prefer="threads")(
            #     delayed(self.__call__)(f, angle, delay=0.01*i) 
            #     for i, f in enumerate(freq)
            # )
            # return np.array(list(results))
        if isinstance(angle, Iterable):
            return np.array([self.__call__(freq, a) for a in angle])
            # return Parallel(n_jobs=4, prefer="threads")(
            #     delayed(self.__call__)(freq, a) for a in angle
            # )
            # return np.array(list(results))
        else:
            _, idx = self._tree.query((freq, angle))
        return self.y[idx]
    
    @property
    def x(self):
        return self._x
    
    @property
    def y(self):
        return self._y

    def _read_data_kenny_format(self, file: str):
        if isinstance(file, Path): file = file.name
        if   file == "data/data-for-kenny-paper-HH.npz": 
            re_key, im_key = 'Cpol(re)(90.0)', 'Cpol(im)(90.0)'
        elif file == "data/data-for-kenny-paper-VV.npz": 
            re_key, im_key = 'Cpol(re)(0.0)', 'Cpol(im)(0.0)'
        else:
            raise RuntimeError(f"{file} not suppported")
        x_names, y_names, mhz, phi, data = np.load(file).values()
        data_dict = {y:d for y,d in zip(y_names,data)}
        dreal, dimag = data_dict[re_key], data_dict[im_key]
        complex_data = (dreal + 1j * dimag).T
        _M, _P = np.meshgrid(mhz, phi, indexing="xy")  # shapes: (101, 181)
        coords = np.stack([_M, _P], axis=-1)           # shape: (101, 181, 2)
        self._x = coords.reshape(-1, 2)
        self._y = complex_data.reshape(-1)
        self.phi = phi
        self.theta = np.full_like(phi, fill_value=90)
        return 0


class OnFlySolver:

    def __init__(
            self, 
            workingpath: str, 
            model_name: str = None, 
            init_freqs: list = [], 
            angles = np.linspace(0, 180, 181),
            sweep_angle_type = 0,
            precision: int = 3
            ):
        "sweep_angle_type: [0, 1] = [phi, theta]"
        self.workingpath = Path(workingpath).absolute().__str__()
        if model_name is None:
            glob_domain_file = list(Path(workingpath).glob("*.domain"))
            if len(glob_domain_file) == 1:
                model_name = glob_domain_file[0].stem
            else:
                raise RuntimeError(
                    "One .domain file needs to exist "
                    f"under {self.workingpath}. "
                    f"Found {len(glob_domain_file)}."
                )
        self.model_name = model_name
        self.in_file = Path(f"{model_name}.in")
        self.solver = Path("./data/VWT-data/VWT-IE-Solver-Unified-DG.x").absolute().__str__()
        self.freqs = init_freqs
        self.sweep_angle_type = sweep_angle_type
        if sweep_angle_type==0:     # phi-sweep
            self.phi = np.array(angles)
            self.theta = 90*np.ones(len(angles))
        elif sweep_angle_type==1:   # theta-sweep
            self.phi = np.zeros(len(angles))
            self.theta = np.array(angles)
        else: raise RuntimeError(f"Invalid sweep_angle_type: {self.sweep_angle_type}")
        self.n_angles = len(angles)
        self.encoder = lambda x: round(float(x), precision)
        self.farfields = {}         # keys: (freq, angle), values: (cpol, xpol)
        self._init_in_file()        # init .in with dummy freq
        self.tmp = {}               # tmp directories for simulations

        # if len(self.freqs) > 0:
        #     if len(self.freqs) > 1: # parallel if multiple freq
        #         print("Now using Parallel solver")
        #         _run_one = lambda f: self.run(f)
        #         freq_to_run = []
        #         for freq in self.freqs:
        #             if (self._is_data_exist(freq)): pass
        #             else: freq_to_run.append(freq)
        #         results = Parallel(n_jobs=4)(  # use $(nproc) / 4 cores
        #             delayed(_run_one)(f) for f in range(freq_to_run)
        #         )
        #         broken_runs = {i: r for i,r in enumerate(results) if r != 0}
        #         if len(broken_runs): 
        #             raise RuntimeError(f"Simulations broken: {broken_runs}")
        #     else:       # sequential if one freq
        #         print("Now using sequential solver")
        #         for freq in self.freqs:
        #             self.__call__(freq, self.phi[0])

        if len(self.freqs) > 0:
            for freq in self.freqs:
                self.__call__(freq, self.phi[0])

        def get_freqs_from_dir(workingpath):
            def rm_r(path: Path):
                if path.is_dir():
                    for child in path.iterdir():
                        rm_r(child)  # recurse into children
                    path.rmdir()      # remove the now-empty directory
                else:
                    path.unlink()     # remove file or symlink
            freqs = []
            for d in Path(workingpath).glob("*/"):
                if d.is_dir():
                    try: freqs.append(float(d.name))
                    except: rm_r(d)
            return freqs
        freqs_computed_before = get_freqs_from_dir(Path(workingpath))
        if len(freqs_computed_before) > 0:
            for freq in freqs_computed_before:
                self._load_farfield_data(f"{self.workingpath}/{freq}/{self.model_name}.efar")
            self.freqs.extend(list(sorted(freqs_computed_before)))

        return None
    
    def __call__(self, freq, angle, delay=0):
        # Parallel(n_jobs=4)(  # use $(nproc) / 4 cores
        #             delayed(_run_one)(f) for f in range(freq_to_run)
        #         )
        time.sleep(delay)
        if isinstance(freq, Iterable):
            # return [self.__call__(f, angle) for f in freq]
            results = Parallel(n_jobs=4, prefer="threads")(
                delayed(self.__call__)(f, angle, delay=0.01*i) 
                for i, f in enumerate(freq)
            )
            return np.array(list(results))
        if isinstance(angle, Iterable):
            results = [self.__call__(freq, a) for a in angle]
            # return Parallel(n_jobs=4, prefer="threads")(
            #     delayed(self.__call__)(freq, a) for a in angle
            # )
            return np.array(results)
        else:
            freq, angle = self.encoder(freq), self.encoder(angle)
            if (self._is_data_exist(freq)): pass
            else: 
                exit = self.run(freq)
                if exit != 0:
                    raise RuntimeError(f"simulation error exit {exit}")
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
        Einc_mag, Einc_phase, Pol_angle, outCurJ = "1", "0", "0", "0"
        inc_E_data = [Einc_mag, Einc_phase, Pol_angle, outCurJ, "\n"]
        for i, (theta, phi) in enumerate(zip(self.theta, self.phi)):
            in_data.append(" ".join([str(i+1), str(theta), str(phi), *inc_E_data]))
        self.in_file.open('w').write("".join(in_data))
        os.chdir(org_path)
        return 0
    
    def _setup_in_file(self, freq_query):
        "assume in self.workingpath && setup .in"
        in_data = self.in_file.open('r').readlines()
        if len(in_data) <= 1:
            raise ValueError(
                f"_setup_in_file expected at least 2 lines, "
                f"but got {len(in_data)}. freq_query={freq_query}, in_data={in_data}"
            )
        n_angles, freq, loss_c, precond_mode, skeleton_c = in_data[1].split()
        in_data[1] = " ".join([
            str(self.n_angles), str(freq_query), loss_c, 
            precond_mode, skeleton_c, "\n"
            ])
        self.in_file.open('w').write("".join(in_data))
        return 0
    
    def _setup_simulation(self, freq_query):
        "mkdir ~tmp_$freq_query && cp $model_name.* $freq_query"
        self.tmp[freq_query] = f"~tmp{len(self.tmp)+1}_{freq_query}"
        working_path_now = self.tmp[freq_query]
        os.mkdir(working_path_now)
        self._cp_all_files(to_dir=working_path_now)
        os.chdir(working_path_now)
        return 0
    
    def _exit_simulation(self, freq_query, org_path):
        "cd .. && mv ~tmp_$freq_query"
        import re
        os.chdir("../")
        old_name = self.tmp[freq_query]
        new_name = re.sub(r"^~tmp\d+_", "", old_name)
        os.rename(old_name, new_name)
        del self.tmp[freq_query]
        os.chdir(org_path)
        return 0
    
    def _cp_all_files(self, from_dir=".", to_dir=".", new_name=False):
        import glob, shutil
        # Path(to_dir).mkdir(exist_ok=True)
        # files_to_cp = list(glob.glob(f"{from_dir}/{self.model_name}.*"))
        # for f in glob.glob(f"{from_dir}/{self.model_name}.*"):
        #     if new_name:
        #         new_name_ = f.replace(f"{self.model_name}.", f"{new_name}.")
        #     else:
        #         new_name_ = f
        #     shutil.copy2(f, f"{to_dir}/{new_name_}")
        from_dir = Path(from_dir)
        to_dir   = Path(to_dir)
        to_dir.mkdir(parents=True, exist_ok=True)
        files_to_move = list(from_dir.glob(f"{self.model_name}*.*"))
        domain_file = from_dir/f"{self.model_name}.domain"
        if domain_file in files_to_move:    # when DDsetup cp all subdomains
            domain_data = [line.strip() for line in domain_file.open('r').readlines()]
            n_subdomains = domain_data[1]
            for subd in domain_data[3:]:
                try: 
                    number, subd_name = subd.split("\t")
                except ValueError:
                    number, subd_name = subd.split()
                files_to_move.append((from_dir/subd_name).with_suffix(".tri"))
                files_to_move.append((from_dir/subd_name).with_suffix(".rgmatflg"))
        for f in files_to_move:
            if f.is_file():
                shutil.copy2(f, to_dir / f.name)
        return 0
    
    def _load_farfield_data(self, farfield_file: str, freq = None):
        if freq is None: freq = float(Path(farfield_file).parent.name)
        freq = self.encoder(freq)
        farfield_data = Path(farfield_file).open('r').readlines()[1:]
        if len(farfield_data) != len(self.phi): 
            raise RuntimeError("data unmatched with queried angle:\n"
                               f"datasize {len(farfield_data)} != request {len(self.phi)}\n"
                               f"at frequency {freq}")
        for d in farfield_data:
            theta, phi, cpol_re, cpol_im, xpol_re, xpol_im = map(float, d.strip("\n").split())
            if self.sweep_angle_type==0:    # phi-sweep
                angle = self.encoder(phi)
            elif self.sweep_angle_type==1:  # theta-sweep
                angle = self.encoder(theta)
            else: raise RuntimeError(f"Invalid sweep_angle_type: {self.sweep_angle_type}")
            self.farfields[(freq, angle)] = (cpol_re+1j*cpol_im, xpol_re+1j*xpol_im)
        self.freqs.append(freq)
        return 0
    
    def run(self, freq_query):
        from subprocess import Popen, STDOUT, PIPE
        print(f"\n ======  Adding Frequency Sample: {freq_query}  ====== \n")
        freq_query = self.encoder(freq_query)
        org_path = os.getcwd()
        os.chdir(self.workingpath)
        self._setup_simulation(freq_query)
        self._setup_in_file(freq_query)
        log_file_path = f"{self.model_name}.log"
        log_monitor_path = f"{self.workingpath}/compute.log"

        # Run simulation
        log_file = open(log_file_path, "w")
        log_monitor = open(log_monitor_path, "w")
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
            log_monitor.write(line)      # write to log file
            log_monitor.flush()          # flush after every line
            sys.stdout.flush()        # optional: ensure console shows immediately

        prog.wait()

        log_file.close()
        log_monitor.close()

        # self._cp_all_files(to_dir=f"./{freq_query}")
        # os.chdir(org_path)
        self._load_farfield_data(f"./{self.model_name}.efar", freq_query)
        self._exit_simulation(freq_query, org_path)
        return 0