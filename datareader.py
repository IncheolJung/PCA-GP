import os, sys
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
from typing import Iterable
from itertools import product
from joblib import Parallel, delayed
import time
from copy import deepcopy


        
def rm_r(path: Path):
    if path.is_dir():
        for child in path.iterdir():
            rm_r(child)  # recurse into children
        path.rmdir()      # remove the now-empty directory
    else:
        path.unlink()     # remove file or symlink



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
            # results = Parallel(n_jobs=4, backend="loky")(
            #     delayed(self.__call__)(f, angle, delay=0.01*i) 
            #     for i, f in enumerate(freq)
            # )
            # return np.array(list(results))
        if isinstance(angle, Iterable):
            return np.array([self.__call__(freq, a) for a in angle])
            # return Parallel(n_jobs=4, backend="loky")(
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
            precision: int = 3,
            usempi=False, mpicomm=None,
            ):

        self.usempi = usempi
        if usempi:
            self.comm = mpicomm
            self.rank = mpicomm.Get_rank()
            self.size = mpicomm.Get_size()
            print(f"[Rank {self.rank}] starting work...")
            self.root_ip = self._get_hostname(rank=0)

        self.init_args = [
            workingpath, model_name, init_freqs, 
            angles, sweep_angle_type, precision, 
            usempi, mpicomm
        ]

        "sweep_angle_type: [0, 1] = [phi, theta]"
        self.workingpath = Path(workingpath).absolute().__str__()
        if model_name is None:
            glob_domain_file = list(Path(workingpath).glob("*.domain"))
            if len(glob_domain_file) == 1:
                model_name = glob_domain_file[0].stem
            else:
                raise RuntimeError(
                    "One .domain file needs to exist",
                    f"under {self.workingpath}.",
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
            self.angles = self.phi
        elif sweep_angle_type==1:   # theta-sweep
            self.phi = np.zeros(len(angles))
            self.theta = np.array(angles)
            self.angles = self.theta
        else: raise RuntimeError(f"Invalid sweep_angle_type: {self.sweep_angle_type}")
        self.n_angles = len(angles)
        self.encoder = lambda x: round(float(x), precision)
        self.freqs = [self.encoder(f) for f in self.freqs]
        self.angles = [self.encoder(a) for a in self.angles]
        self.farfields = {}         # keys: (freq, angle), values: (cpol, xpol)
        if (not usempi) or (usempi and self.rank==0):
            self._init_in_file()        # init .in with dummy freq
        self.tmp = {}               # tmp directories for simulations

        if self.usempi: 
            delay = 0.5*self.rank
        else:
            delay = 0.0

        self._clean_simulations(delay=delay)

        def get_freqs_from_dir(workingpath):
            freqs = []
            for d in Path(workingpath).glob("*/"):
                if d.is_dir():
                    try: freqs.append(float(d.name))
                    except ValueError: rm_r(d)
            return freqs
        
        if usempi:
            if self.rank == 0:
                freqs_computed_before = get_freqs_from_dir(workingpath)
            else:
                freqs_computed_before = None
            freqs_computed_before = self.comm.bcast(freqs_computed_before, root=0)
        else:
            freqs_computed_before = get_freqs_from_dir(workingpath)
        
        if len(freqs_computed_before) > 0:
            for freq in freqs_computed_before:
                self._load_farfield_data(f"{self.workingpath}/{freq}/{self.model_name}.efar")
        
        # if usempi:
        #     self.comm.Barrier()
        #     print(f"[Rank {self.rank}] waiting other ranks...")

        # self.__call__(self.freqs, self.angles)

        if usempi:
            self.comm.Barrier()
            print(f"[Rank {self.rank}] finished data loading")
            results = self.__call__mpi(self.freqs, self.angles)
        else:
            print(f"[OnFlySolver] finished data loading")
            results = self.__call__openmp(self.freqs, self.angles)

        if usempi:
            self.comm.Barrier()
            print(f"[Rank {self.rank}] all simulations done")

        return None
    
    def __call__(self, *args, **kwargs):
        if self.usempi:
            return self.__call__mpi(*args, **kwargs)
        else:
            return self.__call__openmp(*args, **kwargs)
    
    def __call__openmp(self, freq, angle, delay=0):
        # Parallel(n_jobs=4)(  # use $(nproc) / 4 cores
        #             delayed(_run_one)(f) for f in range(freq_to_run)
        #         )
        if isinstance(freq, Iterable):
            return np.array([self.__call__openmp(f, angle) for f in freq])
            # results = Parallel(n_jobs=4, backend="loky")(
            #     delayed(self.__call__openmp)(f, angle, delay=0.01*i) 
            #     for i, f in enumerate(freq)
            # )
            # return np.array(list(results))
        else:
            freq = self.encoder(freq)
            if isinstance(angle, Iterable):
                angle = [self.encoder(a) for a in angle]
            else:
                angle = self.encoder(angle)
            if (self._is_data_exist(freq)): pass
            else: 
                exit_code = self.run(freq, angle)
                if exit_code != 0:
                    raise RuntimeError(f"simulation error exit {exit_code}")
        if isinstance(angle, Iterable):
            return np.array([self.__call__openmp(freq, a) for a in angle])
            # return Parallel(n_jobs=4, backend="loky")(
            #     delayed(self.__call__)(freq, a) for a in angle
            # )
            # return np.array(list(results))
        else:
            freq, angle = self.encoder(freq), self.encoder(angle)
            # print("__call__openmp:", freq, angle)
        return self.farfields[(freq, angle)][0]     # self.farfields[(freq, angle)] = (cpol, xpol)
    
    def _chunk_data(self, freq: Iterable, angle: Iterable) -> list:
        """Split frequency list into roughly equal chunks for each process."""
        n_freq = len(freq)
        if n_freq == 0:   # nothing happens
            chunks = [(None, None) for _ in range(self.size)]
        elif n_freq < self.size:   # chunk both freq and angle
            # find number of sub-chunk (angles)
            n_chunks_per_freq = [0 for _ in range(n_freq)]
            for i in range(self.size): 
                n_chunks_per_freq[i%n_freq] += 1
            # chunk angles first
            def split_array(arr, k):
                n = len(arr)
                if k <= 0:
                    return []
                base = n // k
                remainder = n % k

                chunks = []
                start = 0
                for i in range(k):
                    # First 'remainder' chunks get an extra element
                    end = start + base + (1 if i < remainder else 0)
                    chunks.append(arr[start:end])
                    start = end
                return chunks
            angle_chunks_per_freq = [split_array(angle, k) for k in n_chunks_per_freq]
            # chunk freq-angle
            chunks = [([], []) for _ in range(self.size)]
            mod = 0   # index modifier
            for i, f in enumerate(freq):
                for j, angle_chunk in enumerate(angle_chunks_per_freq[i]):
                    mod += j
                    chunks[i+mod][0].append(f)
                    chunks[i+mod][1].append(angle_chunk)
            # chunks = [([f], angle) for f in freq]
            # chunks.extend([(None, None) for _ in range(self.size - n_freq)])
        else:   # n_freq >= self.size   # Only chunk based on freq
            chunks = [([], []) for _ in range(self.size)]
            for i, f in enumerate(freq):
                chunks[i%self.size][0].append(f)
                chunks[i%self.size][1].append(angle)
        return chunks
    
    def _merge_efar_rcs(self, freq, partial_dirs):
        merged_dir: Path = Path(self.workingpath)/str(freq)
        merged_dir.mkdir(exist_ok=True)
        partial_dirs = [(Path(self.workingpath)/d).absolute() for d in partial_dirs]
        self._cp_all_files(from_dir=partial_dirs[0], to_dir=merged_dir)
        efar_file = merged_dir/f"{self.model_name}.efar"
        rcs_file = merged_dir/f"{self.model_name}.rcs"
        efar_header = efar_file.open('r').readlines()[:1]
        rcs_header = rcs_file.open('r').readlines()[:1]
        efar_data, rcs_data = [], []
        for d in partial_dirs:
            efar_data.extend(
                (d/f"{self.model_name}.efar").open('r').readlines()[1:]
            )
            rcs_data.extend(
                (d/f"{self.model_name}.rcs").open('r').readlines()[1:]
            )
        exit_code_efar = efar_file.open('w').write(''.join([*efar_header, *efar_data]))
        exit_code_rcs = rcs_file.open('w').write(''.join([*rcs_header, *rcs_data]))
        return exit_code_efar + exit_code_rcs
    
    def _clean_simulations(self, delay=0.0):
        time.sleep(delay)
        # complete_freqs = Path(self.workingpath).glob("*")
        partial_dirs = list(Path(self.workingpath).glob("*[[]*[]]"))
        partial_dirs = [d.name for d in sorted(partial_dirs) 
                            if d.is_dir() and not '~tmp' in d.name]
        partial_freqs = list(set([d[:d.index("_[")] for d in partial_dirs]))
        for freq in partial_freqs:
            partials_for_this_freq = [d for d in partial_dirs if freq in d]
            start_list, end_list = [], []
            for p in partials_for_this_freq:
                # suppose dirname is "frequency_[i1_i2]"
                start, end = map(int, p[len(freq)+2:-1].split('_'))
                start_list.append(start)
                end_list.append(end)
            start_list, end_list = sorted(start_list), sorted(end_list)
            start_glob, end_glob = start_list.pop(0), end_list.pop(0)
            index_list = []
            while start_list and end_list:
                start_i, end_i = start_list.pop(0), end_list.pop(0)
                if start_i == end_glob+1:   # sequential: e.g., [[..., 5], [6, ...]]
                    end_glob = end_i
                else:
                    index_list.append((start_glob, end_glob))
                    start_glob, end_glob = start_i, end_i
            index_list.append((start_glob, end_glob))
            if len(index_list) == 1:
                head, tail = index_list[0]
                if (head==0) and (tail==self.n_angles-1):
                    if all([(Path(self.workingpath)/d).exists() 
                            for d in partials_for_this_freq]):
                        self._merge_efar_rcs(float(freq), partials_for_this_freq)
                        for d in partials_for_this_freq:
                            rm_r(Path(self.workingpath)/d)
                    else:
                        print(
                            f"partials_for_this_freq {freq} no longer exists: "
                            f"{partials_for_this_freq}"
                        )
                else:
                    print(
                        f"Broken simulation at {freq}. "
                        f"Stored index are {index_list} "
                        f"while [0, {self.n_angles-1}] is required"
                    )
                    print("Attempting to gather simulations to root...")
                    partials_for_this_freq = [Path(self.workingpath)/d 
                                              for d in partials_for_this_freq]
                    self._transfer_dir_to_root(partials_for_this_freq)
            else:
                print(
                    f"Broken simulation at {freq}. "
                    f"Stored index are {index_list} "
                    f"while [0, {self.n_angles-1}] is required"
                )

        return 0
    
    def _transfer_dir_to_root(self, dirname: Path):
        if isinstance(dirname, Iterable):
            dirname = ' '.join(dirname)
        from subprocess import Popen, STDOUT, PIPE
        dest = f"{self.root_ip}:{self.workingpath}"
        prog = Popen(['scp', '-r', dirname, dest])
        prog.wait()
        return 0
    
    def _get_hostname(self, rank: int = 0):
        with open('hosts.txt', 'r') as f:
            ip_addr = f.readlines()[rank].split()
            if len(ip_addr)==2: ip_addr = ip_addr[0]
        return ip_addr
    
    # def _transfer_dir_to_root(self, dirname: Path):
    #     if isinstance(dirname, Iterable):
    #         return sum(self._transfer_dir_to_root(d) for d in dirname)
    #     return sum(self._transfer_file_to_root(str(f.absolute())) 
    #             for f in dirname.absolute().glob('*')
    #             if f.is_file())
    
    # def _transfer_file_to_root(self, filename: str):
    #     if self.rank != 0:
    #         # Worker reads its local result
    #         with open(filename, "rb") as f:
    #             data = f.read()
    #         # Send file to master
    #         self.comm.send((filename, data), dest=0)
    #     else:
    #         # Master node receives files from all workers
    #         for worker in range(1, self.size):
    #             fname, data = self.comm.recv(source=worker)
    #             # Save to master's directory
    #             if not Path(fname).parent.exists():
    #                 Path(fname).parent.mkdir()
    #             with open(fname, "wb") as f:
    #                 f.write(data)
    #     return 0
    
    def __call__mpi(self, freq, angle, delay=0):
        # Parallel(n_jobs=4)(  # use $(nproc) / 4 cores
        #             delayed(_run_one)(f) for f in range(freq_to_run)
        #         )

        if isinstance(freq, Iterable):
            # return np.array([self.__call__(f, angle) for f in freq])
            # Step 1: Only rank 0 checks which frequencies exist
            if self.rank == 0:
                freqs_encoded = [self.encoder(f) for f in freq]
                freqs_to_run = [f for f in freqs_encoded if not self._is_data_exist(f)]
                chunks = self._chunk_data(freqs_to_run, angle)
                # print("[Rank 0] freqs_to_run:", freqs_to_run)
                # print("[Rank 0] chunks:", chunks)
            else:
                chunks = None
            local_chunks = self.comm.scatter(chunks, root=0)
            # print(f"[Rank {self.rank}] local_chunks", local_chunks)
            local_freq, local_angle = local_chunks
            # print(f"[Rank {self.rank}] local_freq", local_freq)
            # print(f"[Rank {self.rank}] local_angle", local_angle)
            if local_freq is not None:
                min_angles = np.array(local_angle).min(axis=-1)
                max_angles = np.array(local_angle).max(axis=-1)
                print(
                    f"[Rank {self.rank}] attempting to simulate on...",
                    *[f"    {f} MHz, [{min_a}, {max_a}] DEG" 
                        for f, min_a, max_a in zip(local_freq, min_angles, max_angles)],
                    sep='\n',
                    )
                local_exit_code = [self.run(*work) for work in zip(local_freq, local_angle)]
                if any(code != 0 for code in local_exit_code):
                    raise RuntimeError(
                        f"simulation error exit {exit_code} at rank {self.rank}"
                    )
            self._clean_simulations(delay=0.5*self.rank)
            self.comm.Barrier()
            if self.rank == 0:
                return np.array([self.__call__openmp(f, angle) for f in freq])
            else:
                return None
        elif isinstance(angle, Iterable):
            return np.array([self.__call__openmp(freq, a) for a in angle])
            # return Parallel(n_jobs=4, backend="loky")(
            #     delayed(self.__call__)(freq, a) for a in angle
            # )
            # return np.array(list(results))
        else:
            freq, angle = self.encoder(freq), self.encoder(angle)
            if (self._is_data_exist(freq)): pass
            else: 
                exit_code = self.run(freq)
                if exit_code != 0:
                    raise RuntimeError(f"simulation error exit {exit_code}")
            
            if self.rank==0:
                return self.farfields[(freq, angle)][0]     # self.farfields[(freq, angle)] = (cpol, xpol)
            else:
                return None

    # def __call__mpi(self, freq, angle, delay=0):
    #     # Step 1: Encode frequencies and compute which ones to run (rank 0 only)
    #     if self.rank == 0:
    #         freqs_encoded = [self.encoder(f) for f in freq]
    #         freqs_to_run = [f for f in freqs_encoded if not self._is_data_exist(f)]
    #         chunks = self._chunk_data(freqs_to_run, angle)
    #         print(f"[Rank 0] freqs_to_run: {freqs_to_run}")
    #         print(f"[Rank 0] chunks: {chunks}")
    #     else:
    #         chunks = None

    #     # Step 2: Scatter chunks to all ranks
    #     local_freq = self.comm.scatter(chunks, root=0)
    #     print(f"[Rank {self.rank}] attempting to simulate on {local_freq}...")

    #     # Step 3: Each rank runs its own local simulations
    #     local_results = []
    #     for f in local_freq:
    #         exit_code = self.run(f)
    #         if exit_code != 0:
    #             raise RuntimeError(f"simulation error exit {exit_code} at rank {self.rank}")
    #         local_results.append(self._collect_result(f, angle))  # collect whatever result you need

    #     # Step 4: Gather results back to rank 0
    #     all_results = self.comm.gather(local_results, root=0)

    #     if self.rank == 0:
    #         # flatten results
    #         final_results = [item for sublist in all_results for item in sublist]
    #         print(f"[Rank 0] completed frequencies from all ranks: {final_results}")
    #         return np.array(final_results)
    #     else:
    #         return None
    
    def _is_data_exist(self, freq_query):
        return freq_query in self.freqs
    
    def _add_freq(self, new_freq):
        if not isinstance(new_freq, Iterable): new_freq = [new_freq]
        if self.usempi:
            freqs = self.comm.gather(new_freq, root=0)
            if self.rank == 0:
                new_freq = [item for sublist in freqs for item in sublist]
                self.freqs.extend(new_freq)
        else:
            self.freqs.extend(new_freq)
        return 0
    
    def _add_farfield(self, new_farfield):
        if self.usempi:
            farfields_local_keys, farfields_local_vals = zip(*new_farfield.items())
            farfields_keys = self.comm.gather(farfields_local_keys, root=0)
            farfields_vals = self.comm.gather(farfields_local_vals, root=0)
            if self.rank == 0:
                farfields_keys_ravel = [item for sublist in farfields_keys for item in sublist]
                farfields_vals_ravel = [item for sublist in farfields_vals for item in sublist]
                new_farfield = {k: v for k, v in zip(farfields_keys_ravel, farfields_vals_ravel)}
                self.farfields.update(new_farfield)
        else:
            self.farfields.update(new_farfield)
        return 0

    def _find_index_from_angles(self, angle_query):
        return [self.angles.index(a) for a in angle_query]
    
    def _init_in_file(self, freq_query=1280, delay=0):
        time.sleep(delay)
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
    
    def _setup_in_file(self, freq_query, angle_query_idx = None):
        "assume in self.workingpath && setup .in"
        in_data = self.in_file.open('r').readlines()
        if len(in_data) <= 1:
            raise ValueError(
                f"_setup_in_file expected at least 2 lines, "
                f"but got {len(in_data)}. freq_query={freq_query}, in_data={in_data}"
            )
        n_angles, freq, loss_c, precond_mode, skeleton_c = in_data[1].split()
        in_data[1] = " ".join([
            str(len(angle_query_idx)), str(freq_query), loss_c, 
            precond_mode, skeleton_c, "\n"
            ])
        if len(in_data) - 2 >= len(angle_query_idx):
            in_data_angles = [in_data[2:][i] for i in angle_query_idx]
            in_data = [*in_data[:2], *in_data_angles]
        else:
            raise RuntimeError(
                ".in file contains smaller inputs than querried "
                f"{len(in_data) - 2} < {len(angle_query_idx)}"
            )
        self.in_file.open('w').write("".join(in_data))
        return 0
    
    def _setup_simulation(self, freq_query, angle_query_idx = None):
        """
        mkdir ~tmp_$freq_query_$angle_idx_rng
        cp $model_name.* ~tmp_$freq_query_$angle_idx_rng
        - assumes angle_query_idx is in ascending order
        """
        istart, iend = angle_query_idx[0], angle_query_idx[-1]
        dirname = f"~tmp{len(self.tmp)+1}_{freq_query}_[{istart}_{iend}]"
        self.tmp[freq_query] = dirname
        working_path_now = dirname
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
    
    def _load_farfield_data(self, farfield_file: str, freq = None, angle=None):
        if freq is None: 
            try: freq = float(Path(farfield_file).absolute().parent.name)
            except ValueError: 
                for key, value in self.tmp.items():
                    if value == Path(farfield_file).parent.name:
                        freq = key
                        break
                if freq is None:
                    raise FileNotFoundError(
                        "Cannot find farfield data:", 
                        Path(farfield_file).absolute().__str__()
                    )
        if angle is None:
            angle = self.angles
        freq = self.encoder(freq)
        farfield_data = Path(farfield_file).open('r').readlines()[1:]
        if len(farfield_data) != len(angle): 
            farfield_data_from_file = ''.join(Path(farfield_file).open('r').readlines())
            raise RuntimeError(
                "data unmatched with queried angle:", 
                f"datasize {len(farfield_data)} != request {len(self.phi)}", 
                f"at frequency {freq}", 
                f"farfield_file: {Path(farfield_file).absolute().__str__()}", 
                f"{farfield_data_from_file}", 
                sep='\n'
            )
        farfields_new = {}
        freqs_new = []
        for d in farfield_data:
            theta, phi, cpol_re, cpol_im, xpol_re, xpol_im = map(float, d.strip("\n").split())
            if self.sweep_angle_type==0:    # phi-sweep
                angle = self.encoder(phi)
            elif self.sweep_angle_type==1:  # theta-sweep
                angle = self.encoder(theta)
            else: raise RuntimeError(f"Invalid sweep_angle_type: {self.sweep_angle_type}")
            farfields_new[(freq, angle)] = (cpol_re+1j*cpol_im, xpol_re+1j*xpol_im)
        freqs_new.append(freq)
        self._add_freq(freqs_new)
        self._add_farfield(farfields_new)
        return 0
    
    def run(self, freq_query, angle_query = None):
        if self.usempi:
            print(f"[Rank {self.rank}] running simulation at {freq_query}...")
        from subprocess import Popen, STDOUT, PIPE
        if freq_query is None: return 0     # pass
        if angle_query is None:
            angle_query = self.angles
        elif not isinstance(angle_query, Iterable):
            angle_query = [angle_query]
        print(f"\n ======  Adding Frequency Sample: {freq_query}  ====== \n")
        freq_query = self.encoder(freq_query)
        org_path = os.getcwd()
        os.chdir(self.workingpath)
        angle_query_idx = self._find_index_from_angles(angle_query)
        self._setup_simulation(freq_query, angle_query_idx)
        self._setup_in_file(freq_query, angle_query_idx)
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
        self._load_farfield_data(
            f"./{self.model_name}.efar", 
            freq_query, angle_query
        )
        self._exit_simulation(freq_query, org_path)
        return 0
    

class OnFlySolverMyMoM:

    def __init__(
            self, 
            workingpath: str, 
            model_name: str = None, 
            init_freqs: list = [], 
            angles = None,              # dummy input
            sweep_angle_type = None,    # dummy input
            other_args = [90, 0, 0, 1e-5], 
            precision: int = 3,
            usempi=False
            ):
        self.workingpath = Path(workingpath).absolute().__str__()
        self.model_name = model_name
        base_path = Path(workingpath).absolute().parent
        self.solver = base_path/"bin"/"run.sh"  # list of exe
        self.freqs = init_freqs
        self.nodes = []     # will check nodes every time run()
        self.encoder = lambda x: f"{float(x):.3e}"
        self.args = " ".join(map(str, other_args))
        self.currents = {}   # keys: (freq, node), values: current
        self.tmp = {}        # reserved for parallelized solve

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
                    except ValueError: rm_r(d)
            return freqs
        freqs_computed_before = get_freqs_from_dir(workingpath)
        if len(freqs_computed_before) > 0:
            for freq in freqs_computed_before:
                freq = self.encoder(freq)
                self._load_current_data(f"{self.workingpath}/{freq}/I.mat", freq)
            self.freqs.extend(list(sorted(freqs_computed_before)))

        if len(self.freqs) > 0:
            for freq in self.freqs:
                self.__call__(freq, 1)

        return None
    
    def __call__(self, freq, node, delay=0):
        # Handle iterable frequency
        if isinstance(freq, Iterable) and not isinstance(freq, (str, bytes)):
            return np.array([self.__call__(f, node, delay=delay) for f in freq])

        # Handle iterable node
        if isinstance(node, Iterable) and not isinstance(node, (str, bytes)):
            return np.array([self.__call__(freq, n, delay=delay) for n in node])
        else:
            freq, node = self.encoder(freq), node
            if (self._is_data_exist(freq)): pass
            else: 
                exit = self.run(freq)
                if exit != 0:
                    raise RuntimeError(f"simulation error exit {exit}")
        return self.currents[(freq, node)]     # self.currents[(freq, node)] = current
    
    def _is_data_exist(self, freq_query):
        return freq_query in self.freqs
    
    def _add_freq(self, new_freq):
        return self.freqs.append(new_freq)
    
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
        "Now it will be just the mesh file"
        import shutil
        from_dir = Path(from_dir)
        to_dir   = Path(to_dir)
        to_dir.mkdir(parents=True, exist_ok=True)
        files_to_move = list(from_dir.glob(f"{self.model_name}*.*"))
        for f in files_to_move:
            if f.is_file():
                shutil.copy2(f, to_dir / f.name)
        return 0
    
    def _load_current_data(self, current_file: str, freq: float):
        freq = self.encoder(freq)
        current_data = Path(current_file).open('r').readlines()[1:]
        if len(current_data) != len(self.nodes): 
            if not len(self.nodes) == 0:
                current_data_from_file = '\n'.join(Path(current_file).open('r').readlines())
                raise RuntimeError(
                    "data unmatched with queried node ID:",
                    f"datasize {len(current_data)} != request {len(self.nodes)}",
                    f"at frequency {freq}",
                    f"farfield_file: {Path(current_file).absolute().__str__()}",
                    f"{current_data_from_file}",
                    sep='\n'
                )
        def parse_complex(line:str):
            clean_line = line.strip("\n").strip("(").strip(")")
            return complex(*map(float, clean_line.split(",")))
        for i, d in enumerate(current_data):
            self.currents[(freq, i)] = parse_complex(d)
        self.freqs.append(freq)
        return 0
    
    def run(self, freq_query):
        from subprocess import Popen, STDOUT, PIPE
        print(f"\n ======  Adding Frequency Sample: {freq_query}  ====== \n")
        freq_query = self.encoder(freq_query)
        org_path = os.getcwd()
        os.chdir(self.workingpath)
        self._setup_simulation(freq_query)
        log_file_path = f"{self.model_name}.log"
        log_monitor_path = f"{self.workingpath}/compute.log"

        # Run simulation
        log_file = open(log_file_path, "w")
        log_monitor = open(log_monitor_path, "w")
        # Start the process
        prog = Popen(
            ["bash", self.solver, freq_query, self.args],
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
        self._load_current_data(f"./I.mat", freq_query)
        self._exit_simulation(freq_query, org_path)
        return 0
    
    def get_node_ids(self):
        mesh_name = self.model_name + ".line"
        with open(Path(self.workingpath)/mesh_name, 'r') as f:
            scale = f.readline()
            n_nodes = f.readline()
        node_ids = np.arange(int(n_nodes))
        return node_ids
    

if __name__=="__main__":
    workingpath = "./data/MoM-data/test"
    model_name = "test"
    app = OnFlySolverMyMoM(
        workingpath, model_name,
        init_freqs=[200e6]
    )
    print(app([100e6, 200e6], 10))