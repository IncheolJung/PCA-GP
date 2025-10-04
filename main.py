import sys
import warnings
import numpy as np

from itertools import product

from datareader import *
from ReducedBasisGP import *


def configuration():
    from argparse import ArgumentParser
    ac_fx_types = [
        "maximum variance", "expected improvement", "upper confidence bound"
        ]
    xnorm_types = [
        "pass", "z-score", "min-max", "power transform", 
        "standardized power transform", "scaled z-score"
        ]
    sampling_types = [
        "grid", "latin-hyper-cube"
        ]
    sweeping_types = [
        "phi", "theta"
        ]
    parser = ArgumentParser()
    parser.add_argument(
        "-p", "--path-simulation", dest="path", default=None, 
        help=f"User defined simulation directory")
    parser.add_argument(
        "-m", "--model-name", dest="model", default=None, 
        help=f"model name (stem of .domain file) to simulate")
    parser.add_argument(
        "-A", "--angle", "--angle-settings", nargs=3, 
        default=["0", "180", "181"], 
        help=f"[start, end, number] of the angular sweep")
    parser.add_argument(
        "-S", "--ast", "--angular-sweep-type", default="0", 
        help=f"type of angular sweep {sweeping_types}")
    parser.add_argument(
        "-F", "--freq", "--frequency-settings", nargs=3, 
        default=["9500", "10500", "101"], 
        help=f"[start, end, number] of the frequency sweep")
    parser.add_argument(
        "-v", "--validate", action="store_true", 
        help=f"flag for validation")
    parser.add_argument(
        "-a", "--ac-fx", "--acquisition-function", dest="ac_fx", 
        default="0", 
        help=f"type of acquisition function {ac_fx_types}")
    parser.add_argument(
        "-x", "--xn", "--X-normalizer", dest="xn", default="1", 
        help=f"type of X-normalizer {xnorm_types}")
    parser.add_argument(
        "-n", "--n-init", dest="n", default="3", 
        help=f"initial number of frequency samples")
    parser.add_argument(
        "--add", "--freq-add", default="1",
        help=f"number of added frequency samples per Gaussian Process iteration")
    parser.add_argument(
        "-t", "--terms", dest="t", default="3", 
        help=f"number of terms [stacked-RBFP and LF-NSM]")
    parser.add_argument(
        "-s", "--sample", "--sampling-strategy", dest="s", default="0", 
        help=f"type of sampling strategy {sampling_types}")
    parser.add_argument(
        "-i", "--max-iter", dest="i", default="20", 
        help=f"maximum iteration. maximum samples = n_init + max_iter")
    parser.add_argument(
        "-T", "--tol", dest="tol", default="1e-3", 
        help=f"tolerance for varinace. terminates iteration if tol > var")
    return parser.parse_args()


def validate(*args):

    rbgp, f_test, solver, stdout, dir_out = args

    # Predict at new frequency
    pred  = rbgp.reconstruct(f_test)
    # truth = solver(f_test, rbgp.angles)
    truth = np.array([solver(f, a) for f, a in product(f_test, rbgp.angles)])  # (n_init, n_angles)
    truth = truth.reshape(len(f_test), len(rbgp.angles))  # (n_init, n_angles)
    print("\n ======  Final Statistics  ====== \n")
    print("Predicted response shape:", pred.shape)
    print("RMSE:{:.12f}".format(np.mean(np.square(np.abs(pred - truth))) / np.mean(np.square(np.abs(truth)))) )
    stdout.flush()

    from matplotlib.pyplot import subplots, show as pltshow
    from matplotlib import rcParams
    rcParams["text.usetex"] = True
    rcParams["font.size"] = 12

    ########### 1d plot ###########
    hf, hx = subplots(nrows=2, figsize=(12,8), constrained_layout=True)
    for i in range(pred.shape[0]):
        hx[0].plot(rbgp.angles, pred.real[i],  'r-', alpha=0.5, label="PRED RE"  if i == 0 else None)
        hx[0].plot(rbgp.angles, truth.real[i], 'b-', alpha=0.5, label="TRUTH RE" if i == 0 else None)
    for i in range(pred.shape[0]):
        hx[1].plot(rbgp.angles, pred.imag[i],  'r-', alpha=0.5, label="PRED IM"  if i == 0 else None)
        hx[1].plot(rbgp.angles, truth.imag[i], 'b-', alpha=0.5, label="TRUTH IM" if i == 0 else None)
    hf.legend()
    hf.savefig(dir_out/"GP_results_1d.png")
    ######### end 1d plot #########

    ########### 2d plot ###########
    nrows, ncols = 2, 3
    hf, hx = subplots(nrows=nrows, ncols=ncols, figsize=(16,8), constrained_layout=True)
    extent = [rbgp.angles.min(), rbgp.angles.max(), f_test.min(), f_test.max()]
    im = np.empty((nrows,ncols), dtype="object")
    eps = 1e-12  # small number to avoid log(0)
    numer_real = np.square(np.abs(pred.real - truth.real))
    denom_real = np.square(np.max(truth.real) - np.min(truth.real))
    numer_imag = np.square(np.abs(pred.imag - truth.imag))
    denom_imag = np.square(np.max(truth.imag) - np.min(truth.imag))
    np.seterr(all="raise")
    try:
        error_real = 10 * np.log10(numer_real / denom_real + eps*denom_real)
        error_imag = 10 * np.log10(numer_imag / denom_imag + eps*denom_imag)
        error_scale = "dB"
    except FloatingPointError:
        error_real = numer_real
        error_imag = numer_imag
        error_scale = "linear"
    np.seterr()
    print(f"Error scale: {error_scale}")
    error = error_real + 1j*error_imag
    im[0,0] = hx[0,0].imshow(pred.real,  cmap="turbo", aspect="auto", extent=extent)
    im[0,1] = hx[0,1].imshow(truth.real, cmap="turbo", aspect="auto", extent=extent)
    im[1,0] = hx[1,0].imshow(pred.imag,  cmap="turbo", aspect="auto", extent=extent)
    im[1,1] = hx[1,1].imshow(truth.imag, cmap="turbo", aspect="auto", extent=extent)
    im[0,2] = hx[0,2].imshow(error.real,  cmap="hot", aspect="auto", extent=extent)
    im[1,2] = hx[1,2].imshow(error.imag, cmap="hot", aspect="auto", extent=extent)
    hx[0,0].set_title("PRED RE")
    hx[0,1].set_title("TRUTH RE")
    hx[1,0].set_title("PRED IM")
    hx[1,1].set_title("TRUTH IM")
    hx[0,2].set_title(f"RMSE RE [{error_scale}]")
    hx[1,2].set_title(f"RMSE IM [{error_scale}]")
    for i in range(nrows): 
        for j in range(ncols):
            hf.colorbar(im[i,j], ax=hx[i,j])
    hf.savefig(dir_out/"GP_results_2d.png")
    ######### end 2d plot #########
    return 0


# # --- angular PCA --- #
# def export(file_path, np_data, freqs, theta, phi):
#     def make_col(data): return " ".join(map(str, data))
#     def make_row(data): return "\n".join(map(str, data))
#     np_data_ravel = np_data.reshape(-1)
#     angles = list(zip(theta, phi))
#     header = ["Freq", "Theta", "Phi", "Cpol(Re)", "Cpol(Im)"]
#     data = [make_col([f, th, ph, np_data_ravel[i].real, np_data_ravel[i].imag]) 
#             for i, (f, (th, ph)) in enumerate(product(freqs, angles))]
#     data = make_row([make_col(header), *data])
#     return Path(file_path).open('w').write(data)


# --- current PCA --- #
def export(file_path, np_data, freqs, nodes):
    def make_col(data): return " ".join(map(str, data))
    def make_row(data): return "\n".join(map(str, data))
    def parse_complex(data): return f"({data.real},{data.imag})"
    exitcode = []
    header = make_col([np_data.shape[-1], 1])
    for f, data in zip(freqs, np_data):
        data = list(map(parse_complex, data))
        data = make_row([header, *data])
        exitcode.append((Path(file_path)/f"{f}.mat").open('w').write(data))
    return any(exitcode)


def main():
    config = configuration()
    # -----------------------
    # CONFIG
    # -----------------------
    adaptive_basis = True
    acquisition_function = int(config.ac_fx)
    acquisition_function_candidate = [
        "maximum variance", "expected improvement", "upper confidence bound"
    ]
    Xnormalizer_type = int(config.xn)
    Xnormalizer_type_candidate = [
        "pass", "z-score", "min-max", "power transform", 
        "standardized power transform", "scaled z-score"
    ]
    sampling_type = int(config.s)
    sampling_type_candidate = [
        "grid", "latin-hyper-cube"
        ]
    sweep_type = int(config.ast)
    sweep_type_candidate = [
        "phi", "theta"
        ]
    n_init = int(config.n)
    terms  = int(config.t)
    max_iter = int(config.i)
    tol = float(config.tol)
    f_min, f_max, f_num = *map(float, config.freq[:2]), int(config.freq[2])
    a_min, a_max, a_num = *map(float, config.angle[:2]), int(config.angle[2])
    angles = np.linspace(a_min, a_max, a_num)  # 181 angles

    # -----------------------
    # SOLVER
    # -----------------------

    # solver = fileIOdatareader("data/data-for-kenny-paper-HH.npz")
    # solver = fileIOdatareader("data/data-for-kenny-paper-VV.npz")

    # workingpath = config.path
    # if workingpath is None:
    #     workingpath = "./data/VWT-data/circylinder"
    #     print(
    #         "\n"*2, "#"*50, '\n', 
    #         "WORKING_PATH not supplied. ", '\n',
    #         "Fall back to default:", workingpath, '\n', 
    #         "#"*50, "\n"*2,
    #         )

    # solver = OnFlySolver(
    #     workingpath="./data/VWT-data/sphere",
    #     model_name="sphere",
    #     angles=np.linspace(0, 180, 181)
    #     )

    # solver = OnFlySolver(
    #     workingpath="./data/VWT-data/prime-airplane",
    #     model_name="Open-Duct_PRIME_model_meshAA",
    #     angles=np.linspace(0, 180, 181)
    #     )

    # solver = OnFlySolver(
    #     workingpath="./data/VWT-data/circylinder",
    #     model_name="circylinder",
    #     angles=np.linspace(0, 180, 181),
    #     sweep_angle_type=1
    #     )

    # solver = OnFlySolver(
    #     workingpath=workingpath,
    #     model_name=config.model,
    #     angles=angles, 
    #     sweep_angle_type=sweep_type
    #     )

    solver = OnFlySolverMyMoM(
        workingpath="./data/MoM-data/test",
        model_name="test"
        )
    angles = solver.get_node_ids()
    a_min, a_max, a_num = np.min(angles), np.max(angles), len(angles)

    # -----------------------
    # GP TRAINER
    # -----------------------
    # from gp_sklearn import train_gp_sklearn as trainer
    from gp_Kenny import train_model_gp_Kenny as trainer
    # from gp_Kenny_from_mode import train_model_gp_Kenny_from_mode as trainer

    # -----------------------
    # GP MODEL
    # -----------------------
    # model = ReducedBasisGP1D
    # model = ReducedBasisGP2D
    model = ReducedBasisGPMultiTask

    # -----------------------
    # OUTPUT DRIECTORY SETUP
    # -----------------------
    dir_out = Path("out")
    simulation_number = len([d for d in dir_out.glob("*") if d.is_dir()]) + 1
    dir_out = dir_out/Path(f"GP_test_{simulation_number:04d}")
    dir_out.mkdir()
    stdout = Tee(dir_out/"GP_results.log", "w")     # Log file setup
    print(f"\n ======  Simulation {simulation_number} Initialized  ====== \n")
    print(f"  >> Adaptive basis: {adaptive_basis}")
    print(f"  >> Acquisition function: {acquisition_function_candidate[acquisition_function]}")
    print(f"  >> X normalization strategy: {Xnormalizer_type_candidate[Xnormalizer_type]}")
    print(f"  >> Sampling strategy: {sampling_type_candidate[sampling_type]}")
    print(f"  >> n_init: {n_init}")
    print(f"  >> terms: {terms}")
    print(f"  >> max_iter: {max_iter}")
    print(f"  >> tol: {tol}")
    print(f"  >> angles {sweep_type_candidate[sweep_type]}: [start, end, number] = {a_min, a_max, a_num}")
    print(f"\n ======  Simulation {simulation_number} Initialized  ====== \n")
    stdout.flush()
    # -----------------------
    # BEGIN
    # -----------------------
    # f_min, f_max, f_num = 100, 300, 101
    # f_min, f_max, f_num = 9500, 10500, 101
    # f_min, f_max, f_num = 500, 1500, 151
    # angles = np.linspace(0, 180, 181)  # 181 angles
    # angles = np.linspace(0, 180, 19)
    rbgp = model(
        solver, trainer, angles, n_init=n_init, r=n_init, adaptive_r=adaptive_basis, 
        acquisition_type=acquisition_function, 
        Xnormalizer_type=Xnormalizer_type, normalizeY=True, 
        terms=terms, verbose=True
    )
    rbgp.initialize(f_min=f_min, f_max=f_max, sampling_strategy=sampling_type)
    make_pretty_number = lambda freq: str(round(freq, 3))
    pretty_number = list(map(make_pretty_number,rbgp.freqs))
    print(f"\n  >> Initial Frequencies: {pretty_number}\n")

    # max_iter = len(f_test) - n_init
    for it in range(max_iter):  # 5 adaptive iterations
        f_next, ac_fx, POD_energy = \
            rbgp.acquisition_next_frequency(f_min, f_max, f_num, int(config.add))
        print(f"\nIteration {it+1} / {max_iter}: max_acquisition {max(ac_fx):.10f} | pred_to_total_POD_energy_ratio {POD_energy:.10f}")
        print("number of frequency samples:", len(rbgp.freqs))
        if POD_energy < tol: 
            break
        f_next_str = " ".join(["[", *[f"{f:.3f}" for f in f_next], "]"])
        print(f"sampling new frequencies:", f_next_str)
        rbgp.update(f_next)
        stdout.flush()
    print("\n ======  Stopping criterion met.  ====== \n")
    print("  >> Final iteration:", it+1, "/", max_iter, sep="\t")
    print("  >> Final acquisition:", max(ac_fx), sep="\t")
    print("  >> Final pred_to_total_POD_energy_ratio:", POD_energy, sep="\t")
    print("  >> total n_freq:", len(rbgp.freqs), sep="\t")
    stdout.flush()

    f_export = np.linspace(f_min, f_max, f_num)
    # export("freq_sweep.efar", rbgp.reconstruct(f_export), f_export, solver.theta, solver.phi)
    export("CURRENT/", rbgp.reconstruct(f_export), f_export, solver.nodes)

    if config.validate: 
        # f_test = np.linspace(f_min, f_max, 101)
        f_test = np.linspace(f_min, f_max, f_num)
        validate(rbgp, f_test, solver, stdout, dir_out)

    # pltshow()
    return 0


if __name__ == "__main__":
    main()