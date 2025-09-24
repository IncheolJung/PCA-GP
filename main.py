import sys
import warnings
import numpy as np

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
    parser = ArgumentParser()
    parser.add_argument(
        "-a", "--ac-fx", "--acquisition-function", dest="ac_fx", default="0", 
        help=f"type of acquisition function [{ac_fx_types}]")
    parser.add_argument(
        "-x", "--xn", "--X-normalizer", dest="xn", default="2", 
        help=f"type of X-normalizer [{xnorm_types}]")
    parser.add_argument(
        "-n", "--n-init", dest="n", default="3", 
        help=f"initial number of frequency samples")
    parser.add_argument(
        "-t", "--terms", dest="t", default="3", 
        help=f"number of terms [stacked-RBFP and LF-NSM]")
    parser.add_argument(
        "-s", "--sample", "--sampling-strategy", dest="s", default="0", 
        help=f"type of sampling strategy [{sampling_types}]")
    parser.add_argument(
        "-i", "--max-iter", dest="i", default="20", 
        help=f"maximum iteration. maximum samples = n_init + max_iter")
    parser.add_argument(
        "-T", "--tol", dest="tol", default="1e-5", 
        help=f"tolerance for varinace. terminates iteration if tol > var")
    return parser.parse_args()


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
    n_init = int(config.n)
    terms  = int(config.t)
    max_iter = int(config.i)
    tol = float(config.tol)

    # -----------------------
    # SOLVER
    # -----------------------

    solver = fileIOdatareader("data/data-for-kenny-paper-HH.npz")
    # solver = fileIOdatareader("data/data-for-kenny-paper-VV.npz")

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


    # -----------------------
    # GP TRAINER
    # -----------------------
    # from gp_sklearn import train_gp_sklearn as trainer
    from gp_Kenny import train_model_gp_Kenny as trainer
    # from gp_Kenny_from_mode import train_model_gp_Kenny_from_mode as trainer

    # -----------------------
    # GP MODEL
    # -----------------------
    model = ReducedBasisGP1D
    # model = ReducedBasisGP2D
    # model = ReducedBasisGPMultiTask

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
    print(f"\n ======  Simulation {simulation_number} Initialized  ====== \n")
    stdout.flush()
    # -----------------------
    # BEGIN
    # -----------------------
    f_min, f_max, f_num = 9500, 10500, 101
    # f_min, f_max, f_num = 500, 1500, 151
    # f_test = np.linspace(f_min, f_max, 101)
    f_test = np.linspace(f_min, f_max, f_num)
    angles = np.linspace(0, 180, 181)  # 181 angles
    # angles = np.linspace(0, 180, 19)
    rbgp = model(
        solver, trainer, angles, n_init=n_init, r=n_init, adaptive_r=adaptive_basis, 
        acquisition_type=acquisition_function, Xnormalizer_type=Xnormalizer_type, 
        terms=terms, verbose=True
    )
    rbgp.initialize(f_min=f_min, f_max=f_max, sampling_strategy=0)
    make_pretty_number = lambda freq: str(round(freq, 3))
    pretty_number = list(map(make_pretty_number,rbgp.freqs))
    print(f"\n  >> Initial Frequencies: {pretty_number}\n")

    # max_iter = len(f_test) - n_init
    for it in range(max_iter):  # 5 adaptive iterations
        f_next, ac_fx, avg_var = rbgp.acquisition_next_frequency(f_min, f_max, len(f_test))
        print(f"\nIteration {it+1} / {max_iter}: acquisition {ac_fx:.10f} | variance {avg_var:.10f}")
        print("number of frequency samples:", len(rbgp.freqs))
        if avg_var < tol: 
            break
        print(f"sampling new frequency {f_next:.3f}")
        rbgp.update(f_next)
        stdout.flush()
    print("\n ======  Stopping criterion met.  ====== \n")
    print("  >> Final iteration:", it+1, "/", max_iter, sep="\t")
    print("  >> Final acquisition:", ac_fx, sep="\t")
    print("  >> Final variance:", avg_var, sep="\t")
    print("  >> total n_freq:", len(rbgp.freqs), sep="\t")
    stdout.flush()

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
    error_real = 10 * np.log10(numer_real / denom_real)
    error_imag = 10 * np.log10(numer_imag / denom_imag)
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
    hx[0,2].set_title("RMSE RE [dB]")
    hx[1,2].set_title("RMSE IM [dB]")
    for i in range(nrows): 
        for j in range(ncols):
            hf.colorbar(im[i,j], ax=hx[i,j])
    hf.savefig(dir_out/"GP_results_2d.png")
    ######### end 2d plot #########

    # pltshow()
    return 0


if __name__ == "__main__":
    main()