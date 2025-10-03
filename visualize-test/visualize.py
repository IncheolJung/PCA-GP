from numpy import ndarray, array, linspace, argmax

from pathlib import Path
from matplotlib.pyplot import subplots, Axes


def read_Feko(datapath: Path):
    return array([line.strip().split() for line in datapath.open("r").readlines()[2:]], dtype=float)


def read_MoM(datapath: Path):
    def parse_complex(line: str):
        line = line.strip()
        i0, i1, i2 = line.find("("), line.find(","), line.find(")")
        return float(line[i0+1:i1]), float(line[i1+1:i2])
        # if "," in line:     # C++ out
        #     i0, i1, i2 = line.find("("), line.find(","), line.find(")")
        #     return float(line[i0+1:i1]), float(line[i1+1:i2])
        # else:
        #     complex_value = complex(line)
        #     return complex_value.real, complex_value.imag
    return array([[i+1, *parse_complex(line)] for i, line in enumerate(datapath.open("r").readlines()[1:])], dtype=float)

def get_err(data, ref):
    if len(data) == len(ref):
        err1 = (data[:,1] - ref[:,1])**2
        err2 = (data[:,2] - ref[:,2])**2
        div1 = (ref[:,1].max() - ref[:,1].min())**2
        div2 = (ref[:,2].max() - ref[:,2].min())**2
        rel_SE = array([ref.T[0], err1/div1, err2/div2]).T
    else:
        rel_SE = array([[], [], []])
        raise Exception(
            f"WARNING: dimension mismatch ({len(data)}!={len(ref)}). I.mat needs re-computation.")
    return rel_SE

# def plot(mom_data: ndarray):
#     fig, ax = subplots(figsize=(16,9), constrained_layout=True)
#     ax: Axes
#     ax.plot(mom_data.T[0].astype(int), mom_data.T[1], "r-", lw=1, alpha=0.6, label="MoM (Re)")
#     ax.plot(mom_data.T[0].astype(int), mom_data.T[2], "b-", lw=1, alpha=0.6, label="MoM (Im)")
#     ax.set_xlim([mom_data.T[0].min(), mom_data.T[0].max()])
#     # ax.set_ylim([-0.03, 0.03])
#     # ax.set_ylim([1.1 * mom_data.T[1].min(), 1.1 * mom_data.T[1].max()])
#     ax.set_xticks(linspace(min(mom_data.T[0]), max(mom_data.T[0]), 11, dtype=int))
#     # ax.set_yticks(linspace(min(mom_data.T[1]), max(mom_data.T[1]), 7, dtype=float))
#     ax.legend()
#     return fig


def plot(mom_data: ndarray):
    fig, ax1 = subplots(figsize=(16, 9), constrained_layout=True)
    ax1: Axes

    # Left y-axis → Real part
    ax1.plot(mom_data.T[0].astype(int), mom_data.T[1], "r-", lw=1, alpha=0.6, label="MoM (Re)")
    ax1.set_ylabel("Real part", color="r")
    ax1.tick_params(axis="y", labelcolor="r")

    # Right y-axis → Imaginary part
    ax2 = ax1.twinx()
    ax2.plot(mom_data.T[0].astype(int), mom_data.T[2], "b-", lw=1, alpha=0.6, label="MoM (Im)")
    ax2.set_ylabel("Imaginary part", color="b")
    ax2.tick_params(axis="y", labelcolor="b")

    # X-axis setup
    ax1.set_xlim([mom_data.T[0].min(), mom_data.T[0].max()])
    ax1.set_xticks(linspace(min(mom_data.T[0]), max(mom_data.T[0]), 11, dtype=int))

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    return fig


def plot_overlap(feko_data: ndarray, mom_data: ndarray):
    fig, ax = subplots(figsize=(6,5), constrained_layout=True)
    ax: Axes
    ax.plot(feko_data.T[0].astype(int), feko_data.T[1], "r-", label="Feko (Re)")
    ax.plot(mom_data.T[0].astype(int), mom_data.T[1], "rs", label="MoM (Re)")
    ax.plot(feko_data.T[0].astype(int), feko_data.T[2], "b-", label="Feko (Im)")
    ax.plot(mom_data.T[0].astype(int), mom_data.T[2], "bs", label="MoM (Im)")
    ax.set_xlim([feko_data.T[0].min(), feko_data.T[0].max()])
    # ax.set_ylim([1.1 * feko_data.T[1].min(), 1.1 * feko_data.T[1].max()])
    ax.set_xticks(linspace(min(feko_data.T[0]), max(feko_data.T[0]), 11, dtype=int))
    # ax.set_yticks(linspace(min(feko_data.T[1]), max(feko_data.T[1]), 7, dtype=float))
    ax.legend()
    return fig


def plot_sep(feko_data: ndarray, mom_data: ndarray):
    fig, ax = subplots(figsize=(11,5), ncols=2, constrained_layout=True)
    ax: ndarray[Axes]
    ax[0].plot(feko_data.T[0].astype(int), feko_data.T[1], "r-", label="Feko (Re)")
    ax[1].plot(mom_data.T[0].astype(int), mom_data.T[1], "r:", label="MoM (Re)")
    ax[0].plot(feko_data.T[0].astype(int), feko_data.T[2], "b-", label="Feko (Im)")
    ax[1].plot(mom_data.T[0].astype(int), mom_data.T[2], "b:", label="MoM (Im)")
    # ax[0].plot(feko_data.T[0].astype(int), (feko_data.T[1]**2 + feko_data.T[2]**2)**.5, "r-", label="Feko (amp)")
    # ax[1].plot(mom_data.T[0].astype(int), (mom_data.T[1]**2 + mom_data.T[2]**2)**.5, "r:", label="MoM (amp)")
    for ax_i in ax:
        ax_i.set_xlim([feko_data.T[0].min(), feko_data.T[0].max()])
        # ax_i.set_ylim([1.1 * feko_data.T[1].min(), 1.1 * feko_data.T[1].max()])
        ax_i.set_xticks(linspace(min(feko_data.T[0]), max(feko_data.T[0]), 11, dtype=int))
        # ax_i.set_yticks(linspace(min(feko_data.T[1]), max(feko_data.T[1]), 7, dtype=float))
        ax_i.legend()
    return fig


def args_parse():
    from argparse import ArgumentParser
    args = ArgumentParser()
    args.add_argument("-e", "--error", action="store_true", 
                      help="flag to calculate error")
    args.add_argument("-c", "--compare", action="store_true", 
                      help="flag to calculate error between BF vs. ST solvers")
    args.add_argument("-s", "--scale", default="5.0",
                      help="quantity used to compute characteristic impedance for center-fed antenna")
    args.add_argument("-f", "--freq", default="10e6",
                      help="quantity used to compute characteristic impedance for center-fed antenna")
    args.add_argument("-a", "--radius", default="1e-5",
                      help="quantity used to compute characteristic impedance for center-fed antenna")
    return args.parse_args()


def main():
    args = args_parse()
    mat_names = [
        "bin/I.mat",
        "I_reconstructed.mat",
        # "bin/I-Eigen.mat",
        # "bin/I_STRUM_COMBINE.mat",
        # "bin/I_BF_f90.mat",
    ]
    fig_names = [
        Path(m.replace("bin/", "").replace("I", "I_profile", 1)).with_suffix(".png") 
        for m in mat_names
        ]
    for mat_name, fig_name in zip(mat_names, fig_names):
        mom_data = read_MoM(datapath = Path(mat_name))
        ref_data = read_MoM(datapath = Path("bin/I.mat"))
        plot(mom_data).savefig(fig_name)
        print("  Plot for {} saved to {}".format(mat_name, fig_name))

        # if mat_name=="bin/I.mat":
        if mat_name=="I_reconstructed.mat":
            # Characteristic Impedance (Theoretical)
            l, wvlth = 0.3*float(args.scale), 3e8/float(args.freq)
            pi = 3.141592
            # https://phys.libretexts.org/Bookshelves/Electricity_and_Magnetism/Electromagnetics_II_(Ellingson)/10%3A_Antennas/10.06%3A_Impedance_of_the_Electrically-Short_Dipole
            from numpy import log
            Rrad = 20 * pi**2 * (l/wvlth)**2
            Xrad = - (120 * wvlth) / (pi * l) * log(2*l/float(args.radius))
            print("  >> Z_in (short-dipole) = {:.2e}".format(Rrad+1j*Xrad))
            # https://en.wikipedia.org/wiki/Dipole_antenna#Induced_EMF_method
            from scipy.special import sici
            from numpy import sin, cos
            coef = 377 / (2*pi*sin(pi/wvlth * l)**2)
            gamma_e = 0.57721566    # Euler's constant
            ln_hkl, ln_kl = log(pi/wvlth * l), log(2*pi/wvlth * l)
            sin_kl, cos_kl = sin(2*pi/wvlth * l), cos(2*pi/wvlth * l)
            si_kl, ci_kl   = sici(2*pi/wvlth * l)
            si_2kl, ci_2kl = sici(4*pi/wvlth * l)
            si__, ci__ = sici(4*pi/wvlth * float(args.radius)**2 / l)
            Rdip = coef*(gamma_e + ln_kl - ci_kl + 0.5*sin_kl * (si_2kl-2*si_kl) + 0.5*cos_kl * (ci_2kl-2*ci_kl+gamma_e+ln_hkl))
            Xdip = coef*(si_kl + 0.5*cos_kl * (-si_2kl+2*si_kl) + 0.5*sin_kl * (ci_2kl-2*ci_kl+ci__))
            print("  >> Z_in (Induced EMF)  = {:.2e}".format(Rdip+1j*Xdip))

            # Characteristic Impedance (Numerical)
            feed_id = int(len(mom_data)/2)
            Zin = 1 / (mom_data[feed_id,1]+1j*mom_data[feed_id,2])
            # id1 = int(len(mom_data)/2/2*1)
            # id2 = int(len(mom_data)/2/2*3)
            # I_feed_from_j = (mom_data[id1,1]+1j*mom_data[id1,2])
            # I_feed_from_rho = -1j*2*pi*float(args.freq)*(mom_data[id2,1]+1j*mom_data[id2,2])
            # Zin = 1 / (I_feed_from_j+I_feed_from_rho)
            print("  >> Z_in (numerical)    = {:.2e}".format(Zin))

        if args.error:
            err = get_err(mom_data, ref_data)
            plot(err).savefig(fig_name.with_stem(f"{fig_name.stem}_ERR"))
            print(f"  >> RMSE = {(err[:,1].mean()**2+err[:,2].mean()**2)**.5:.4%}")
    if args.compare:
        mom_data = read_MoM(datapath = Path("bin/I_BF_f90.mat"))
        ref_data = read_MoM(datapath = Path("bin/I_STRUM_COMBINE.mat"))
        fig_name = Path("I_profile_COMPARE_ERR.png")
        err = get_err(mom_data, ref_data)
        plot(err).savefig(fig_name)
        print(f"  >> RMSE = {(err[:,1].mean()**2+err[:,2].mean()**2)**.5:.4%}")
    # plot(read_MoM(datapath = Path("bin/I.mat"))).savefig("I_profile.png")
    # # plot(read_MoM(datapath = Path("bin/I_STRUM1.mat"))).savefig("I_profile_STRUM1.png")
    # # plot(read_MoM(datapath = Path("bin/I_STRUM4.mat"))).savefig("I_profile_STRUM4.png")
    # # plot(read_MoM(datapath = Path("bin/I_STRUM16.mat"))).savefig("I_profile_STRUM16.png")
    # plot(read_MoM(datapath = Path("bin/I_STRUM_COMBINE.mat"))).savefig("I_profile_STRUM_COMBINE.png")
    # plot(read_MoM(datapath = Path("bin/I_BF_f90.mat"))).savefig("I_profile_BF_f90.png")
    # # plot_sep(read_Feko(datapath = Path("Feko/02.dat")), read_MoM(datapath = Path("bin/I.mat"))).savefig("Feko/MoM_vs_Feko.png")
    # # plot_sep(read_Feko(datapath = Path("Feko/02.dat")), read_MoM(datapath = Path("bin/test_I.mat"))).savefig("Feko/MoM_vs_Feko.png")
    # # for i in range(10):
    # #     plot(read_MoM(datapath = Path(f"bin/I_BF_f90_{i}.mat"))).savefig(f"I_profile_BF_f90_{i}.png")
    return


if __name__ == "__main__":
    main()
