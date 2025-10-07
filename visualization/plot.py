# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import pyvista as pv
import numpy as np
from pathlib import Path
import struct


PREFIX = "  >> "


def parse_complex(value_str: str):
    value_comp = value_str.strip('(').strip(')').split(',')
    return complex(*map(float, value_comp))


def read_mesh(filename):
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    idx = 0
    scale = float(lines[idx]); idx += 1

    n_points = int(lines[idx]); idx += 1
    points = []
    for _ in range(n_points):
        x, y, z = map(float, lines[idx].split())
        points.append((x * scale, y * scale, z * scale))
        idx += 1

    n_edges = int(lines[idx]); idx += 1
    edges = []
    for _ in range(n_edges):
        i, j = map(int, lines[idx].split())
        edges.append((i, j))
        idx += 1

    n_dirichlet = int(lines[idx]); idx += 1
    dirichlet = []
    for _ in range(n_dirichlet):
        index, value = lines[idx].split()
        dirichlet.append((int(index), parse_complex(value)))
        idx += 1

    return points, edges, dirichlet


def read_current(filename, dirichlet):
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    idx = 0
    nrows, ncols = map(int, lines[idx].split()); idx += 1
    data = np.empty((nrows, ncols), dtype=complex)
    for i in range(nrows):
        line_parsed = map(parse_complex, lines[idx].split()); idx += 1
        data[i] = np.array(list(line_parsed), dtype=complex)
    # data = 0.5 * (np.concatenate([[0], data]) + np.concatenate([data, [0]]))
    # idx_to_remove = np.array([i for i, d in dirichlet], dtype=int)
    # idx_to_remove = np.unique(np.concatenate([idx_to_remove, idx_to_remove+1]))[1:-1]
    return data.squeeze()


# def __build_Dmat(Nj, Nc):
#     Dmat = np.zeros((Nj, Nc), dtype=complex)
#     for j in range(Nc):
#         Dmat[j,   j] += 1.0
#         Dmat[j+1, j] -= 1.0
#     return


# def __build_Dmat(Nj, Nc):
#     "Assum AEFIE and then use Z_rJ sub-block as Dmat"
#     def str_to_complex(entry: str):
#         return complex(*entry.strip("(").strip(")").strip(" ").split(","))
#     Z_name = Path("bin/Z.mat")
#     data = [line.strip("\n").split(" ") for line in Z_name.open("r").readlines()]
#     data = np.array([map(str_to_complex, line) for line in data[:Nj, Nj:]])
#     assert(data.shape[0]==Nj)
#     assert(data.shape[1]==Nc)
#     return data
TYPE_SIZE = 32  # must match C++ TYPE_SIZE

def __build_Dmat(Nj, Nc):
#     "Assum AEFIE and then use Z_rJ sub-block as Dmat"
    Z_name = Path("bin/Z.mat")
    Dmat = np.zeros((Nj, Nc), dtype=complex)
    with Z_name.open("rb") as f:
        # Read type (not used)
        type_buf = f.read(TYPE_SIZE)
        type_str = type_buf.decode('utf-8').rstrip('\x00')

        # Read dimensions
        nrows = struct.unpack("Q", f.read(8))[0]  # size_t assumed 8 bytes
        ncols = struct.unpack("Q", f.read(8))[0]

        assert nrows >= Nj and ncols >= Nj + Nc

        # Read entire matrix as complex<double> (myComplex)
        # Assuming myComplex = std::complex<double> = two doubles (real, imag)
        raw = f.read(nrows * ncols * 16)  # 16 bytes per complex
        data = np.frombuffer(raw, dtype=np.complex64).reshape(nrows, ncols)

    # Extract Dmat as Z_rJ sub-block (rows 0:Nj, cols Nj:Nj+Nc)
    # Dmat = data[:Nj, Nj:Nj+Nc]
    Dmat = data[Nj:Nj+Nc, :Nj].T   # Z_rhoJ
    assert(Dmat.shape[0] == Nj)
    assert(Dmat.shape[1] == Nc)
    assert(type_str == "AEFIE_MATRIX")
    return Dmat


def __post_process_current_AEFIE(current, freq):
    Nj = (len(current)+1)//2    # len(current) = 2*Nj - 1
    Nc = Nj-1
    assert(Nj+Nc == len(current))
    omega = 2 * np.pi * freq
    jomega = 1j * omega
    C = 3e8
    jk = jomega / C
    k_scalar = omega / C
    # J / rho scaling assumed (J := jk J && rho := C rho)
    # Jscale, rhoscale = 1, 1 / (omega / C)
    mu0 = 4 * np.pi * 1e-7
    eps0 = 1.0 / (mu0 * C * C)
    Z0 = np.sqrt(mu0/eps0)
    Jscale, rhoscale = k_scalar, 1 / omega
    potential_scale, continuity_scale = 1, 1 / Z0
    J_sol, rho_sol = (1/Jscale) * current[:Nj], (1/rhoscale) * current[Nj:]
    Dmat = Jscale * continuity_scale * __build_Dmat(Nj, Nc)  # Dmat is already scaled
    current_out = J_sol - (1/jomega) * (Dmat @ rho_sol)
    # current_out = J_sol
    return current_out

# def __post_process_current_AEFIE(current, freq):
#     # inputs: Dmat_physical (Nj x Nc), current (length Nj+Nc), freq
#     jomega = 1j*2*np.pi*freq
#     C = 3e8
#     Nj = (len(current)+1)//2
#     Nc = Nj-1
#     Jp = current[:Nj]    # scaled J' (you use J := jk J' so be careful which)
#     rhop = current[Nj:]  # scaled rho'
#     Dmat = __build_Dmat(Nj, Nc)     # read in Z_rJ
#     DT_physical = C * C * __build_Dmat(Nj, Nc).T
    
#     print("max |real(Jp)|, max |imag(Jp)|:", np.max(np.abs(np.real(Jp))), np.max(np.abs(np.imag(Jp))))
#     print("max |real(rhop)|, max |imag(rhop)|:", np.max(np.abs(np.real(rhop))), np.max(np.abs(np.imag(rhop))))

#     # if Dmat is the physical D (not Z_rhoJ), use:
#     res = DT_physical @ (Jp) - (jomega/C**2) * rhop   # scaled form from earlier derivations
#     print("continuity relres:", np.linalg.norm(res)/ (np.linalg.norm(rhop)+1e-30))

#     jk = jomega / C
#     J_from_J = (1/jk) * Jp                     # direct from scaled J
#     # reconstruct via rho: J_from_rho = -(1/(jomega*C))*D_physical @ (rhop)
#     J_from_rho = -(1/(jomega*C)) * (DT_physical.T @ rhop)

#     print("rel diff J_from_J vs J_from_rho:", np.linalg.norm(J_from_J - J_from_rho)/ (np.linalg.norm(J_from_J)+1e-30))
    
#     return (1/jk) * current[:Nj] - (1/jomega) * (Dmat @ current[Nj:])  # Dmat is already scaled



def sanity_check_current_with_mesh(current, points, freq = 100e6):
    if len(current) == len(points):
        current_out = current
    elif len(current) == 2*len(points)-1:
        current_out = __post_process_current_AEFIE(current, freq)
        print(f"{PREFIX}AEFIE solution assumed: solution merged at {freq:.3e} Hz")
    else:
        msg = f"Unmatched current and mesh: current.size() [{len(current)}] != mesh.size() [{len(points)}]"
        raise RuntimeError(msg)
    return current_out


def add_network_to_plotter(plotter, points, edges, dirichlet, current=None, title=None):
    """Adds network visualization to a PyVista plotter"""
    points = np.array(points)
    lines = []
    for i, j in edges:
        lines.extend([2, i, j])

    mesh = pv.PolyData(points, lines=np.array(lines))

    scalar_name = "_".join([title, "current"])

    sargs = dict(
        title=scalar_name,
        height=0.40,
        width=0.10,
        vertical=True,
        position_x=0.80,
        position_y=0.30,
        label_font_size=8,
        title_font_size=10,
        shadow=True,
        n_labels=5,
        fmt="%.4e"
    )

    if current is not None:
        mesh.point_data[scalar_name] = np.array(current)
        mesh = mesh.point_data_to_cell_data()
        plotter.add_mesh(
            mesh,
            scalars=scalar_name,
            cmap="turbo",
            show_scalar_bar=True,
            scalar_bar_args=sargs,
            below_color="grey",
            clim=[np.min(current), np.max(current)],  # independent range
            line_width=3
        )
    else:
        plotter.add_mesh(mesh, color='black', line_width=3)
        plotter.add_points(points, color='black', point_size=3, render_points_as_spheres=True)

        if dirichlet:
            d_indices = [d[0] for d in dirichlet]
            plotter.add_points(points[d_indices], color='black', point_size=3, render_points_as_spheres=True)

    if title:
        plotter.add_text(title, position='upper_left', font_size=12, color='black')

    plotter.add_axes(
        line_width=5,
        cone_radius=0.6,
        shaft_length=0.7,
        tip_length=0.3,
        ambient=0.5,
        label_size=(0.4, 0.16),
    )

    return 0



def visualize_with_pyvista(
        points, edges, dirichlet, current=None, figure_path=None
    ):

    # Pack mesh for compactness
    mesh = points, edges, dirichlet

    # Create a single plotter with 3 subplots
    plotter = pv.Plotter(shape=(1, 4), window_size=[2400, 600], off_screen=True)
    
    # Plot absolute values
    plotter.subplot(0, 0)
    add_network_to_plotter(plotter, *mesh, np.abs(current), title="ABS")
    
    # Plot absolute values
    plotter.subplot(0, 1)
    add_network_to_plotter(plotter, *mesh, 10*np.log10(np.abs(current)+1e-12), title="ABS [dB]")
    
    # Plot real components
    plotter.subplot(0, 2)
    add_network_to_plotter(plotter, *mesh, np.real(current), title="REAL")
    
    # Plot imaginary components
    plotter.subplot(0, 3)
    add_network_to_plotter(plotter, *mesh, np.imag(current), title="IMAG")
    
    # Link all cameras so zooming/panning affects all subplots
    plotter.link_views()
    
    # Display the combined plot
    # plotter.show()
    plotter.screenshot(figure_path)
    return 0


def plot_node_vs_current(current):
    from matplotlib.axes import Axes
    from matplotlib.pyplot import subplots
    fig, ax1 = subplots(figsize=(16, 9), constrained_layout=True)
    ax1: Axes

    # Left y-axis → Real part
    ax1.plot(current.real, "r-", lw=1, alpha=0.6, label="MoM (Re)")
    ax1.set_ylabel("Real part", color="r")
    ax1.tick_params(axis="y", labelcolor="r")

    # Right y-axis → Imaginary part
    ax2 = ax1.twinx()
    ax2.plot(current.imag, "b-", lw=1, alpha=0.6, label="MoM (Im)")
    ax2.set_ylabel("Imaginary part", color="b")
    ax2.tick_params(axis="y", labelcolor="b")

    # X-axis setup
    ax1.set_xmargin(0)
    # ax1.set_xlim([current[0].min(), current[0].max()])
    # ax1.set_xticks(np.linspace(np.min(current[0]), np.max(current[0]), 11, dtype=int))

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    return fig


def export_current(current, filename: str):
    filename = Path(filename)
    filename: Path

    def _c_style_out(current):
        return f"({current.real},{current.imag})"

    header = " ".join(map(str, [len(current), 1]))
    writer = "\n".join([header, *list(map(_c_style_out, current))])

    filename.open("w").write(writer)
    print(f"Current data stored at {filename}")
    return 0


def argparser():
    from argparse import ArgumentParser
    parser = ArgumentParser(prog="Read in mesh and current to visualize current in 3D")
    parser.add_argument("MESH", help="mesh file (.line) to identify geometry")
    parser.add_argument("CURRENT", help="current file (I.mat) to plot")
    parser.add_argument("-o", "--out", help="filename to save the plot (default: I_profile.png)", 
                        default="I_profile.png")
    parser.add_argument("-f", "--freq", default=1e8, 
                        help="current frequency. only used for combining AEFIE solution. " \
                             "defaults to 100e6")
    args = parser.parse_args()
    args_dict = {
        "mesh_path":        Path(args.MESH),
        "current_path":     Path(args.CURRENT),
        "export_path":      Path(args.out),
        "frequency":        float(args.freq),
    }
    return args_dict


def main():
    args = argparser()
    mesh_path, current_path = args["mesh_path"], args["current_path"]
    export_path, frequency = args["export_path"], args["frequency"]
    points, edges, dirichlet = read_mesh(mesh_path)
    current = read_current(current_path, dirichlet)
    current = sanity_check_current_with_mesh(current, points, freq=frequency)
    fig_path = current_path.with_suffix(".png")
    visualize_with_pyvista(points, edges, dirichlet, current, fig_path)
    plot_node_vs_current(current).savefig(export_path)
    print(f"{PREFIX}Solution ({len(current)}) plotted and saved to {export_path}")
    export_current(current, "I_reconstructed.mat")
    return 0


if __name__ == "__main__": main()
