import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

tableau = list(mcolors.TABLEAU_COLORS)

import pypesh.visualisation as visual
import pypesh.stokes_flow as sf

from pathlib import Path


parent_dir = Path(__file__).parent

numerical_path = parent_dir / "data/this_work.csv"
numerical = np.loadtxt(numerical_path, delimiter=",", skiprows=1)


def clift_approximation(pe):
    return (1 / 2) * (1 + (1 + 2 * pe) ** (1 / 3))


peclet_values = np.logspace(-1, 5, 300)
analytic_clift = clift_approximation(peclet_values)

numerical_clift_path = parent_dir / "data/clift.csv"
numerical_clift = np.loadtxt(numerical_clift_path, delimiter=",", skiprows=1)

friedlander_path = parent_dir / "data/friedlander.csv"
friedlander = np.loadtxt(friedlander_path, delimiter=",", skiprows=1)

kutateladze_path = parent_dir / "data/kutateladze.csv"
kutateladze = np.loadtxt(kutateladze_path, delimiter=",", skiprows=1)

feng_path = parent_dir / "data/feng.csv"
feng = np.loadtxt(feng_path, delimiter=",", skiprows=1)

kramers_path = parent_dir / "data/kramers.csv"
kramers = np.loadtxt(kramers_path, delimiter=",", skiprows=1)

westerberg_path = parent_dir / "data/westerberg.csv"
westerberg = np.loadtxt(westerberg_path, delimiter=",", skiprows=1)


ball_radius = 0.91
maximal_radius = 0.3
peclet = 7 * 10**5
# fem_cross = np.array([[0, 1], [0.1, 0.9], [0.2, 0.8]])
fem_cross = visual.draw_cross_section_fem(
    peclet, ball_radius, maximal_radius=maximal_radius
)
# fem_cross = np.array([[0, 1], [0.1, 0.9], [0.2, 0.8]])

"""
WARNING trials 10**4 is expensive in computation time
"""
traj_cross = visual.draw_cross_section_traj(
    peclet,
    ball_radius,
    mesh_out=15,
    mesh_jump=20,
    trials=10000,
    spread=10,
)

"""
to test plot setting
"""
# traj_cross = visual.draw_cross_section_traj(
#     peclet,
#     ball_radius,
#     mesh_out=4,
#     mesh_jump=10,
#     trials=200,
# )

# traj_cross = np.array([[0, 1], [0.1, 0.9], [0.2, 0.8]])

spread = 10
stream_radius = sf.streamline_radius(5, ball_radius)


def dispersion(peclet):
    """
    Fucntion used to estimate dispesion
    """
    return 10 * (1 / peclet) ** (1 / 2)


fontsize = 26 * 1.2 * 0.8 * 0.8 / 0.7 * 1.12
marker_size = 80
plt.figure(figsize=(16 * 0.85, 9 * 0.85))
plt.rcParams.update({"text.usetex": True, "font.family": "Times"})

plt.scatter(
    traj_cross[:, 0],
    traj_cross[:, 1],
    color='k',
    linestyle="None",
    s=marker_size,
    marker="o",
    edgecolor="k",
    label=rf"pychastic",
)
plt.plot(
    fem_cross[:, 0],
    fem_cross[:, 1],
    color='k',
    linestyle="-",
    ms=2,
    label=rf"scikit-fem",
)

plt.scatter(
    [-1,-1],
    [-1,-1],
    color='w',
    label=r"$\textrm{\textit{Pe}}=7 \times 10^5$",
)

plt.scatter(
    [-1,-1],
    [-1,-1],
    color='w',
    label=r"$\beta=0.09$",
)


vertical_lines = [
    stream_radius - spread * dispersion(peclet),
    stream_radius - dispersion(peclet),
    stream_radius + dispersion(peclet),
    stream_radius + spread * dispersion(peclet),
]

plt.vlines(
    vertical_lines,
    [-0.11],
    [1.2],
    color="0.6",
    linestyles="--",
)

plt.vlines(
    [stream_radius],
    [0],
    [1.2],
    color="0.2",
    linestyles="--",
)

plt.hlines(
    [0],
    [-1],
    [1.2],
    color="0.2",
    # linestyles="--",
    lw=0.1
)

text_height = -0.09

plt.text(
    (vertical_lines[0] + vertical_lines[1]) / 2,
    text_height,
    r"\texttt{coarse}",
    ha="center",
    fontsize=fontsize,
)
plt.text(
    (vertical_lines[1] + vertical_lines[2]) / 2,
    text_height,
    r"\texttt{fine}",
    ha="center",
    fontsize=fontsize,
)
plt.text(
    (vertical_lines[2] + vertical_lines[3]) / 2,
    text_height,
    r"\texttt{coarse}",
    ha="center",
    fontsize=fontsize,
)


plt.xlim(0, maximal_radius - 0.005)
plt.ylim(-0.11, 1.2)
plt.tick_params(axis="both", labelsize=fontsize)

plt.xlabel(r"Distance from axis $\rho$", fontsize=fontsize)
plt.ylabel(r"Hitting probability $p_{\textrm{hit}}$", fontsize=fontsize)

plt.legend(
    fontsize=fontsize,
    frameon=True,
    facecolor="white",
    framealpha=0.8,
    edgecolor="none",
    loc=(0.6, 0.5),
)

# for i, x in enumerate([r"(a)", r"(b)"]):
#     axes[i].text(
#         0.01,
#         0.92,
#         x,
#         transform=axes[i].transAxes,
#         # ha="center",
#         # va="center",
#         color="k",
#         fontsize=fontsize,
#         backgroundcolor="#ffffffc0",
#     )


plt.tight_layout()
tosave = parent_dir / "graphics/regions.pdf"
plt.savefig(tosave, bbox_inches="tight", pad_inches=0.02)