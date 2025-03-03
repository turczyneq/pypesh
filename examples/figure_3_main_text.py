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


ball_radius = 0.92
maximal_radius = 0.3
peclet = 10**5
# fem_cross = np.array([[0, 1], [0.1, 0.9], [0.2, 0.8]])
fem_cross = visual.draw_cross_section_fem(
    peclet, ball_radius, maximal_radius=maximal_radius
)

"""
WARNING trials 10**4 is expensive in computation time
"""
traj_cross = visual.draw_cross_section_traj(
    peclet,
    ball_radius,
    mesh_out=4,
    mesh_jump=10,
    trials=10000,
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

spread = 4
stream_radius = sf.streamline_radius(5, ball_radius)


def dispersion(peclet):
    """
    Fucntion used to estimate dispesion
    """
    return 10 * (1 / peclet) ** (1 / 2)


# Plot all data
fontsize = 26 * 0.96 * (53.249 / 52.022)
marker_size = 80
plt.rcParams.update({"text.usetex": True, "font.family": "Times"})
fig, axes = plt.subplots(2, 1, figsize=(10, 11), height_ratios=[1, 1])

"""
FIRST PLOT
"""

# Plot our data
(our_work,) = axes[0].loglog(
    numerical[:, 0],
    numerical[:, 1],
    label="This work",
    color="k",
    linewidth=4,
    zorder=2,
)

# Plot Clift data
(clift_approx,) = axes[0].loglog(
    peclet_values,
    analytic_clift,
    label="Clift et al. (fit.)",
    color=tableau[7],
    linestyle="-",
    linewidth=3,
    zorder=1,
)

#
# EXPERIMENTAL
#

kuta = axes[0].scatter(
    kutateladze[:, 2],
    kutateladze[:, 3],
    label="Kutateladze et al. (exp.)",
    color=tableau[1],
    marker="o",
    s=marker_size,
    edgecolor="k",
    zorder=3,
)

kra = axes[0].scatter(
    kramers[:, 2],
    kramers[:, 3],
    label="Kramers et al. (exp.)",
    color=tableau[6],
    marker="s",
    s=marker_size,
    edgecolor="k",
    zorder=3,
)

#
# NUMERICAL
#

west = axes[0].scatter(
    westerberg[:, 2],
    westerberg[:, 3],
    label="Westerberg \& Finlayson (sim.)",
    color=tableau[9],
    marker="s",
    s=marker_size,
    edgecolor="k",
    zorder=3,
)

clift_num = axes[0].scatter(
    numerical_clift[:, 2],
    numerical_clift[:, 3],
    label="Clift et al. (sim.)",
    color=tableau[0],
    marker="o",
    s=marker_size,
    edgecolor="k",
    zorder=3,
)

frie = axes[0].scatter(
    friedlander[:, 2],
    friedlander[:, 3],
    label="Friedlander (sim.)",
    color=tableau[4],
    marker="D",
    s=marker_size,
    edgecolor="k",
    zorder=3,
)

feng = axes[0].scatter(
    feng[:, 2],
    feng[:, 3],
    label="Feng et al. (sim.)",
    color=tableau[3],
    marker="D",
    s=marker_size,
    edgecolor="k",
    zorder=3,
)

leg1 = [our_work, clift_approx, clift_num, west, frie, feng]
leg2 = [kuta, kra]

# Legend
legend1 = axes[0].legend(
    handles=leg1,
    frameon=False,
    labelspacing=0.2,
    handlelength=0.6,
    loc=(0, 0.35),
    prop={"family": "Times", "size": fontsize},
)

legend2 = axes[0].legend(
    handles=leg2,
    frameon=False,
    labelspacing=0.2,
    handlelength=0.6,
    loc=(0.5, 0),
    prop={"family": "Times", "size": fontsize},
)

axes[0].add_artist(legend1)
axes[0].add_artist(legend2)

axes[0].set_xscale("log")
axes[0].set_yscale("log")
axes[0].set_xlim(0.44, 3 * 10**4)
axes[0].set_ylim(0.8, 50)
axes[0].tick_params(axis="both", labelsize=fontsize)

axes[0].set_xlabel(r"Peclet number $\textrm{Pe}$", fontsize=fontsize)
axes[0].set_ylabel(r"Sherwood number $\textrm{Sh}$", fontsize=fontsize)


"""
SECOND PLOT
"""

axes[1].scatter(
    traj_cross[:, 0],
    traj_cross[:, 1],
    color='k',
    linestyle="None",
    s=marker_size,
    marker="o",
    edgecolor="k",
    label=rf"pychastic for $\textrm{{Pe}} = 10^{round(np.log10(peclet))}$",
)
axes[1].plot(
    fem_cross[:, 0],
    fem_cross[:, 1],
    color='k',
    linestyle="-",
    ms=2,
    label=rf"scikit-fem for $\textrm{{Pe}} = 10^{round(np.log10(peclet))}$",
)

vertical_lines = [
    stream_radius - spread * dispersion(peclet),
    stream_radius - dispersion(peclet),
    stream_radius + dispersion(peclet),
    stream_radius + spread * dispersion(peclet),
]

axes[1].vlines(
    vertical_lines,
    [-0.11],
    [1.2],
    color="0.6",
    linestyles="--",
)

axes[1].vlines(
    [stream_radius],
    [0],
    [1.2],
    color="0.2",
    linestyles="--",
)

axes[1].hlines(
    [0],
    [-1],
    [1.2],
    color="0.2",
    # linestyles="--",
    lw=0.1
)

text_height = -0.09

axes[1].text(
    (vertical_lines[0] + vertical_lines[1]) / 2,
    text_height,
    "Coarse",
    ha="center",
    fontsize=fontsize,
)
axes[1].text(
    (vertical_lines[1] + vertical_lines[2]) / 2,
    text_height,
    "Fine",
    ha="center",
    fontsize=fontsize,
)
axes[1].text(
    (vertical_lines[2] + vertical_lines[3]) / 2,
    text_height,
    "Coarse",
    ha="center",
    fontsize=fontsize,
)


axes[1].set_xlim(0, maximal_radius - 0.005)
axes[1].set_ylim(-0.11, 1.2)
axes[1].tick_params(axis="both", labelsize=fontsize)

axes[1].set_xlabel(r"Distance from axis $\rho$", fontsize=fontsize)
axes[1].set_ylabel(r"Hitting probability $p_{\textrm{hit}}$", fontsize=fontsize)

axes[1].legend(
    fontsize=fontsize,
    frameon=True,
    facecolor="white",
    framealpha=0.8,
    edgecolor="none",
    loc=(0.45, 0.69),
)

for i, x in enumerate([r"(a)", r"(b)"]):
    axes[i].text(
        0.01,
        0.92,
        x,
        transform=axes[i].transAxes,
        # ha="center",
        # va="center",
        color="k",
        fontsize=fontsize,
        backgroundcolor="#ffffffc0",
    )


plt.tight_layout()
tosave = parent_dir / "graphics/literature_comparison_and_spread.pdf"
plt.savefig(tosave, bbox_inches="tight", pad_inches=0.02)

# Show plot
# plt.show()
