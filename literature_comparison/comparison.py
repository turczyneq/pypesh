import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.colors as mcolors
tableau = list(mcolors.TABLEAU_COLORS)

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


# Plot all data
fontsize = 26 * 1.2
marker_size = 80
plt.figure(figsize=(16 * 0.85, 9 * 0.85))
plt.rcParams.update({"text.usetex": True, "font.family": "Times"})

# Plot our data
(our_work,) = plt.loglog(
    numerical[:, 0],
    numerical[:, 1],
    label="This work",
    color="k",
    linewidth=4,
    zorder=2,
)

# Plot Clift data
(clift_approx,) = plt.loglog(
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

kuta = plt.scatter(
    kutateladze[:, 2],
    kutateladze[:, 3],
    label="Kutateladze et al. (exp.)",
    color=tableau[1],
    marker="o",
    s=marker_size,
    edgecolor="k",
    zorder=3,
)

kra = plt.scatter(
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

west = plt.scatter(
    westerberg[:, 2],
    westerberg[:, 3],
    label="Westerberg \& Finlayson (sim.)",
    color=tableau[9],
    marker="s",
    s=marker_size,
    edgecolor="k",
    zorder=3,
)

clift_num = plt.scatter(
    numerical_clift[:, 2],
    numerical_clift[:, 3],
    label="Clift et al. (sim.)",
    color=tableau[0],
    marker="o",
    s=marker_size,
    edgecolor="k",
    zorder=3,
)

frie = plt.scatter(
    friedlander[:, 2],
    friedlander[:, 3],
    label="Friedlander (sim.)",
    color=tableau[4],
    marker="D",
    s=marker_size,
    edgecolor="k",
    zorder=3,
)

feng = plt.scatter(
    feng[:, 2],
    feng[:, 3],
    label="Feng et al. (sim.)",
    color=tableau[3],
    marker="D",
    s=marker_size,
    edgecolor="k",
    zorder=3,
)


# Logarithmic scale
plt.xscale("log")
plt.yscale("log")
plt.xlim(0.44, 3 * 10**4)
plt.ylim(0.8, 50)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)

# Labels and Title
plt.xlabel(r"Peclet number $\textrm{Pe}$", fontsize=fontsize)
plt.ylabel(r"Sherwood number $\textrm{Sh}$", fontsize=fontsize)

leg1 = [our_work, clift_approx, clift_num, west, frie, feng]
leg2 = [kuta, kra]

# Legend
legend1 = plt.legend(
    handles=leg1,
    # fontsize=fontsize,
    frameon=False,
    labelspacing=0.2,
    handlelength=0.6,
    loc=(0, 0.48),
    prop={'family':"Times", 'size':fontsize}
)

legend2 = plt.legend(
    handles=leg2,
    # fontsize=fontsize,
    frameon=False,
    labelspacing=0.2,
    handlelength=0.6,
    loc=(0.5, 0),
    # loc=(0, 0.44),
    prop={'family':"Times", 'size':fontsize}
)

plt.gca().add_artist(legend1)
plt.gca().add_artist(legend2)

# plt.tight_layout()
tosave = parent_dir / "literature_comparison.pdf"
plt.savefig(tosave, bbox_inches="tight", pad_inches=0.02)

# Show plot
plt.show()
