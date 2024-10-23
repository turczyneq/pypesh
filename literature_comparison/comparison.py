import numpy as np
import matplotlib.pyplot as plt
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


# Plot all data
fontsize = 15
plt.figure(figsize=(7, 7))
plt.rcParams.update({"text.usetex": True, "font.family": "Cambria"})

# Plot our data
plt.loglog(
    numerical[:, 0],
    numerical[:, 1],
    label="This work",
    color="k",
    linewidth=4,
    zorder=2,
)

# Plot Clift data
plt.loglog(
    peclet_values,
    analytic_clift,
    label="Clift et al. (analytic)",
    color="C7",
    linestyle="-",
    linewidth=3,
    zorder=1,
)

#
# NUMERICAL
#

plt.scatter(
    numerical_clift[:, 2],
    numerical_clift[:, 3],
    label="Clift et al. (numerical)",
    color="C0",
    marker="o",
    s=50,
    edgecolor="k",
    zorder=3,
)

plt.scatter(
    friedlander[:, 2],
    friedlander[:, 3],
    label="Friedlander (numerical)",
    color="C4",
    marker="D",
    s=50,
    edgecolor="k",
    zorder=3,
)

plt.scatter(
    westerberg[:, 2],
    westerberg[:, 3],
    label="Westerberg \& Finlayson (numerical)",
    color="C9",
    marker="s",
    s=50,
    edgecolor="k",
    zorder=3,
)

plt.scatter(
    feng[:, 2],
    feng[:, 3],
    label="Feng et al. (numerical)",
    color="C3",
    marker="D",
    s=50,
    edgecolor="k",
    zorder=3,
)

#
# EXPERIMENTAL
#

plt.scatter(
    kutateladze[:, 2],
    kutateladze[:, 3],
    label="Kutateladze et al. (experimental)",
    color="C1",
    marker="o",
    s=50,
    edgecolor="k",
    zorder=3,
)

plt.scatter(
    kramers[:, 2],
    kramers[:, 3],
    label="Kramers et al. (experimental)",
    color="C6",
    marker="s",
    s=50,
    edgecolor="k",
    zorder=3,
)

# Logarithmic scale
plt.xscale("log")
plt.yscale("log")
plt.xlim(0.44, 5 * 10**4)
plt.ylim(0.8, 30)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)

# Labels and Title
plt.xlabel(r"Peclet Number $\left(\mathrm{Pe}\right)$", fontsize=fontsize)
plt.ylabel(r"Sherwood Number $\left(\mathrm{Sh}\right)$", fontsize=fontsize)

# Legend
plt.legend(fontsize=fontsize, frameon=False)
plt.tight_layout()
tosave = parent_dir.parent / "graphics/ignore/literature_comparison.pdf"
plt.savefig(tosave)

# Show plot
plt.show()
