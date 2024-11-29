import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import groupby

parent_dir = Path(__file__).parent

fem_path = parent_dir / "data" / "fem_pe_vs_sh.csv"
fem = np.loadtxt(fem_path, delimiter=",", skiprows=1)
fem_radius_zero = fem[fem[:, 1] == 1]
# we have to split into listst wiht different r_syf
fem = fem[fem[:, 0] < 10**6]
fem_sorted = fem[fem[:, 1].argsort()]
fem_sorted = fem_sorted[fem_sorted[:, 1] != 1]
fem_grouped = groupby(fem_sorted, key=lambda x: x[1])
fem_plt = {k: np.array(list(g)) for k, g in fem_grouped}

py_path = parent_dir / "data" / "py_pe_vs_sh.csv"
py = np.loadtxt(py_path, delimiter=",", skiprows=1)
# we have to split into listst wiht different r_syf
py_sorted = py[py[:, 1].argsort()]
py_sorted = py_sorted[py_sorted[:, 1] != 1]
py_grouped = groupby(py_sorted, key=lambda x: x[1])
py_plt = {k: np.array(list(g)) for k, g in py_grouped}


def clift_approximation(pe):
    return (1 / 2) * (1 + (1 + 2 * pe) ** (1 / 3))


peclet_values = np.logspace(-1, 12, 300)
analytic_clift = clift_approximation(peclet_values)


# Plot all data

fontsize = 28
plt.figure(figsize=(7.5, 6.75))
plt.rcParams.update({"text.usetex": True, "font.family": "Times"})


empty = plt.scatter([0], [0], label=" ", color="none", facecolors="none")

femdots = [
    plt.scatter(
        data[:, 0],
        data[:, 2],
        label=f"${1-data[1,1]:5.3f}$",
        color=f"C{i}",
        zorder=1,
    )
    for i, data in enumerate(fem_plt.values())
]

beta_zero = plt.scatter(
    fem_radius_zero[:, 0],
    fem_radius_zero[:, 2],
    label=f"${1-fem_radius_zero[1,1]:5.3f}$",
    color=f"C{len(fem_plt.values())}",
    zorder=1,
)

num = 0
for data in py_plt.values():
    # Plot our data
    plt.scatter(data[:, 0], data[:, 2], color=f"C{num}", facecolors="none", zorder=0)
    num += 1


# Plot Clift data
(clift,) = plt.loglog(
    peclet_values,
    analytic_clift,
    label="Clift et al. (analytic)",
    color="k",
    linestyle="-",
    linewidth=2,
    zorder=0,
)

# add dummy plt to make legend

fem = plt.scatter([0], [0], label="scikit-fem", color="k")

traj = plt.scatter([0], [0], label=r"pychastic", color="k", facecolors="none")

femdots = [empty] + femdots + [beta_zero]

# femlegend
legend1 = plt.legend(
    handles=femdots,
    fontsize=fontsize,
    frameon=False,
    labelspacing=0.3,
    handlelength=0.3,
    loc=(0.01, 0.1),
)

plt.text(
    10,
    8 * 10**9,
    r"$\beta = $",
    ha="center",
    fontsize=fontsize,
)

utils = [empty, clift, fem, traj]

legend2 = plt.legend(
    handles=utils,
    fontsize=fontsize,
    frameon=False,
    loc=(0.3, 0.7),
    labelspacing=0.3,
    handlelength=0.8,
)

plt.gca().add_artist(legend1)
plt.gca().add_artist(legend2)

# Logarithmic scale
plt.xscale("log")
plt.yscale("log")
plt.xlim(0.5, 10**12)
plt.ylim(0.9, 8 * 10**10)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)

# Labels and Title
plt.xlabel(r"Peclet number $\left(Pe\right)$", fontsize=fontsize)
plt.ylabel(r"Sherwood number $\left(Sh\right)$", fontsize=fontsize)

plt.tight_layout()
tosave = parent_dir / "graphics/sh_vs_pe.pdf"
plt.savefig(tosave)

# Show plot
plt.show()
