import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import groupby
import pypesh.analytic as analytic

parent_dir = Path(__file__).parent

fem_path = parent_dir / "fem_pe_vs_sh.csv"
fem = np.loadtxt(fem_path, delimiter=",", skiprows=1)
# we have to split into listst wiht different r_syf
fem_sorted = fem[fem[:, 1].argsort()]
fem_grouped = groupby(fem_sorted, key=lambda x: x[1])
fem_plt = {k: np.array(list(g)) for k, g in fem_grouped}

py_path = parent_dir / "py_pe_vs_sh.csv"
py = np.loadtxt(py_path, delimiter=",", skiprows=1)
# we have to split into listst wiht different r_syf
py_sorted = py[py[:, 1].argsort()]
py_grouped = groupby(py_sorted, key=lambda x: x[1])
py_plt = {k: np.array(list(g)) for k, g in py_grouped}


fem_difference = {
    key: np.array(
        [
            [
                peclet,
                ball_radius,
                (sherwood - analytic.our_approximation(peclet, ball_radius)) / sherwood,
            ]
            for peclet, ball_radius, sherwood in value
        ]
    )
    for key, value in fem_plt.items()
}

py_difference = {
    key: np.array(
        [
            [
                peclet,
                ball_radius,
                (sherwood - analytic.our_approximation(peclet, ball_radius)) / sherwood,
            ]
            for peclet, ball_radius, sherwood in value
        ]
    )
    for key, value in py_plt.items()
}

# Plot all data

fontsize = 26
marker_size = 80
plt.figure(figsize=(16 * 0.85, 9 * 0.85))
plt.rcParams.update({"text.usetex": True, "font.family": "Times"})


femdots = [
    plt.scatter(
        data[:, 0],
        data[:, 2],
        label=f"${1-data[1,1]:5.3f}$",
        color=f"C{i}",
        zorder=1,
    )
    for i, data in enumerate(fem_difference.values())
]

for i, data in enumerate(py_difference.values()):
    plt.scatter(data[:, 0], data[:, 2], color=f"C{i}", facecolors="none", zorder=0)


# # Plot Clift data
# (clift,) = plt.loglog(
#     peclet_values,
#     analytic_clift,
#     label="Clift et al. (analytic)",
#     color="k",
#     linestyle="-",
#     linewidth=2,
#     zorder=0,
# )

# add dummy plt to make legend

fem = plt.scatter([0], [0], label="scikit-fem", color="k")

traj = plt.scatter([0], [0], label=r"pychastic", color="k", facecolors="none")

empty = plt.scatter([0], [0], label=" ", color="none", facecolors="none")

utils = [fem, traj]

# femlegend
legend1 = plt.legend(
    handles=femdots + utils,
    fontsize=fontsize,
    frameon=False,
    labelspacing=0.3,
    handlelength=0.3,
    loc=(0.01, 0.7),
    ncols=4,
)

plt.text(
    2,
    0.37,
    r"$\beta = $",
    ha="center",
    fontsize=fontsize,
)

plt.gca().add_artist(legend1)

# Logarithmic scale
plt.xscale("log")
plt.xlim(0.5, 10**12)
plt.ylim(-0.2, 0.4)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)

# Labels and Title
plt.xlabel(r"Peclet number $\left(Pe\right)$", fontsize=fontsize)
plt.ylabel(r"Relative error $\left(\frac{Sh - Sh_{\mathrm{f}}}{Sh}\right)$", fontsize=fontsize)

plt.tight_layout()
tosave = parent_dir.parent / "graphics/numerical_vs_analytic.pdf"
plt.savefig(tosave)

# Show plot
plt.show()
