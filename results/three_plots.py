import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import groupby
import pypesh.analytic as analytic
import matplotlib.colors as mcolors
tableau = list(mcolors.TABLEAU_COLORS)

parent_dir = Path(__file__).parent

"""sherwood versus peclet"""

fem_path = parent_dir / "data" / "fem_pe_vs_sh.csv"
fem = np.loadtxt(fem_path, delimiter=",", skiprows=1)
fem_radius_zero = fem[fem[:, 1] == 1]
fem_radius_zero = fem_radius_zero[fem_radius_zero[:, 0] < 10**7]
# we have to split into listst wiht different r_syf
fem = fem[fem[:, 0] < 10**6]
fem_sorted = fem[fem[:, 1].argsort()]
fem_sorted = fem_sorted[fem_sorted[:, 1] != 1]
fem_grouped = groupby(fem_sorted, key=lambda x: x[1])
fem_sherwood = {k: np.array(list(g)) for k, g in fem_grouped}

py_path = parent_dir / "data" / "py_pe_vs_sh.csv"
py = np.loadtxt(py_path, delimiter=",", skiprows=1)
py = py[py[:, 0] > 10**4]
# we have to split into listst wiht different r_syf
py_sorted = py[py[:, 1].argsort()]
py_sorted = py_sorted[py_sorted[:, 1] != 1]
py_grouped = groupby(py_sorted, key=lambda x: x[1])
py_sherwood = {k: np.array(list(g)) for k, g in py_grouped}


"""modified sherwood"""

fem_path = parent_dir / "data" / "mod_fem_pe_vs_sh.csv"
fem = np.loadtxt(fem_path, delimiter=",", skiprows=1)
fem = fem[fem[:, 0] < 10**6]
# we have to split into listst wiht different r_syf
fem_sorted = fem[fem[:, 1].argsort()]
fem_sorted = fem_sorted[fem_sorted[:, 1] != 1]
fem_grouped = groupby(fem_sorted, key=lambda x: x[1])
fem_modified_sherwood = {k: np.array(list(g)) for k, g in fem_grouped}

py_path = parent_dir / "data" / "mod_py_pe_vs_sh.csv"
py = np.loadtxt(py_path, delimiter=",", skiprows=1)
py = py[py[:, 0] > 10**5]
# we have to split into listst wiht different r_syf
py_sorted = py[py[:, 1].argsort()]
py_sorted = py_sorted[py_sorted[:, 1] != 1]
py_grouped = groupby(py_sorted, key=lambda x: x[1])
py_modified_sherwood = {k: np.array(list(g)) for k, g in py_grouped}


"""our approximations"""

fem_path = parent_dir / "data" / "fem_pe_vs_sh.csv"
fem = np.loadtxt(fem_path, delimiter=",", skiprows=1)

# fem = fem[fem[:, 0] < 5 * 10**6]
# we have to split into listst wiht different r_syf
fem_sorted = fem[fem[:, 1].argsort()]
fem_grouped = groupby(fem_sorted, key=lambda x: x[1])
fem_plt = {k: np.array(list(g)) for k, g in fem_grouped}
for i in [1, 0.999, 0.998, 0.995]:
    del fem_plt[i]

py_path = parent_dir / "data" / "py_pe_vs_sh.csv"
py = np.loadtxt(py_path, delimiter=",", skiprows=1)

# py = py[py[:, 0] > 10**4]
# we have to split into listst wiht different r_syf
py_sorted = py[py[:, 1].argsort()]
py_grouped = groupby(py_sorted, key=lambda x: x[1])
py_plt = {k: np.array(list(g)) for k, g in py_grouped}
for i in [1, 0.999, 0.998, 0.995]:
    del py_plt[i]


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

cutout = {
    0.8: (10**4, 5 * 10**2),
    0.9: (3 * 10**4, 1.5 * 10**4),
    0.95: (4.5 * 10**5, 5 * 10**4),
    0.98: (10**6, 7 * 10**5),
    0.99: (1.5 * 10**6, 1.5 * 10**6),
}

for key, value in cutout.items():
    fem_difference[key] = fem_difference[key][fem_difference[key][:, 0] < value[0]]
    py_difference[key] = py_difference[key][py_difference[key][:, 0] >= value[1]]


def clift_approximation(pe):
    return (1 / 2) * (1 + (1 + 2 * pe) ** (1 / 3))


peclet_values = np.logspace(-1, 12, 300)
analytic_clift = clift_approximation(peclet_values)


fontsize = 15
marker_size = 10

plt.rcParams.update({"text.usetex": True, "font.family": "Times"})
fig, axes = plt.subplots(3, 1, figsize=(10.5 * 0.6, 16 * 0.6), sharex=True)
# plt.figure(figsize=(16 * 0.85, 9 * 0.85))
# plt.rcParams.update({"text.usetex": True, "font.family": "Times"})

"""first plot"""

empty = axes[0].scatter(
    [0],
    [0],
    label=" ",
    color="none",
    facecolors="none",
    s=marker_size,
)

femdots = [
    axes[0].scatter(
        data[:, 0],
        data[:, 2],
        label=f"${1-data[1,1]:5.3f}$",
        color=tableau[i],
        zorder=1,
        s=marker_size,
    )
    for i, data in enumerate(fem_sherwood.values())
]

beta_zero = axes[0].scatter(
    fem_radius_zero[:, 0],
    fem_radius_zero[:, 2],
    label=f"${1-fem_radius_zero[1,1]:5.3f}$",
    color=tableau[len(fem_sherwood.values())],
    zorder=1,
    s=marker_size,
)

num = 0
for data in py_sherwood.values():
    # Plot our data
    axes[0].scatter(
        data[:, 0],
        data[:, 2],
        color=tableau[num],
        facecolors="none",
        zorder=0,
        s=marker_size,
    )
    num += 1


# Plot Clift data
(clift,) = axes[0].loglog(
    peclet_values,
    analytic_clift,
    label="Clift et al. (analytic)",
    color="k",
    linestyle="-",
    linewidth=2,
    zorder=0,
)

femdots = [empty] + femdots + [beta_zero]

# femlegend
legend1 = axes[0].legend(
    handles=femdots,
    fontsize=fontsize,
    frameon=False,
    labelspacing=0.1,
    handlelength=0.1,
    loc=(0.01, 0.35),
    ncols=2,
)

axes[0].text(
    0.08,
    0.76,
    r"$\beta = $",
    ha="center",
    fontsize=fontsize,
    transform=axes[0].transAxes,
)

axes[0].add_artist(legend1)


"""second plot"""

# Plot Clift data
(clift,) = axes[1].loglog(
    peclet_values,
    analytic_clift,
    label="Clift et al.",
    color="k",
    linestyle="-",
    linewidth=2,
    zorder=-1,
)

num = 0
for data in fem_modified_sherwood.values():
    # Plot our data
    axes[1].scatter(
        data[:, 0],
        data[:, 2],
        label=f"$\\beta = {round(10000*(1-data[1,1]))/10000}$",
        color=tableau[num],
        zorder=1,
        s=marker_size,
    )
    num += 1

num = 0
for data in py_modified_sherwood.values():
    # Plot our data
    axes[1].scatter(
        data[:, 0],
        data[:, 2],
        color=tableau[num],
        facecolors="none",
        zorder=0,
        s=marker_size,
    )
    num += 1

beta_zero = axes[1].scatter(
    fem_radius_zero[:, 0],
    fem_radius_zero[:, 2],
    label=f"${1-fem_radius_zero[1,1]:5.3f}$",
    color=tableau[len(fem_sherwood.values())],
    zorder=1,
    s=marker_size,
)

# for limits, modified sherwood should be 1
axes[1].plot(
    peclet_values,
    [1 for x in peclet_values],
    color="k",
    linestyle="--",
    linewidth=2,
    zorder=-1,
)


# add dummy plt to make legend

fem = axes[1].scatter(
    [0],
    [0],
    label="scikit-fem",
    color="k",
    s=marker_size,
)

traj = axes[1].scatter(
    [0],
    [0],
    label=r"pychastic",
    color="k",
    facecolors="none",
    s=marker_size,
)

utils = [clift, fem, traj]

legend2 = axes[1].legend(
    handles=utils,
    fontsize=fontsize,
    frameon=False,
    loc=(0.01, 0.55),
    labelspacing=0.1,
    handlelength=0.4,
)
axes[1].add_artist(legend2)


"""third plot"""


femdots = [
    axes[2].scatter(
        data[:, 0],
        data[:, 2],
        label=f"${1-data[1,1]:5.3f}$",
        color=tableau[i],
        zorder=-i,
        s=marker_size,
    )
    for i, data in enumerate(fem_difference.values())
]

for i, data in enumerate(py_difference.values()):
    axes[2].scatter(
        data[:, 0],
        data[:, 2],
        color=tableau[i],
        facecolors="none",
        zorder=0,
        s=marker_size,
    )


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

fem = axes[2].scatter(
    [0],
    [0],
    label="scikit-fem",
    color="k",
    s=marker_size,
)

traj = axes[2].scatter(
    [0],
    [0],
    label=r"pychastic",
    color="k",
    facecolors="none",
    s=marker_size,
)

empty = axes[2].scatter(
    [0],
    [0],
    label=" ",
    color="none",
    facecolors="none",
    s=marker_size,
)

utils = [fem, traj]


# # femlegend
# legend1 = axes[2].legend(
#     handles=femdots + utils,
#     fontsize=fontsize,
#     frameon=False,
#     labelspacing=0.3,
#     handlelength=0.3,
#     loc=(0.01, 0.7),
#     ncols=4,
# )

# axes[2].text(
#     2,
#     0.37,
#     r"$\beta = $",
#     ha="center",
#     fontsize=fontsize,
# )

# axes[2].add_artist(legend1)


"""legends"""

for ax in axes:
    ax.set_xscale("log")
    ax.set_xlim(5, 10**12)


for ax in axes[:2]:
    ax.set_yscale("log")

axes[0].set_ylim(0.5, 8 * 10**10)
axes[1].set_ylim(0.7, 80)
axes[2].set_ylim(-0.05, 0.18)

for i, x in enumerate([r"(a)", r"(b)", r"(c)"]):
    axes[i].text(0.01, 0.9, x, transform=axes[i].transAxes, fontsize=fontsize)


for ax in axes[1:3]:
    ax.tick_params(axis="x", labelsize=fontsize, top=True)
    ax.tick_params(axis="y", labelsize=fontsize)

axes[0].tick_params(axis="x", labelsize=fontsize)
axes[0].tick_params(axis="y", labelsize=fontsize)

for ax in axes:
    x_range = np.log10(ax.get_xlim()[1]) - np.log10(ax.get_xlim()[0])

    y_range = (
        np.log10(ax.get_ylim()[1]) - np.log10(ax.get_ylim()[0])
        if ax.get_yscale() == "log"
        else ax.get_ylim()[1] - ax.get_ylim()[0]
    )
    ax.set_aspect(0.53 * x_range / y_range)

# axes[0].set_ylabel(r"Sherwood number $\left(Sh\right)$", fontsize=fontsize)
# axes[1].set_ylabel(
#     r"Modified Sherwood number $\left(\widetilde{Sh}\right)$", fontsize=fontsize
# )
# axes[2].set_ylabel(
#     r"Relative error $\left(\frac{Sh - Sh_{\mathrm{f}}}{Sh}\right)$", fontsize=fontsize
# )

axes[0].set_ylabel(r"$Sh$", fontsize=fontsize)
axes[1].set_ylabel(r"$\widetilde{Sh}$", fontsize=fontsize)
axes[2].set_ylabel(r"$(Sh - Sh_{\mathrm{f}}) / Sh$", fontsize=fontsize)

axes[2].set_xlabel(
    r"Peclet number $\left(Pe\right)$",
    fontsize=fontsize,
)

# fig.text(
#     0.5,
#     0.04,
#     r"Peclet number $\left(Pe\right)$",
#     fontsize=fontsize,
#     ha="center",
#     va="center",
# )
# Labels and Title
# plt.xlabel(r"Peclet number $\left(Pe\right)$", fontsize=fontsize)

# fig.tight_layout()
plt.subplots_adjust(hspace=0)

tosave = parent_dir / "graphics/big_plot.pdf"
fig.savefig(tosave, bbox_inches="tight", pad_inches=0.02)

# Show plot
plt.show()
