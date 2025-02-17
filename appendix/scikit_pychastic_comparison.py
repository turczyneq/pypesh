import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import groupby
import pypesh.analytic as analytic
import matplotlib.colors as mcolors

tableau = list(mcolors.TABLEAU_COLORS)

parent_dir = Path(__file__).parent

data_dir = parent_dir.parent / "results"

"""modified sherwood"""

fem_path = data_dir / "data" / "fem_pe_vs_sh.csv"
fem = np.loadtxt(fem_path, delimiter=",", skiprows=1)
fem = fem[fem[:, 0] < 10**6]
fem = fem[fem[:, 0] > 10**3]
# we have to split into listst wiht different r_syf
fem_sorted = fem[fem[:, 1].argsort()]
fem_sorted = fem_sorted[fem_sorted[:, 1] != 1]
fem_grouped = groupby(fem_sorted, key=lambda x: x[1])
fem_modified_sherwood = {k: np.array(list(g)) for k, g in fem_grouped}

py_path = data_dir / "data" / "py_pe_vs_sh.csv"
py = np.loadtxt(py_path, delimiter=",", skiprows=1)
py = py[py[:, 0] < 10**6]
py = py[py[:, 0] > 10**3]
# we have to split into listst wiht different r_syf
py_sorted = py[py[:, 1].argsort()]
py_sorted = py_sorted[py_sorted[:, 1] != 1]
py_grouped = groupby(py_sorted, key=lambda x: x[1])
py_modified_sherwood = {k: np.array(list(g)) for k, g in py_grouped}


del py_modified_sherwood[0.8]
del fem_modified_sherwood[0.8]

comparison = {
    key: np.array(
        [
            [
                fem_array[0],
                fem_array[1],
                (py_array[2] - fem_array[2]) / py_array[2],
            ]
            for fem_array, py_array in zip(value, py_modified_sherwood[key])
        ]
    )
    for key, value in fem_modified_sherwood.items()
}


def clift_approximation(pe):
    return (1 / 2) * (1 + (1 + 2 * pe) ** (1 / 3))


peclet_values = np.logspace(-1, 12, 300)
analytic_clift = clift_approximation(peclet_values)


fontsize = 15 * 15 / 14 * (23.8 / 21)  * (11.34 / 12.23)
marker_size = 10

plt.rcParams.update({"text.usetex": True, "font.family": "Times"})
plt.figure(figsize=(10, 6))

empty = plt.scatter(
    [0],
    [0],
    label=" ",
    color="none",
    facecolors="none",
    s=marker_size,
)

femdots = [
    plt.scatter(
        data[:, 0],
        data[:, 2],
        label=f"${1-data[1,1]:5.3f}$",
        color=tableau[i],
        zorder=1,
        s=marker_size,
    )
    for i, data in enumerate(comparison.values())
]

femdots = [empty] + femdots

# femlegend
legend1 = plt.legend(
    handles=femdots,
    fontsize=fontsize,
    frameon=False,
    labelspacing=0.1,
    handlelength=0.1,
    loc=(0.01, 0.05),
    ncols=2,
)

plt.text(
    0.06,
    0.27,
    r"$\beta = $",
    ha="center",
    fontsize=fontsize,
    transform=plt.gca().transAxes,
)

# plt.add_artist(legend1)

"""legends"""

plt.xscale("log")
plt.xlim(10**3, 10**6)
plt.ylim(-0.05, 0.09)
plt.tick_params(axis="both", labelsize=fontsize)
plt.ylabel(r"$(\textrm{Sh}_{\texttt{pychastic}}-\textrm{Sh}_{\texttt{scikit-fem}})/\textrm{Sh}_{\texttt{pychastic}}$", fontsize=fontsize)
plt.xlabel(r"Peclet number $\textrm{Pe}$", fontsize=fontsize)


tosave = parent_dir / "graphics/pychastic_scikit_comparison.pdf"
plt.savefig(tosave, bbox_inches="tight", pad_inches=0.02)

# Show plot
plt.show()
