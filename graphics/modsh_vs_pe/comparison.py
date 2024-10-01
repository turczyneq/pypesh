import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import groupby

parent_dir = Path(__file__).parent
fem_path = parent_dir / "mod_fem_pe_vs_sh.csv"
fem = np.loadtxt(fem_path, delimiter=",", skiprows=1)
# we have to split into listst wiht different r_syf
fem_sorted = fem[fem[:, 1].argsort()]
fem_grouped = groupby(fem_sorted, key=lambda x: x[1])
fem_plt = {k: np.array(list(g)) for k, g in fem_grouped}

py_path = parent_dir / "mod_py_pe_vs_sh.csv"
py = np.loadtxt(py_path, delimiter=",", skiprows=1)
# we have to split into listst wiht different r_syf
py_sorted = py[py[:, 1].argsort()]
py_grouped = groupby(py_sorted, key=lambda x: x[1])
py_plt = {k: np.array(list(g)) for k, g in py_grouped}


def clift_approximation(pe):
    # mod_sh = sh
    return (1 / 2) * (1 + (1 + 2 * pe) ** (1 / 3))


peclet_values = np.logspace(-1, 13, 300)
analytic_clift = clift_approximation(peclet_values)


# Plot all data

plt.figure(figsize=(12, 7))
plt.rcParams.update({"text.usetex": True, "font.family": "Cambria"})

# Plot Clift data
plt.loglog(
    peclet_values,
    analytic_clift,
    label="Clift et al.",
    color="k",
    linestyle="-",
    linewidth=2,
    zorder = 0
)

num = 0
for data in fem_plt.values():
    # Plot our data
    plt.scatter(
        data[:, 0],
        data[:, 2],
        label=f"$r_{{particle}} = {round(10000*(1-data[1,1]))/10000}$",
        color=f"C{num}"
    )
    num +=1

num = 0
for data in py_plt.values():
    # Plot our data
    plt.scatter(
        data[:, 0],
        data[:, 2],
        color=f"C{num}",
        facecolors='none'
    )
    num +=1

# for limits, modified sherwood should be 1
plt.loglog(
    peclet_values,
    [1 for x in peclet_values],
    color="k",
    linestyle="--",
    linewidth=2,
    zorder = 0
)


#add dummy plt to make legend

plt.scatter(
    [0],
    [0],
    label="FEM",
    color="k"
)

plt.scatter(
    [0],
    [0],
    label=r"pychastic",
    color="k",
    facecolors='none'
)

# Logarithmic scale
plt.xscale("log")
plt.yscale("log")
plt.xlim(0.5, 5*10**12)
plt.ylim(0.95, 90)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# Labels and Title
plt.xlabel(r"Peclet Number $\left(\mathrm{Pe}\right)$", fontsize=20)
plt.ylabel(r"Modified Sherwood Number $\left(\widetilde{\mathrm{Sh}}\right)$", fontsize=20)

# Legend
plt.legend(fontsize=20, frameon=False, loc = 1)
plt.tight_layout()
tosave = parent_dir.parent / "ignore/modsh_vs_pe.pdf"
plt.savefig(tosave)

# Show plot
plt.show()