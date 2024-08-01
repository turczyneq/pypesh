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


# Plot all data

plt.figure(figsize=(8, 5))
plt.rcParams.update({"text.usetex": True, "font.family": "Cambria"})

# Plot our data
plt.loglog(
    numerical[:, 0],
    numerical[:, 1],
    label="This work",
    color="C0",
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
plt.loglog(
    numerical_clift[:, 0],
    numerical_clift[:, 1],
    label="Clift et al. (numerical)",
    color="C7",
    dashes=[4, 0.5, 0.5, 0.5],
    linewidth=3,
    zorder=1,
)
plt.loglog(
    0.5 * friedlander[:, 0],
    0.5 * friedlander[:, 1],
    label="Friedlander (numerical)",
    color="C7",
    dashes=[4, 0.5, 0.5, 0.5, 0.5, 0.5],
    linewidth=3,
    zorder=1,
)

# Plot Kutateladze experimental data
plt.scatter(
    0.5 * kutateladze[:, 0],
    kutateladze[:, 1],
    label="Kutateladze et al.",
    color="C1",
    marker="o",
    s=50,
    edgecolor="k",
    zorder=3,
)

# Plot Feng experimental data
plt.scatter(
    feng[:, 0],
    feng[:, 1],
    label="Feng et al.",
    color="C3",
    marker="D",
    s=50,
    edgecolor="k",
    zorder=4,
)

# Plot Kramers experimental data
plt.scatter(
    2*kramers[:, 0],
    kramers[:, 1],
    label="Kramers et al.",
    color="C6",
    marker="s",
    s=50,
    edgecolor="k",
    zorder=3,
)

# Logarithmic scale
plt.xscale("log")
plt.yscale("log")
plt.xlim(0.5, 10**5)
plt.ylim(0.8, 30)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

# Labels and Title
plt.xlabel(r"Peclet Number $\left(\mathrm{Pe}\right)$", fontsize=15)
plt.ylabel(r"Sherwood Number $\left(\mathrm{Sh}\right)$", fontsize=15)

# Legend
plt.legend(fontsize=15, frameon=False)


# Show plot
plt.show()
