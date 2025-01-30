import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

parent_dir = Path(__file__).parent

data_histogram = parent_dir / "data" / "wiliams_2022.csv"

data = np.loadtxt(data_histogram, delimiter=",", skiprows=1)

fontsize = 15
plt.rcParams.update({"text.usetex": True, "font.family": "Times", "savefig.dpi": 300})
fig, ax = plt.subplots(figsize=(8, 4))
bins = np.logspace(np.log10(min(data[:, 0])), np.log10(max(data[:, 0])), 20)
ax.hist(data[:, 0], bins=bins, color="#4e79a7", edgecolor="black", label="Histogram, data Williams et. al.")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"particle radius ($r$) [m]", fontsize=fontsize)
ax.set_ylabel(r"occurrence ($n$)", fontsize=fontsize)

x = np.logspace(np.log10(0.25e-5), np.log10(0.015), 100)
y = 3 * (10**2 * x) ** -3
ax.plot(x, y, color="black", label=r"Slope $r^{-3}$ from Jackson et al. (1997)")


ax.set_xlim(0.345e-5, 0.013)
ax.tick_params(which="both", labelsize=fontsize)
ax.legend(fontsize=fontsize, frameon=False)

tosave = parent_dir / "graphics/histogram.pdf"
plt.savefig(
    tosave,
    bbox_inches="tight",
    pad_inches=0.02,
)
plt.show()
