import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from pypesh.mpl.curved_text import CurvedText
import matplotlib.colors as mcolors
tableau = list(mcolors.TABLEAU_COLORS)


def clift(pe):
    return (1 / 2) * (1 + (1 + 2 * pe) ** (1 / 3))


def advect(pe, beta):
    return (pe / 4) * ((beta**2) * (3 - beta) / 2)


def our_flux(pe, beta):
    return clift(pe) + advect(pe, beta)


def relative_advective_flux(pe, beta):
    return advect(pe, beta) / (clift(pe) + advect(pe, beta))


def relative_clift_flux(pe, beta):
    return clift(pe) / (clift(pe) + advect(pe, beta))


def give_peclet(r1, u1, r2, u2):
    return np.abs(u1 - u2) * (r1 + r2) / (diffusion(r1) + diffusion(r2))


def diffusion(r):
    k_b = 1.38 * 10 ** (-23)
    temperature = 280
    viscosity = 1.6 * 10 ** (-3)

    return k_b * temperature / (6 * np.pi * viscosity * r)


def stokes(r, drho):
    viscosity = 1.6 * 10 ** (-3)
    g = np.log(45) + 6
    return (2 / 9) * (drho * g / viscosity) * (r) ** 2


def to_lower_band(r_bact):
    to_min = []
    for drho in np.linspace(30, 200, 20):
        to_min = to_min + [
            relative_clift_flux(
                give_peclet(r_snow, stokes(r_snow, drho), r_bact, 0),
                r_bact / (r_snow + r_bact),
            )
            for r_snow in np.logspace(-5, -2, 20)
        ]

    return np.min(to_min)


def to_upper_band(r_bact):
    to_max = []
    for drho in np.linspace(30, 200, 20):
        to_max = to_max + [
            relative_clift_flux(
                give_peclet(r_snow, stokes(r_snow, drho), r_bact, 0),
                r_bact / (r_snow + r_bact),
            )
            for r_snow in np.logspace(-5, -2, 20)
        ]
    return np.max(to_max)


r_bacteria_list = np.linspace(10 ** (-5), 5, 400)


upper_band = [to_upper_band(10 ** (-6) * r_bacteria) for r_bacteria in r_bacteria_list]
lower_band = [to_lower_band(10 ** (-6) * r_bacteria) for r_bacteria in r_bacteria_list]

fontsize = 26 * 1.2
marker_size = 80
plt.figure(figsize=(16 * 0.85, 9 * 0.85))
plt.rcParams.update({"text.usetex": True, "font.family": "Times"})


plt.plot(
    r_bacteria_list,
    lower_band,
    c="k",
)

plt.plot(
    r_bacteria_list,
    upper_band,
    c="k",
)

plt.fill_between(
    r_bacteria_list,
    lower_band,
    np.zeros_like(r_bacteria_list),
    alpha=0.3,
    color="#a0cbe8",
)

plt.fill_between(
    r_bacteria_list,
    upper_band,
    np.ones_like(r_bacteria_list),
    alpha=0.3,
    color="#ffbe7d",
)


plt.text(
    0.15,
    0.1,
    r"diffusion share",
    ha="center",
    va="top",
    fontsize=fontsize,
    transform=plt.gca().transAxes,
    color=tableau[0],
    weight="bold",
)

plt.text(
    0.7,
    0.8,
    r"direct interceptions share",
    ha="center",
    va="top",
    fontsize=fontsize,
    transform=plt.gca().transAxes,
    color=tableau[1],
    weight="bold",
)

### make curved text
curve = (np.array(upper_band) + 2 * np.array(lower_band)) * (1 / 3)

# plt.plot(r_bacteria_list, y_values)

added_text = CurvedText(
    x=r_bacteria_list[100:],
    y=curve[100:],
    text=r"changes with bigger particle $a$ and $U$",
    va="bottom",
    axes=plt.gca(),
    fontsize=fontsize,
)


plt.xlim(0, r_bacteria_list[-1])
plt.ylim(0, 1.001)

plt.xlabel(r"Size of small particles ($b$) [$\mu$m]", fontsize=fontsize)
plt.ylabel(r"Partial contribution", fontsize=fontsize)

plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)

plt.tight_layout()
parent_dir = Path(__file__).parent
tosave = parent_dir / "graphics/encounter_type_with_small_particles.pdf"
plt.savefig(
    tosave,
    bbox_inches="tight",
    pad_inches=0.02,
)
plt.show()
