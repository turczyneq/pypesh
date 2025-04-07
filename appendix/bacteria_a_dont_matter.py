import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from pypesh.mpl.curved_text import CurvedText
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
import matplotlib.image as mpimg

tableau = list(mcolors.TABLEAU_COLORS)

parent_dir = Path(__file__).parent


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
    temperature = 277
    viscosity = 1.6 * 10 ** (-3)

    return k_b * temperature / (6 * np.pi * viscosity * r)


def stokes(r, drho):
    viscosity = 1.6 * 10 ** (-3)
    g = np.log(45) + 6
    return (2 / 9) * (drho * g / viscosity) * (r) ** 2


rho_list = np.linspace(30, 200, 10)
a_list = np.logspace(np.log10(0.00002), -3, 10)


def to_lower_band(r_bact):
    to_min = []
    for drho in rho_list:
        to_min = to_min + [
            relative_clift_flux(
                give_peclet(r_snow, stokes(r_snow, drho), r_bact, 0),
                r_bact / (r_snow + r_bact),
            )
            for r_snow in a_list
        ]

    return np.min(to_min)


def to_upper_band(r_bact):
    to_max = []
    for drho in rho_list:
        to_max = to_max + [
            relative_clift_flux(
                give_peclet(r_snow, stokes(r_snow, drho), r_bact, 0),
                r_bact / (r_snow + r_bact),
            )
            for r_snow in a_list
        ]
    return np.max(to_max)


r_bacteria_list = np.linspace(10 ** (-5), 25, 1000)


def for_line(r_snow, drho):
    return [
        relative_clift_flux(
            give_peclet(r_snow, stokes(r_snow, drho), 10 ** (-6) * r_bact, 0),
            10 ** (-6) * r_bact / (r_snow + 10 ** (-6) * r_bact),
        )
        for r_bact in r_bacteria_list
    ]


# line_1 = np.array(for_line(20 * 10 ** (-6), drho=20))
# line_2 = np.array(for_line(10 ** (4) * 10 ** (-6), drho=300))
line_3 = np.array(for_line(0.3 * 10 ** (-3), drho=70))
line_4 = np.array(for_line(10 ** (-3), drho=30))
line_5 = np.array(for_line(20 * 10 ** (-6), drho=200))
upper_band = [to_upper_band(10 ** (-6) * r_bacteria) for r_bacteria in r_bacteria_list]
lower_band = [to_lower_band(10 ** (-6) * r_bacteria) for r_bacteria in r_bacteria_list]

fontsize = 15 * 15 / 14 * (23.8 / 21)  * (11.34 / 12.23)
marker_size = 10

plt.rcParams.update({"text.usetex": True, "font.family": "Times"})
plt.figure(figsize=(10, 6))

#
# Complicated graph
#

plt.plot(
    r_bacteria_list,
    lower_band,
    c="k",
    ls="--",
    label=r"$a=10^3$ $\mu$m" + "\n" + r"$\Delta \rho = 200$ kg/$\textrm{m}^3$",
)

plt.plot(
    r_bacteria_list,
    line_5,
    ls='--',
    c=tableau[3],
    label=r"$a=20$ $\mu$m" + "\n" + r"$\Delta \rho = 200$ kg/$\textrm{m}^3$",
)

plt.plot(
    r_bacteria_list,
    line_4,
    c=tableau[2],
    label=r"$a=10^3$ $\mu$m" + "\n" + r"$\Delta \rho = 30$ kg/$\textrm{m}^3$",
)

plt.plot(
    r_bacteria_list,
    upper_band,
    c="k",
    label=r"$a=20$ $\mu$m" + "\n" + r"$\Delta \rho = 30$ kg/$\textrm{m}^3$",
)


plt.fill_between(
    r_bacteria_list,
    lower_band,
    np.zeros_like(r_bacteria_list),
    color="#aec7e8",
)

plt.fill_between(
    r_bacteria_list,
    upper_band,
    np.ones_like(r_bacteria_list),
    color="#ffbb78",
)

# plt.plot(
#     r_bacteria_list,
#     line_1,
#     c="#98df8a",
#     ls="--",
#     label=r"$a=50$ $\mu$m" + "\n" + r"$\Delta \rho=30$ kg $\textrm{m}^{-3}$",
# )

# plt.plot(
#     r_bacteria_list,
#     line_2,
#     c="#ff9896",
#     ls="--",
#     label=r"$a=10^3$ $\mu$m" + "\n" + r"$\Delta \rho=120$ kg $\textrm{m}^{-3}$",
# )


plt.text(
    0.3,
    0.2,
    "advection--\n--diffusion share",
    ha="left",
    va="top",
    fontsize=fontsize,
    transform=plt.gca().transAxes,
    color=tableau[0],
    weight="bold",
)

plt.text(
    0.82,
    0.92,
    r"direct interception share",
    ha="center",
    va="top",
    fontsize=fontsize,
    transform=plt.gca().transAxes,
    color=tableau[1],
    weight="bold",
)


added_text = CurvedText(
    x=r_bacteria_list[18:],
    y=line_3[18:],
    text=r"varying marine snow parameters $a$ and $U$",
    ha="center",
    va="center",
    axes=plt.gca(),
    fontsize=fontsize,
)

# added_text = CurvedText(
#     x=r_bacteria_list[40:],
#     y=line_1[40:],
#     text=r"small and slow",
#     ha="center",
#     va="center",
#     axes=plt,
#     fontsize=fontsize,
# )


# added_text = CurvedText(
#     x=r_bacteria_list[30:],
#     y=line_2[30:],
#     text=r"big and fast",
#     ha="center",
#     va="center",
#     axes=plt,
#     fontsize=fontsize,
# )


for val in [0.3, 0.4, 2.5, 3.5]:
    plt.plot([val, val], [0, 1], c="gray", ls="--")

plt.text(
    3.5,
    0.6,
    r"\textit{Thalassiosira}",
    va="bottom",
    ha="right",
    fontsize=fontsize,
    rotation=90,
)

plt.text(
    2.5,
    0.5,
    r"\textit{E. Huxleyi}",
    va="bottom",
    ha="right",
    fontsize=fontsize,
    rotation=90,
)
plt.text(
    0.4,
    0.4,
    r"\textit{Pelagibacter}",
    va="bottom",
    ha="right",
    fontsize=fontsize,
    rotation=90,
)

plt.text(
    0.3,
    0.3,
    r"\textit{Prochlorococcus}",
    va="bottom",
    ha="right",
    fontsize=fontsize,
    rotation=90,
)


plt.xlim(1e-1, r_bacteria_list[-1])
plt.xscale("log")

plt.ylim(-0.06, 1.001)

plt.xlabel(r"Size of suspended objects $b$ [$\mu$m]", fontsize=fontsize)
plt.ylabel(r"Partial contribution", fontsize=fontsize)

plt.plot([2, 2], [-0.06, 0], c="k", ls="--", lw=0.5)

plt.text(
    0.5,
    -0.03,
    r"picoplankton",
    ha="center",
    va="center",
    fontsize=fontsize,
    # transform=plt.transAxes,
    # color=tableau[0],
    weight="bold",
)

plt.text(
    7,
    -0.03,
    r"nanoplankton",
    ha="center",
    va="center",
    fontsize=fontsize,
    # transform=plt.transAxes,
    # color=tableau[1],
    weight="bold",
)

plt.legend(
    fontsize=fontsize,
    frameon=True,
    facecolor="white",
    framealpha=0.3,
    edgecolor="none",
    loc=(0.68, 0.2),
)

plt.tick_params(which="both", labelsize=fontsize, left=True, labelleft=True)

tosave = parent_dir / "graphics/encounter_type_a_dont_matters.pdf"
plt.savefig(
    tosave,
    bbox_inches="tight",
    pad_inches=0.02,
)

plt.show()
