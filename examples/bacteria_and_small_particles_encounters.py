import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from pypesh.mpl.curved_text import CurvedText
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


def make_length_scale(xargs, yargs, thickness, height, c, label, ax):
    ax.plot(
        xargs,
        yargs,
        c=c,
        lw=thickness,
    )
    ax.plot(
        [xargs[0], xargs[0]],
        [yargs[0] - height, yargs[0] + height],
        c=c,
        lw=thickness,
    )
    ax.plot(
        [xargs[1], xargs[1]],
        [yargs[0] - height, yargs[0] + height],
        c=c,
        lw=thickness,
    )

    axes[0].text(
        (xargs[0] + xargs[1]) / 2,
        yargs[0] + 0.035,
        label,
        c=c,
        va="top",
        ha="center",
        fontsize=fontsize,
    )
    return None


width = 0.3
space = 0.02

image_path = parent_dir / "images" / "pelagibacter.jpg"
pelagibacter = mpimg.imread(image_path)
x_min, x_max, y_min_from_top = 100, 400, 100
y_max_from_top = (
    int(round((x_max - x_min) * ((1 / 3 - 2 * space) / width))) + y_min_from_top
)
pelagibacter = pelagibacter[y_min_from_top:y_max_from_top, x_min:x_max]

image_path = parent_dir / "images" / "roseobacter.jpg"
roseobacter = mpimg.imread(image_path)
x_min, x_max, y_min_from_top = 0, 700, 100
y_max_from_top = (
    int(round((x_max - x_min) * ((1 / 3 - 2 * space) / width)) * (989 / 697))
    + y_min_from_top
)
roseobacter = roseobacter[y_min_from_top:y_max_from_top, x_min:x_max]

image_path = parent_dir / "images" / "synechococcus.png"
synechococcus = mpimg.imread(image_path)
x_min, x_max, y_min_from_top = 50, 450, 20
y_max_from_top = (
    int(round((x_max - x_min) * ((1 / 3 - 2 * space) / width))) + y_min_from_top
)
synechococcus = synechococcus[y_min_from_top:y_max_from_top, x_min:x_max]

fontsize = 15 * 1.3 * 11.5 / 16 * 27.5 / 33.5
marker_size = 80
plt.rcParams.update({"text.usetex": True, "font.family": "Times", "savefig.dpi": 700})

fig, axes = plt.subplots(
    1,
    2,
    figsize=(19.5 * 0.6, 10.5 * 0.6),
    sharey=True,
    gridspec_kw={"width_ratios": [width, 1]},
)

axes[0].set_xlim(0, width)
axes[0].set_ylim(0, 1)

axes[0].imshow(
    synechococcus,
    extent=(0, width, space, 1 / 3 - space),
    origin="upper",
)
axes[0].imshow(
    pelagibacter,
    aspect="equal",
    extent=(0, width, 1 / 3 + space, 2 / 3 - space),
    origin="upper",
)
axes[0].imshow(
    roseobacter,
    extent=(0, width, 2 / 3 + space, 1 - space),
    origin="upper",
)
axes[0].axis("off")


dist = 0.01
axes[0].text(
    dist,
    1 / 3 - space - dist,
    r"\textit{Synechococcus}",
    c="k",
    va="top",
    ha="left",
    fontsize=fontsize,
)

axes[0].text(
    dist,
    2 / 3 - space - dist,
    r"\textit{Pelagibacter}",
    c="w",
    va="top",
    ha="left",
    fontsize=fontsize,
)

axes[0].text(
    dist,
    1 - space - dist,
    r"\textit{Roseobacter}",
    c="w",
    va="top",
    ha="left",
    fontsize=fontsize,
)
make_length_scale(
    [0.075, 0.18], [0.88, 0.88], 0.5, 0.002, "w", r"$0.6$ $\mu$m", axes[0]
)
make_length_scale(
    [0.0173, 0.229 - 0.0573 - 0.04], [0.4, 0.4], 0.5, 0.002, "w", r"$2$ $\mu$m", axes[0]
)
make_length_scale([0.18, 0.248], [0.1, 0.1], 0.5, 0.002, "k", r"$5$ $\mu$m", axes[0])


axes[1].plot(
    r_bacteria_list,
    lower_band,
    c="k",
)

axes[1].plot(
    r_bacteria_list,
    upper_band,
    c="k",
)

axes[1].fill_between(
    r_bacteria_list,
    lower_band,
    np.zeros_like(r_bacteria_list),
    alpha=0.3,
    color="#a0cbe8",
)

axes[1].fill_between(
    r_bacteria_list,
    upper_band,
    np.ones_like(r_bacteria_list),
    alpha=0.3,
    color="#ffbe7d",
)


axes[1].text(
    0.25,
    0.1,
    r"diffusion share",
    ha="center",
    va="top",
    fontsize=fontsize,
    transform=axes[1].transAxes,
    color=tableau[0],
    weight="bold",
)

axes[1].text(
    0.8,
    0.8,
    r"direct interceptions share",
    ha="center",
    va="top",
    fontsize=fontsize,
    transform=axes[1].transAxes,
    color=tableau[1],
    weight="bold",
)

### make curved text
curve = (np.array(upper_band) + np.array(lower_band)) * (1 / 2)

# axes[1]plot(r_bacteria_list, y_values)

added_text = CurvedText(
    x=r_bacteria_list[100:],
    y=curve[100:],
    text=r"changes with bigger particle $a$ and $U$",
    va="bottom",
    axes=axes[1],
    fontsize=fontsize,
)

for val in [0.3, 0.5, 2.8]:
    axes[1].axvline(x=val, c="gray", ls="--")

axes[1].text(
    2.8,
    0.7,
    r"\textit{Synechococcus}",
    va="top",
    ha="right",
    fontsize=fontsize,
    rotation=90,
)

axes[1].text(
    0.5,
    0.8,
    r"\textit{Pelagibacter}",
    va="top",
    ha="right",
    fontsize=fontsize,
    rotation=90,
)

axes[1].text(
    0.3,
    0.9,
    r"\textit{Roseobacter}",
    va="top",
    ha="right",
    fontsize=fontsize,
    rotation=90,
)


axes[1].set_xlim(0, r_bacteria_list[-1])
axes[1].set_ylim(0, 1.001)

axes[1].set_xlabel(r"Size of small particles ($b$) [$\mu$m]", fontsize=fontsize)
axes[1].set_ylabel(r"Partial contribution", fontsize=fontsize)

axes[1].tick_params(which="both", labelsize=fontsize, left=True, labelleft=True)

plt.subplots_adjust(wspace=0.08)
tosave = parent_dir / "graphics/encounter_type_with_small_particles.pdf"
plt.savefig(
    tosave,
    bbox_inches="tight",
    pad_inches=0.02,
)

# plt.show()
