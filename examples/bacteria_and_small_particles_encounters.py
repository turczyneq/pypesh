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
a_list = np.logspace(np.log10(0.00002), -2, 10)


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


line_1 = np.array(for_line(20 * 10 ** (-6), drho=20))
line_2 = np.array(for_line(10 ** (4) * 10 ** (-6), drho=300))
line_3 = np.array(for_line(0.3 * 10 ** (-3), drho=70))
upper_band = [to_upper_band(10 ** (-6) * r_bacteria) for r_bacteria in r_bacteria_list]
lower_band = [to_lower_band(10 ** (-6) * r_bacteria) for r_bacteria in r_bacteria_list]


def make_length_scale(xargs, yargs, thickness, height, c, label, axes):
    axes.plot(
        xargs,
        yargs,
        c=c,
        lw=thickness,
    )
    axes.plot(
        [xargs[0], xargs[0]],
        [yargs[0] - height, yargs[0] + height],
        c=c,
        lw=2 * thickness,
    )
    axes.plot(
        [xargs[1], xargs[1]],
        [yargs[0] - height, yargs[0] + height],
        c=c,
        lw=2 * thickness,
    )

    axes.text(
        xargs[0],
        yargs[0] + 0.11,
        label,
        c=c,
        va="top",
        ha="left",
        fontsize=fontsize,
    )
    return None


images_aspect = 1

image_path = parent_dir / "images" / "pelagibacter.jpg"
pelagibacter = mpimg.imread(image_path)
x_min, x_max, y_min_from_top = 100, 400, 100
y_max_from_top = int(round((x_max - x_min) * images_aspect) + y_min_from_top)
pelagibacter = pelagibacter[y_min_from_top:y_max_from_top, x_min:x_max]


image_path = parent_dir / "images" / "emiliania_huxleyi.png"
huxleyi = mpimg.imread(image_path)
x_min, x_max, y_min_from_top = 0, 650, 0
y_max_from_top = (
    int(round((x_max - x_min) * images_aspect) + y_min_from_top) + y_min_from_top
)
huxleyi = huxleyi[y_min_from_top:y_max_from_top, x_min:x_max]


image_path = parent_dir / "images" / "thalassiosira_pseudonana.png"
thalassiosira = mpimg.imread(image_path)
x_min, x_max, y_min_from_top = 100, 800, 0
y_max_from_top = (
    int(round((x_max - x_min) * images_aspect) + y_min_from_top) + y_min_from_top
)
thalassiosira = thalassiosira[y_min_from_top:y_max_from_top, x_min:x_max]


image_path = parent_dir / "images" / "prochloroccus.png"
prochloroccus = mpimg.imread(image_path)
x_min, x_max, y_min_from_top = 0, 600, 50
y_max_from_top = (
    int(round((x_max - x_min) * images_aspect) + y_min_from_top) + y_min_from_top
)
prochloroccus = prochloroccus[y_min_from_top:y_max_from_top, x_min:x_max]

fontsize = 11.5 * (11.3 / 9)
marker_size = 80
plt.rcParams.update({"text.usetex": True, "font.family": "Times", "savefig.dpi": 700})

image_colum_width = 0.25  # multiples of plot width
figure_aspect_ratio = 300 / 800

fig = plt.figure(
    constrained_layout=True,
    figsize=(11.7, 11.7 * figure_aspect_ratio),
)
gs = fig.add_gridspec(
    2,
    3,
    wspace=0.02,
    hspace=0.02,
    width_ratios=[image_colum_width, image_colum_width, 1],
)

axes_nw = fig.add_subplot(gs[0, 0])
axes_ne = fig.add_subplot(gs[0, 1])
axes_sw = fig.add_subplot(gs[1, 0])
axes_se = fig.add_subplot(gs[1, 1])
axes_big = fig.add_subplot(gs[:, 2])

for image_ax in [axes_nw, axes_ne, axes_sw, axes_se]:
    image_ax.set_xlim(0, 1)
    image_ax.set_ylim(0, 1)
    image_ax.set_axis_off()

for image_ax, image in zip(
    [axes_nw, axes_ne, axes_sw, axes_se],
    [thalassiosira, huxleyi, pelagibacter, prochloroccus],
):
    image_ax.imshow(image, aspect="equal", extent=(0, 1, 0, 1))

for image_ax, text in zip(
    [axes_nw, axes_ne, axes_sw, axes_se],
    [
        r"\textit{Thalassiosira}",
        r"\textit{E. Huxleyi}",
        r"\textit{Pelagibacter}",
        r"\textit{Prochlorococcus}",
    ],
):
    image_ax.text(
        0.05,
        0.95,
        text,
        c="w",
        va="top",
        ha="left",
        fontsize=fontsize,
        # transform=axes_big.transAxes,
        bbox=dict(boxstyle="square,pad=0.15", fc="0.2", ec="none"),
    )

for image_ax, text in zip(
    [axes_nw, axes_ne, axes_sw, axes_se],
    [
        r"$2$ $\mu$m",
        r"$5$ $\mu$m",
        r"$2$ $\mu$m",
        r"$1.5$ $\mu$m",
    ],
):
    make_length_scale([0.04, 0.34], [0.03, 0.03], 0.5, 0.005, "w", text, image_ax)

#
# Complicated graph
#

axes_big.plot(
    r_bacteria_list,
    lower_band,
    c="k",
    ls="--",
    label=r"$a=10^4$ $\mu$m" + "\n" + r"$U \approx 2\times10^6$ m/day",
)

axes_big.plot(
    r_bacteria_list,
    upper_band,
    c="k",
    label=r"$a=20$ $\mu$m" + "\n" + r"$U \approx 1.4$ m/day",
)

axes_big.fill_between(
    r_bacteria_list,
    lower_band,
    np.zeros_like(r_bacteria_list),
    color="#aec7e8",
)

axes_big.fill_between(
    r_bacteria_list,
    upper_band,
    np.ones_like(r_bacteria_list),
    color="#ffbb78",
)

# axes_big.plot(
#     r_bacteria_list,
#     line_1,
#     c="#98df8a",
#     ls="--",
#     label=r"$a=50$ $\mu$m" + "\n" + r"$\Delta \rho=30$ kg $\textrm{m}^{-3}$",
# )

# axes_big.plot(
#     r_bacteria_list,
#     line_2,
#     c="#ff9896",
#     ls="--",
#     label=r"$a=10^3$ $\mu$m" + "\n" + r"$\Delta \rho=120$ kg $\textrm{m}^{-3}$",
# )


axes_big.text(
    0.1,
    0.2,
    r"diffusion share",
    ha="center",
    va="top",
    fontsize=fontsize,
    transform=axes_big.transAxes,
    color=tableau[0],
    weight="bold",
)

axes_big.text(
    0.82,
    0.8,
    r"direct interception share",
    ha="center",
    va="top",
    fontsize=fontsize,
    transform=axes_big.transAxes,
    color=tableau[1],
    weight="bold",
)


added_text = CurvedText(
    x=r_bacteria_list[20:],
    y=line_3[20:],
    text=r"changes with larger particle $a$ and $U$",
    ha="center",
    va="center",
    axes=axes_big,
    fontsize=fontsize,
)

# added_text = CurvedText(
#     x=r_bacteria_list[40:],
#     y=line_1[40:],
#     text=r"small and slow",
#     ha="center",
#     va="center",
#     axes=axes_big,
#     fontsize=fontsize,
# )


# added_text = CurvedText(
#     x=r_bacteria_list[30:],
#     y=line_2[30:],
#     text=r"big and fast",
#     ha="center",
#     va="center",
#     axes=axes_big,
#     fontsize=fontsize,
# )


for val in [0.3, 0.4, 2.5, 3.5]:
    axes_big.plot([val, val], [0, 1], c="gray", ls="--")

axes_big.text(
    3.5,
    0.6,
    r"\textit{Thalassiosira}",
    va="bottom",
    ha="right",
    fontsize=fontsize,
    rotation=90,
)

axes_big.text(
    2.5,
    0.5,
    r"\textit{E. Huxleyi}",
    va="bottom",
    ha="right",
    fontsize=fontsize,
    rotation=90,
)
axes_big.text(
    0.4,
    0.4,
    r"\textit{Pelagibacter}",
    va="bottom",
    ha="right",
    fontsize=fontsize,
    rotation=90,
)

axes_big.text(
    0.3,
    0.3,
    r"\textit{Prochlorococcus}",
    va="bottom",
    ha="right",
    fontsize=fontsize,
    rotation=90,
)


axes_big.set_xlim(1e-1, r_bacteria_list[-1])
axes_big.set_xscale("log")

axes_big.set_ylim(-0.06, 1.001)

axes_big.set_xlabel(r"Size of small particles $b$ [$\mu$m]", fontsize=fontsize)
axes_big.set_ylabel(r"Partial contribution", fontsize=fontsize)

axes_big.plot([2, 2], [-0.06, 0], c="k", ls="--", lw=0.5)

axes_big.text(
    0.5,
    -0.03,
    r"picoplankton",
    ha="center",
    va="center",
    fontsize=fontsize,
    # transform=axes_big.transAxes,
    # color=tableau[0],
    weight="bold",
)

axes_big.text(
    7,
    -0.03,
    r"nanoplankton",
    ha="center",
    va="center",
    fontsize=fontsize,
    # transform=axes_big.transAxes,
    # color=tableau[1],
    weight="bold",
)

plt.legend(
    fontsize=fontsize,
    frameon=True,
    facecolor="white",
    framealpha=0.3,
    edgecolor="none",
    loc=(0.67, 0.2),
)

axes_big.tick_params(which="both", labelsize=fontsize, left=True, labelleft=True)

tosave = parent_dir / "graphics/encounter_type_with_small_particles.pdf"
plt.savefig(
    tosave,
    bbox_inches="tight",
    pad_inches=0.02,
)

plt.show()
