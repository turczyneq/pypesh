import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.colors as mcolors

tableau = list(mcolors.TABLEAU_COLORS)

parent_dir = Path(__file__).parent

pe_values = np.logspace(-1, 12, 2000)
rsyf_values = np.logspace(-3.5, np.log10(0.5), 2000)

pe_min, pe_max = pe_values[0], pe_values[-1]
rsyf_min, rsyf_max = rsyf_values[0], rsyf_values[-1]


def clift(pe):
    return (1 / 2) * (1 + (1 + 2 * pe) ** (1 / 3))


def advect(pe, beta):
    return (pe / 4) * ((beta**2) * (3 - beta) / 2)


def diffusion(r):
    k_b = 1.38 * 10 ** (-23)
    temperature = 277
    viscosity = 1.6 * 10 ** (-3)

    return k_b * temperature / (6 * np.pi * viscosity * r)


def peclet(r1, u1, r2, u2):
    return np.abs(u1 - u2) * (r1 + r2) / (diffusion(r1) + diffusion(r2))


iversen = 10 ** (-3) * np.array([1, 1.5])
chajwa = 10 ** (-3) * np.array([0.18, 0.9])
chase = 10 ** (-6) * np.array([4, 10])
bacterium = 10 ** (-6) * np.array([1, 0])
mucus = 10 ** (-6) * np.array([4, 0])


def pair(i_1, i_2):
    return np.array(
        [peclet(i_1[0], i_1[1], i_2[0], i_2[1]), i_2[0] / (i_1[0] + i_2[0])]
    )


ivchaj = pair(iversen, chajwa)
ivchase = pair(iversen, chase)
chajchase = pair(chajwa, chase)
chasechase = pair(chase, chase)

withbact = [pair(value, bacterium) for value in [iversen, chajwa, chase]]
withmucus = [pair(value, mucus) for value in [iversen, chajwa, chase]]

to_plot = np.array([ivchaj, ivchase, chajchase, chasechase] + withbact)


def make_trianlge(pos, radius, inner_radius):
    phi = np.pi - (np.pi / 6) - np.arccos(1 / (2 * inner_radius)) - np.pi / 2

    angles = [
        (-np.pi / 6 + phi, inner_radius),
        (-np.pi / 6, 1),
        (-np.pi / 6 - phi, inner_radius),
        (-5 * np.pi / 6 + phi, inner_radius),
        (-5 * np.pi / 6, 1),
        (-5 * np.pi / 6 - phi, inner_radius),
        (np.pi / 2 + phi, inner_radius),
        (np.pi / 2, 1),
        (np.pi / 2 - phi, inner_radius),
        (-np.pi / 6 + phi, inner_radius),
    ]

    verts = np.array(
        [
            [
                radius * r * np.cos(ang) + pos[0],
                radius * r * np.sin(ang) + pos[1] - radius / 5,
            ]
            for ang, r in angles
        ]
    )

    codes = [
        mpath.Path.MOVETO,
        mpath.Path.CURVE3,
        mpath.Path.CURVE3,
        mpath.Path.LINETO,
        mpath.Path.CURVE3,
        mpath.Path.CURVE3,
        mpath.Path.LINETO,
        mpath.Path.CURVE3,
        mpath.Path.CURVE3,
        mpath.Path.LINETO,
    ]

    return mpath.Path(verts, codes)


def make_rhombus(pos, radius, inner_radius):
    phi = np.pi - (np.pi / 6) - np.arccos(1 / (2 * inner_radius)) - np.pi / 2

    angles = [
        (-5 * np.pi / 6, 1),
        (-5 * np.pi / 6 - phi, inner_radius),
        (np.pi / 2 + phi, inner_radius),
        (np.pi / 2, 1),
        (np.pi / 2 - phi, inner_radius),
        (-np.pi / 6 + phi, inner_radius),
        (-np.pi / 6, 1),
    ]

    verts = np.array(
        [
            [
                radius * inner_radius * np.cos(-np.pi / 6 + phi + np.pi) + pos[0],
                radius * inner_radius * np.sin(-np.pi / 6 + phi + np.pi)
                - radius / 2
                + pos[1],
            ]
        ]
        + [
            [
                radius * r * np.cos(ang) + pos[0],
                radius * r * (np.sin(ang)) + radius / 2 + pos[1],
            ]
            for ang, r in angles
        ]
        + [
            [
                radius * r * np.cos(ang + np.pi) + pos[0],
                radius * r * (np.sin(ang + np.pi)) - radius / 2 + pos[1],
            ]
            for ang, r in angles[1:-1]
        ]
    )

    codes = [
        mpath.Path.MOVETO,
        mpath.Path.CURVE3,
        mpath.Path.CURVE3,
        mpath.Path.LINETO,
        mpath.Path.CURVE3,
        mpath.Path.CURVE3,
        mpath.Path.LINETO,
        mpath.Path.CURVE3,
        mpath.Path.CURVE3,
        mpath.Path.LINETO,
        mpath.Path.CURVE3,
        mpath.Path.CURVE3,
        mpath.Path.LINETO,
    ]

    return mpath.Path(verts, codes)


def to_plt(pos):
    xpos = np.log10(pos[0])
    ypos = np.log10(pos[1])
    x_min, x_max = np.log10(pe_min), np.log10(pe_max)
    y_min, y_max = np.log10(rsyf_min), np.log10(rsyf_max)

    return (xpos - x_min) / (x_max - x_min), (ypos - y_min) / (y_max - y_min)


fontsize = 20 * (9 / 6.7) * (23.5 / 20.8)
take_each = 1
error_range = 20
plt.figure(figsize=(10, 10))
plt.rcParams.update({"text.usetex": True, "font.family": "Times", "savefig.dpi": 700})


x_start = 0.10
start_position = 0.4
dy = 0.115

for i, text in enumerate(
    ["bacteria", "small flake", "flake with\nmucus", "large flake"]
):
    plt.text(
        x_start,
        start_position - i * dy,
        text,
        ha="left",
        va="center",
        fontsize=fontsize,
        transform=plt.gca().transAxes,
        color="k",
    )

circle_radius = 0.019
box_size = 0.025
triangle_size = 0.025
rhombus_size = 0.02
distance = 0.02

artists = [
    mpatches.Circle(
        (x_start - 0.05, start_position),
        circle_radius,
        ec="none",
        fc=tableau[0],
        transform=plt.gca().transAxes,
    ),
    mpatches.FancyBboxPatch(
        (x_start - 0.05 - box_size / 2, start_position - dy - box_size / 2),
        box_size,
        box_size,
        ec="none",
        fc=tableau[1],
        transform=plt.gca().transAxes,
        boxstyle=mpatches.BoxStyle("Round", pad=0.005),
    ),
    mpatches.PathPatch(
        make_trianlge((x_start - 0.05, start_position - 2 * dy), triangle_size, 0.9),
        ec="none",
        fc=tableau[2],
        transform=plt.gca().transAxes,
    ),
    mpatches.PathPatch(
        make_rhombus((x_start - 0.05, start_position - 3 * dy), rhombus_size, 0.9),
        ec="none",
        fc=tableau[3],
        transform=plt.gca().transAxes,
    ),
]

for artist in artists:
    plt.gca().add_artist(artist)

artists = [
    mpatches.Circle(
        to_plt(to_plot[-1]) - np.array([distance, 0.0]),
        circle_radius,
        ec="none",
        fc=tableau[0],
        transform=plt.gca().transAxes,
    ),
    mpatches.FancyBboxPatch(
        to_plt(to_plot[-1])
        - np.array([box_size / 2, box_size / 2])
        + np.array([distance, 0.0]),
        box_size,
        box_size,
        ec="none",
        fc=tableau[1],
        transform=plt.gca().transAxes,
        boxstyle=mpatches.BoxStyle("Round", pad=0.005),
    ),
    mpatches.Circle(
        to_plt(to_plot[-2]) - (8 / 9) * np.array([distance, 0.0]),
        circle_radius,
        ec="none",
        fc=tableau[0],
        transform=plt.gca().transAxes,
    ),
    mpatches.PathPatch(
        make_trianlge(
            to_plt(to_plot[-2]) + (8 / 9) * np.array([distance, 0.0]),
            triangle_size,
            0.9,
        ),
        ec="none",
        fc=tableau[2],
        transform=plt.gca().transAxes,
    ),
    mpatches.Circle(
        to_plt(to_plot[-3]) - (12/13) * np.array([distance, 0.0]),
        circle_radius,
        ec="none",
        fc=tableau[0],
        transform=plt.gca().transAxes,
    ),
    mpatches.PathPatch(
        make_rhombus(
            to_plt(to_plot[-3]) + (12/13) * np.array([distance, 0.0]), rhombus_size, 0.9
        ),
        ec="none",
        fc=tableau[3],
        transform=plt.gca().transAxes,
    ),
]


for artist in artists:
    plt.gca().add_artist(artist)

artists = [
    mpatches.FancyBboxPatch(
        to_plt(to_plot[2])
        - np.array([box_size / 2, box_size / 2])
        - np.array([distance, 0.0]),
        box_size,
        box_size,
        ec="none",
        fc=tableau[1],
        transform=plt.gca().transAxes,
        boxstyle=mpatches.BoxStyle("Round", pad=0.005),
    ),
    mpatches.PathPatch(
        make_trianlge(
            to_plt(to_plot[2]) + np.array([distance, 0.0]), triangle_size, 0.9
        ),
        ec="none",
        fc=tableau[2],
        transform=plt.gca().transAxes,
    ),
    mpatches.FancyBboxPatch(
        to_plt(to_plot[1])
        - np.array([box_size / 2, box_size / 2])
        - (10/11) * np.array([distance, 0.0]),
        box_size,
        box_size,
        ec="none",
        fc=tableau[1],
        transform=plt.gca().transAxes,
        boxstyle=mpatches.BoxStyle("Round", pad=0.005),
    ),
    mpatches.PathPatch(
        make_rhombus(to_plt(to_plot[1]) + (10/11) * np.array([distance, 0.0]), rhombus_size, 0.9),
        ec="none",
        fc=tableau[3],
        transform=plt.gca().transAxes,
    ),
    mpatches.PathPatch(
        make_trianlge(
            to_plt(to_plot[0]) - (4 / 5) * np.array([distance, 0.0]),
            triangle_size,
            0.9,
        ),
        ec="none",
        fc=tableau[2],
        transform=plt.gca().transAxes,
    ),
    mpatches.PathPatch(
        make_rhombus(
            to_plt(to_plot[0]) + (4 / 5) * np.array([distance, 0.0]),
            rhombus_size,
            0.9,
        ),
        ec="none",
        fc=tableau[3],
        transform=plt.gca().transAxes,
    ),
]

for artist in artists:
    plt.gca().add_artist(artist)

plt.text(
    0.04,
    0.96,
    r"(c)",
    ha="center",
    va="center",
    color="k",
    fontsize=fontsize,
    transform=plt.gca().transAxes,
)

plt.text(
    0.04,
    0.7,
    r"\emph{pure diffusion}",
    ha="center",
    va="center",
    color="k",
    rotation="vertical",
    fontsize=fontsize,
    transform=plt.gca().transAxes,
)

plt.text(
    0.78,
    0.95,
    r"\emph{direct interception}",
    ha="center",
    va="center",
    color="k",
    fontsize=fontsize,
    transform=plt.gca().transAxes,
)

plt.text(
    0.8,
    0.13,
    r"\emph{advection --}",
    ha="center",
    va="top",
    color="k",
    fontsize=fontsize,
    # rotation=45,
    transform=plt.gca().transAxes,
)
plt.text(
    0.84,
    0.048,
    r"\emph{ -- diffusion}",
    ha="center",
    va="center",
    color="k",
    fontsize=fontsize,
    # rotation=45,
    transform=plt.gca().transAxes,
)

# plt.scatter(
#     to_plot[:, 0],
#     to_plot[:, 1],
#     # s=0.2,
#     s=0.5,
#     c="k",
# )

box_x_lim = 4 * 10 ** (4)
box_y_lim = 10 ** (-2)

plt.plot(
    [box_x_lim, box_x_lim],
    [rsyf_min, box_y_lim],
    c="k",
)

plt.plot(
    [pe_min, box_x_lim],
    [box_y_lim, box_y_lim],
    c="k",
)


plt.xscale("log")
plt.yscale("log")

plt.xlim(pe_min, pe_max)
plt.ylim(rsyf_min, rsyf_max)

plt.xlabel(r"P\'{e}clet number $\textrm{Pe}$", fontsize=fontsize)
plt.ylabel(r"Colliders' size ratio $\beta$", fontsize=fontsize)

plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)


xtick_list = [10 ** (pow) for pow in [ 1, 3, 5, 7, 9, 11]]
plt.xticks(xtick_list)

tosave = parent_dir / "graphics/beta_pe_for_players.pdf"
plt.savefig(
    tosave,
    bbox_inches="tight",
    pad_inches=0,
)

plt.show()
