import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

parent_dir = Path(__file__).parent

pe_values = np.logspace(0, 12, 2000)
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

fontsize = 20 * (9 / 6.7) * (23.5 / 20.8)
take_each = 1
error_range = 20
plt.figure(figsize=(10, 10))
plt.rcParams.update({"text.usetex": True, "font.family": "Times", "savefig.dpi": 700})


x_start = 0.13
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


plt.scatter(
    to_plot[:, 0],
    to_plot[:, 1],
    # s=0.2,
    s=0.5,
    c="k",
)

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

plt.xlabel(r"Peclet number $\textrm{Pe}$", fontsize=fontsize)
plt.ylabel(r"Colliders' size ratio $\beta$", fontsize=fontsize)

plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)

tosave = parent_dir / "graphics/beta_pe_for_players.pdf"
plt.savefig(
    tosave,
    bbox_inches="tight",
    pad_inches=0.02,
)

plt.show()
