import pypesh.fem as fem
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

parent_dir = Path(__file__).parent

peclet = 10**4
ball_radius = 0.9
names = [
    "msh",
    "far_msh",
    "scale",
    "width",
    "floor",
    "ceiling",
]


# # # Calculation of points used in plot

# # # Cheap in time version of calculations, below expensive version

msh_list = np.logspace(-1, -2, 2)
msh_far_list = np.logspace(0.8, -1, 2)
scale_list = np.linspace(1 / 6, 1, 2)
length_list = np.linspace(2, 10, 2)


# # # long calculation

# msh_list = np.logspace(-1, -3, 20)
# msh_far_list = np.logspace(0.8, -1.5, 20)
# scale_list = np.linspace(1 / 6, 2, 20)
# length_list = np.linspace(2, 15, 20)


def export_data(msh_list, msh_far_list, scale_list, length_list, scale):
    sh_msh = np.array(
        [
            fem._sherwood_fem_custom_mesh(peclet, ball_radius, mesh_size=msh)
            for msh in msh_list
        ]
    )

    sh_msh_far = np.array(
        [
            fem._sherwood_fem_custom_mesh(peclet, ball_radius, far_mesh=msh)
            for msh in msh_far_list
        ]
    )

    sh_scale = np.array(
        [
            fem._sherwood_fem_custom_mesh(peclet, ball_radius, cell_size=sc)
            for sc in scale_list
        ]
    )

    sh_width = np.array(
        [
            fem._sherwood_fem_custom_mesh(peclet, ball_radius, width=len)
            for len in length_list
        ]
    )
    sh_floor = np.array(
        [
            fem._sherwood_fem_custom_mesh(peclet, ball_radius, floor=len)
            for len in length_list
        ]
    )
    sh_ceiling = np.array(
        [
            fem._sherwood_fem_custom_mesh(peclet, ball_radius, ceiling=len)
            for len in length_list
        ]
    )

    to_save = {
        "msh": np.array([1 / msh_list, scale * sh_msh]),
        "far_msh": np.array([1 / msh_far_list, scale * sh_msh_far]),
        "scale": np.array([scale_list, scale * sh_scale]),
        "width": np.array([length_list, scale * sh_width]),
        "floor": np.array([length_list, scale * sh_floor]),
        "ceiling": np.array([length_list, scale * sh_ceiling]),
    }

    for key, data in to_save.items():
        file_path = parent_dir / "data" / f"{key}.csv"
        np.savetxt(
            file_path,
            data.T,
            delimiter=",",
            # header="xargs,yargs",
            comments="",
        )
    return None


# export_data(msh_list, msh_far_list, scale_list, length_list, 1)

predicted_value = np.loadtxt(parent_dir / "data" / f"msh.csv", delimiter=",")[-1, 1]

fontsize = 10 * (9.5 / 7) * (12.8 / 8.7)
plt.rcParams.update({"text.usetex": True, "font.family": "Times", "savefig.dpi": 300})
fig, axes = plt.subplots(
    2,
    3,
    figsize=(22 * 0.6, 10 * 0.6),
    sharey=True,
    # sharex=True,
    gridspec_kw={
        "width_ratios": [1, 1, 1],
        "height_ratios": [1, 1],
        "wspace": 0,
        "hspace": 0.4,
    },
)

stuff_to_plt = [
    np.loadtxt(parent_dir / "data" / f"{name}.csv", delimiter=",") for name in names
]

vlines = [1 / 0.01, 1 / 0.5, 1.0, 10, 10, 10]
for i, ax in enumerate(axes.flatten()):
    hlength = stuff_to_plt[i][-1, 0] - stuff_to_plt[i][0, 0]
    ax.scatter(stuff_to_plt[i][:, 0], stuff_to_plt[i][:, 1])
    ax.axhline(stuff_to_plt[0][-1, 1], linestyle="--", color="gray", linewidth=0.8)
    ax.axvline(vlines[i], linestyle="--", color="k", linewidth=1)
    if i != 0 and i != 1:
        ax.text(
            vlines[i] - 0.09 * hlength,
            80,
            "default",
            ha="left",
            fontsize=fontsize,
            rotation=90,
        )
    else:
        ax.text(
            10
            ** (
                np.log10(vlines[i] / (hlength))
                - 0.09
                * (np.log10(stuff_to_plt[i][-1, 0]) - np.log10(stuff_to_plt[i][0, 0])) * 0.95
            )
            * hlength,
            80,
            "default",
            ha="left",
            fontsize=fontsize,
            rotation=90,
        )


for axlist in axes:
    for i, ax in enumerate(axlist):
        ax.set_xmargin(0.05)
        ax.set_ymargin(0.05)
        if i != 0:
            ax.tick_params(which="both", labelsize=fontsize, left=False)
        else:
            ax.tick_params(which="both", labelsize=fontsize)

axes[0, 0].set_ylabel(
    r"Sh",
    fontsize=fontsize,
)
axes[1, 0].set_ylabel(
    r"Sh",
    fontsize=fontsize,
)

xlabel_list = [
    r"mesh tightness (1/\texttt{mesh})",
    r"far mesh tightness (1/\texttt{mesh_far})",
    r"size of cell (\texttt{cell_size})",
    r"width of cell (\texttt{width})",
    r"distance to floor (\texttt{floor})",
    r"distance to ceiling (\texttt{ceiling})",
]
alphabet = [r"(a)", r"(b)", r"(c)", r"(d)", r"(e)", r"(f)"]
for i, ax in enumerate(axes.flatten()):
    ax.set_xlabel(xlabel_list[i], fontsize=fontsize)
    ax.text(
        0.89,
        0.85,
        alphabet[i],
        transform=ax.transAxes,
        fontsize=fontsize,
    )


axes[0, 0].set_xscale("log")
axes[0, 1].set_xscale("log")

# axes[0, 0].set_yscale("log")
# axes[0, 1].set_yscale("log")


tosave = parent_dir / "graphics/different_mesh.pdf"
plt.savefig(
    tosave,
    bbox_inches="tight",
    pad_inches=0.02,
)

plt.show()
