import pypesh.fem as fem
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

peclet = 5 * 10**3
ball_radius = 0.05

# # # Cheap in time version of calculations, below expensive version

msh_list = np.logspace(-0.1, -2, 10)
msh_far_list = np.logspace(1, -2, 10)
scale_list = np.linspace(1 / 8, 1, 10)
length_list = np.linspace(2, 10, 10)


sh_msh = [
    fem._sherwood_fem_custom_mesh(peclet, ball_radius, mesh_size=msh)
    for msh in msh_list
]

sh_msh_far = [
    fem._sherwood_fem_custom_mesh(peclet, ball_radius, mesh_size=msh)
    for msh in msh_list
]

sh_scale = [
    fem._sherwood_fem_custom_mesh(peclet, ball_radius, cell_size=sc)
    for sc in scale_list
]

sh_width = [
    fem._sherwood_fem_custom_mesh(peclet, ball_radius, width=len) for len in length_list
]
sh_floor = [
    fem._sherwood_fem_custom_mesh(peclet, ball_radius, floor=len) for len in length_list
]
sh_ceiling = [
    fem._sherwood_fem_custom_mesh(peclet, ball_radius, ceiling=len)
    for len in length_list
]

fontsize = 10
plt.rcParams.update({"text.usetex": True, "font.family": "Times", "savefig.dpi": 300})
fig, axes = plt.subplots(
    2,
    3,
    figsize=(22 * 0.6, 10 * 0.6),
    # sharey=True
    # sharex=True,
    gridspec_kw={
        "width_ratios": [1, 1, 1],
        "height_ratios": [1, 1],
        "wspace": 0.25,
        "hspace": 0.25,
    },
)
stuff_to_plt = [
    (1 / msh_list, sh_msh),
    (1 / msh_far_list, sh_msh_far),
    (scale_list, sh_scale),
    (length_list, sh_width),
    (length_list, sh_floor),
    (length_list, sh_ceiling),
]
for i, ax in enumerate(axes.flatten()):
    ax.scatter(stuff_to_plt[i][0], stuff_to_plt[i][1])
    ax.axhline(sh_msh[-1], linestyle='--', color='k', linewidth=1)

for axlist in axes:
    for ax in axlist:
        ax.set_ylabel(
            r"Sherwood number $\left(\textrm{Sh}\times10^{-3}\right)$",
            fontsize=fontsize,
        )
        ax.tick_params(which="both", labelsize=fontsize)
        ax.set_xmargin(0.05)
        ax.set_ymargin(0.05)

xlabel_list = [
    r"mesh tightness (1/\texttt{mesh})",
    r"far mesh tightness (1/\texttt{mesh_far})",
    r"size of cell (\texttt{cell_size})",
    r"width of cell (\texttt{width})",
    r"distance to floor (\texttt{floor})",
    r"distance to ceiling (\texttt{ceiling})",
]
for i, ax in enumerate(axes.flatten()):
    ax.set_xlabel(xlabel_list[i], fontsize=fontsize)

axes[0, 0].set_xscale("log")
axes[0, 1].set_xscale("log")

axes[0, 0].set_yscale("log")
axes[0, 1].set_yscale("log")

parent_dir = Path(__file__).parent
tosave = parent_dir / "graphics/different_mesh.pdf"
plt.savefig(
    tosave,
    bbox_inches="tight",
    pad_inches=0.02,
)

plt.show()
