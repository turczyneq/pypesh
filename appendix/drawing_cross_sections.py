import pypesh.visualisation as visual
import matplotlib.pyplot as plt
import pypesh.stokes_flow as sf
from pathlib import Path
import numpy as np
import matplotlib.colors as mcolors
import jax

tableau = list(mcolors.TABLEAU_COLORS)

parent_dir = Path(__file__).parent

ball_radius = 0.9
maximal_radius = 0.3

# tesing = visual.draw_cross_section_traj(
#         10**3,
#         ball_radius,
#         mesh_out=15,
#         mesh_jump=20,
#         spread=10,
#         trials=10000,
#         partition=5,
#     )

# print(tesing)

fontsize = fontsize = 15 * 15 / 14
plt.rcParams.update({"text.usetex": True, "font.family": "Times"})
plt.figure(figsize=(10, 6))

for i, peclet in enumerate([10**3, 10**4, 10**5, 10**6]):
    fem_cross = visual.draw_cross_section_fem(
        peclet,
        ball_radius,
        maximal_radius=maximal_radius,
    )
    traj_cross = visual.draw_cross_section_traj(
        peclet,
        ball_radius,
        mesh_out=15,
        mesh_jump=20,
        spread=10,
        trials=10000,
        partition=5,
    )
    plt.plot(
        traj_cross[:, 0],
        traj_cross[:, 1],
        color=tableau[i],
        linestyle="None",
        ms=8,
        marker="o",
    )
    plt.plot(
        fem_cross[:, 0],
        fem_cross[:, 1],
        color=tableau[i],
        linestyle="-",
        ms=2,
        label=rf"$Pe = 10^{round(np.log10(peclet))}$",
    )
    jax.clear_caches()


traj_cross = visual.draw_cross_section_traj(
    10**8,
    ball_radius,
    mesh_out=15,
    mesh_jump=20,
    spread=10,
    trials=10000,
    partition=5,
)
plt.plot(
    traj_cross[:, 0],
    traj_cross[:, 1],
    color=tableau[4],
    linestyle="None",
    ms=8,
    marker="o",
)

plt.plot(
    [0, 0],
    [0, 0],
    color=tableau[4],
    linestyle="-",
    ms=2,
    label=rf"$Pe = 10^8$",
)

# add dummy plt to make legend
plt.plot([0], [-1], label="scikit-fem", color="k", linestyle="-")

plt.plot([0], [-1], label=r"pychastic", color="k", linestyle="None", ms=8, marker="o")

# Add labels and title
plt.vlines(
    [sf.streamline_radius(5, ball_radius)],
    [-0.05],
    [1.2],
    color="0.6",
    linestyles="--",
)

plt.xlim(0, maximal_radius)
plt.ylim(-0.05, 1.1)
plt.yticks(fontsize=fontsize)
plt.xticks(fontsize=fontsize)

plt.xlabel(r"Distance from axis $(\rho)$", fontsize=fontsize)
plt.ylabel(r"Hitting probability $(p)$", fontsize=fontsize)

plt.legend(fontsize=fontsize, frameon=False)
plt.tight_layout()
tosave = parent_dir / "graphics/cross_sections.pdf"
plt.savefig(tosave, bbox_inches="tight", pad_inches=0.02)

# Show the plot
plt.show()
