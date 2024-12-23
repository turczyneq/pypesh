import pypesh.visualisation as visual
import matplotlib.pyplot as plt
import pypesh.stokes_flow as sf
from pathlib import Path
import numpy as np

parent_dir = Path(__file__).parent

ball_radius = 0.9
maximal_radius = 0.3

fontsize = 15
plt.rcParams.update({"text.usetex": True, "font.family": "Cambria"})
plt.figure(figsize=(10, 6))

i = 1
for peclet in [10**4, 10**5, 10**6]:
    fem_cross = visual.draw_cross_section_fem(
        peclet, ball_radius, maximal_radius=maximal_radius
    )
    traj_cross = visual.draw_cross_section_traj(peclet, ball_radius,    mesh_out=4,
    mesh_jump=8,)
    plt.plot(
        traj_cross[:, 0],
        traj_cross[:, 1],
        color=f"C{i}",
        linestyle="None",
        ms=8,
        marker="o",
    )
    plt.plot(
        fem_cross[:, 0],
        fem_cross[:, 1],
        color=f"C{i}",
        linestyle="-",
        ms=2,
        label=rf"$Pe = 10^{round(np.log10(peclet))}$",
    )

    i += 1

# add dummy plt to make legend
plt.plot([0], [-1], label="FEM", color="k", linestyle="-")

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
plt.savefig(tosave)

# Show the plot
plt.show()
