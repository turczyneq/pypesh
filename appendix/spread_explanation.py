import pypesh.visualisation as visual
import matplotlib.pyplot as plt
import pypesh.stokes_flow as sf
from pathlib import Path
import numpy as np

parent_dir = Path(__file__).parent

ball_radius = 0.9
maximal_radius = 0.3
peclet = 10**5
fem_cross = visual.draw_cross_section_fem(
    peclet, ball_radius, maximal_radius=maximal_radius
)

'''
WARNING trials 10**4 is expensive in computation time
'''
traj_cross = visual.draw_cross_section_traj(
    peclet,
    ball_radius,
    mesh_out=4,
    mesh_jump=10,
    trials=10000,
)


'''
to test plot setting
'''
# traj_cross = visual.draw_cross_section_traj(
#     peclet,
#     ball_radius,
#     mesh_out=4,
#     mesh_jump=10,
#     trials=200,
# )


spread = 4
stream_radius = sf.streamline_radius(5, ball_radius)


def dispersion(peclet):
    """
    Fucntion used to estimate dispesion
    """
    return 10 * (1 / peclet) ** (1 / 2)


fontsize = 15 * 15 / 14 * (23.8 / 21) * (11.34 / 12.23)
plt.rcParams.update({"text.usetex": True, "font.family": "Times"})
plt.figure(figsize=(10, 6))

plt.plot(
    traj_cross[:, 0],
    traj_cross[:, 1],
    color="C1",
    linestyle="None",
    ms=8,
    marker="o",
    label=rf"pychastic for $\textrm{{Pe}} = 10^{round(np.log10(peclet))}$",
)
plt.plot(
    fem_cross[:, 0],
    fem_cross[:, 1],
    color="C1",
    linestyle="-",
    ms=2,
    label=rf"scikit-fem for $\textrm{{Pe}} = 10^{round(np.log10(peclet))}$",
)

vertical_lines = [
    stream_radius - spread * dispersion(peclet),
    stream_radius - dispersion(peclet),
    stream_radius + dispersion(peclet),
    stream_radius + spread * dispersion(peclet),
]

plt.vlines(
    vertical_lines,
    [-0.05],
    [1.2],
    color="0.6",
    linestyles="--",
)

plt.vlines(
    [stream_radius],
    [-0.05],
    [1.2],
    color="0.2",
    linestyles="--",
)

plt.text(
    (vertical_lines[0] + vertical_lines[1]) / 2,
    1.15,
    r"\texttt{coarse}",
    ha="center",
    fontsize=fontsize,
)
plt.text(
    (vertical_lines[1] + vertical_lines[2]) / 2,
    1.15,
    r"\texttt{fine}",
    ha="center",
    fontsize=fontsize,
)
plt.text(
    (vertical_lines[2] + vertical_lines[3]) / 2,
    1.15,
    r"\texttt{coarse}",
    ha="center",
    fontsize=fontsize,
)


plt.xlim(0, maximal_radius)
plt.ylim(-0.05, 1.1)
plt.yticks(fontsize=fontsize)
plt.xticks(fontsize=fontsize)

plt.xlabel(r"Distance from axis $\rho$", fontsize=fontsize)
plt.ylabel(r"Hitting probability $p$", fontsize=fontsize)

plt.legend(
    fontsize=fontsize, frameon=True, facecolor="white", framealpha=1, edgecolor="none"
)
plt.tight_layout()
tosave = parent_dir / "graphics/regions.pdf"
plt.savefig(tosave, bbox_inches="tight", pad_inches=0.02)

# Show the plot
plt.show()
