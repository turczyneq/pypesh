import pypesh.generate_mesh as gen_msh
import skfem.visuals.matplotlib as skmpl
import matplotlib.pyplot as plt
import pypesh.trajectories as traj
import pypesh.fem as fem
import pypesh.stokes_flow as sf
import pypesh.mpl.streamplot_many_arrows as many_arrows
from pypesh.mpl.fenix_arrow import FenixArrowStyle
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Wedge
from matplotlib.patches import FancyArrowPatch, ArrowStyle
from skfem import (
    BilinearForm,
)

from skfem import asm, solve, condense
from skfem.helpers import grad, dot
import numpy as np
import jax
import jax.numpy as jnp

peclet = 500
ball_radius = 0.7

# # # graphical utils


def make_arrow(start, end, ax, scale=1, mutation_scale=2, tpercent=0.15):
    style = FenixArrowStyle(
        "fenix",
        head_length=2 * scale,
        head_width=2 * scale,
        tail_width=0.008 * scale,
        tpercent=tpercent,
    )
    arrow = FancyArrowPatch(
        start,
        end,
        mutation_scale=mutation_scale,
        arrowstyle=style,
        color="k",
        zorder=3,
        joinstyle="miter",
        clip_on=False,
    )
    ax.add_patch(arrow)
    return None


def make_line(start, end, ax, ls="--", lw=1):
    ax.plot(
        start,
        end,
        zorder=3,
        c="k",
        linestyle=ls,
        linewidth=lw,
        clip_on=False,
    )
    return None


# # # make mesh

mesh, basis = fem.get_mesh(peclet)


# # # calculate solution by fem
@BilinearForm
def advection(k, l, m):
    # Coordinate fields
    r, z = m.x

    v_r, v_y, v_z = sf.stokes_around_sphere_explicite(r, z, ball_radius)

    return (l * v_r * grad(k)[0] + l * v_z * grad(k)[1]) * 2 * np.pi * r


@BilinearForm
def claplace(u, v, m):
    """Laplace operator in cylindrical coordinates."""
    r, z = m.x
    return dot(grad(u), grad(v)) * 2 * np.pi * r


# Assemble the system matrix
A = asm(claplace, basis) + peclet * asm(advection, basis)
# Identify the interior degrees of freedom
interior = basis.complement_dofs(basis.get_dofs({"bottom", "ball"}))
# Boundary condition
u = basis.zeros()
u[basis.get_dofs("bottom")] = 1.0
u[basis.get_dofs("ball")] = 0.0
# Solve the problem
u = solve(*condense(A, x=u, I=interior))

fontsize = 10 * 10 / 6
plt.rcParams.update({"text.usetex": True, "font.family": "Times", "savefig.dpi": 300})
fig = plt.figure(figsize=(11.5 * 0.6, 11.5 * 0.6))


height = 11
make_arrow(
    (1, -height), (10, -height), plt.gca(), scale=0.6, mutation_scale=4, tpercent=0.05
)
make_arrow(
    (9, -height), (0, -height), plt.gca(), scale=0.6, mutation_scale=4, tpercent=0.05
)
make_line([0, 0], [-10, -height - 0.5], plt.gca(), lw=0.8)
make_line([10, 10], [-10, -height - 0.5], plt.gca(), lw=0.8)
plt.gca().text(
    5,
    -10.8,
    r"width$\times$cell\_size",
    ha="center",
    # transform=plt.gca().transAxes,
    fontsize=fontsize,
)

dist = 11
make_line([10, dist + 0.5], [-10, -10], plt.gca(), lw=0.8)
make_line([0, dist + 0.5], [0, 0], plt.gca(), lw=0.8)
make_line([10, dist + 0.5], [10, 10], plt.gca(), lw=0.8)
make_arrow((dist, -9), (dist, 0), plt.gca(), scale=0.6, mutation_scale=4, tpercent=0.05)
make_arrow(
    (dist, -1), (dist, -10), plt.gca(), scale=0.6, mutation_scale=4, tpercent=0.05
)
make_arrow((dist, 9), (dist, 0), plt.gca(), scale=0.6, mutation_scale=4, tpercent=0.05)
make_arrow((dist, 1), (dist, 10), plt.gca(), scale=0.6, mutation_scale=4, tpercent=0.05)
plt.gca().text(
    dist - 0.35,
    -5,
    r"floor$\times$cell\_size",
    ha="center",
    va="center",
    # transform=plt.gca().transAxes,
    fontsize=fontsize,
    rotation=90,
)
plt.gca().text(
    dist - 0.35,
    5,
    r"ceiling$\times$cell\_size",
    ha="center",
    va="center",
    # transform=plt.gca().transAxes,
    fontsize=fontsize,
    rotation=90,
)

plt.gca().text(
    0.5,
    5,
    r"mesh",
    c="w",
    ha="center",
    va="center",
    # transform=plt.gca().transAxes,
    fontsize=fontsize,
    rotation=90,
)

plt.gca().text(
    6,
    -7,
    r"far_mesh",
    ha="center",
    va="center",
    # transform=plt.gca().transAxes,
    fontsize=fontsize,
    backgroundcolor="#fff",
    bbox=dict(color="w", alpha=0.8, linewidth=1e-10, joinstyle="round"),
)


tric1 = plt.tripcolor(
    -mesh.p[0],
    mesh.p[1],
    mesh.t.T,
    u,
    shading="gouraud",
    cmap="viridis",
    zorder=1,
    rasterized=True,
)
tric1.set_clim(vmin=0, vmax=1)

skmpl.draw(mesh, ax=plt.gca(), zorder=0)

plt.gca().add_artist(
    Wedge(
        (0.0, 0.0),
        ball_radius,
        theta1=90,
        theta2=-90,
        edgecolor="k",
        facecolor="#fff",
        hatch="/////",
        zorder=2,
        linewidth=1e-10,
    )
)
plt.gca().add_artist(
    Arc(
        (0, 0),
        2,
        2,
        color="w",
        linestyle="--",
        theta1=90,
        theta2=-90,
        zorder=2,
        linewidth=1,
    )
)

cmap = plt.get_cmap("viridis")
plt.gca().add_artist(
    Wedge(
        (0.0, 0.0),
        1.0,
        theta1=90,
        theta2=-90,
        edgecolor=cmap(0),
        facecolor=cmap(0),
        zorder=1,
        linewidth=1e-10,
    )
)

plt.xlim(-10, 10)
plt.ylim(-10, 10)

plt.tick_params(
    which="both", left=False, labelleft=False, bottom=False, labelbottom=False
)


cbar_ax = fig.add_axes([0.1, 0.11, 0.02, 0.771])  # [left, bottom, width, height]
cbar = fig.colorbar(tric1, cax=cbar_ax, location="left")
cbar.ax.tick_params(labelsize=fontsize)
cbar.set_label(r"Concentration ($\varphi$)", fontsize=fontsize)
plt.subplots_adjust(wspace=-0.01)


from pathlib import Path

parent_dir = Path(__file__).parent
tosave = parent_dir / "graphics/mesh_and_solution.pdf"

plt.savefig(
    tosave,
    bbox_inches="tight",
    pad_inches=0.02,
)

# plt.show()
