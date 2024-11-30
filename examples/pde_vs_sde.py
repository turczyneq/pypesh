import pypesh.trajectories as traj
import pypesh.fem as fem
import pypesh.stokes_flow as sf
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from skfem import (
    BilinearForm,
)

from skfem import asm, solve, condense
from skfem.helpers import grad, dot
import numpy as np


def draw_pde_vs_sde(
    peclet,
    ball_radius,
    positions,
    limits=[-2.5, 2.5, -2.5, 5],
    t_max=20,
    downstream_distance=2.5,
    save="no",
):
    """
    Draws pde versus sde aproach.

    Parameters
    ----------
    peclet : float
        Peclet number defined as R u / D.

    ball_radius : float
        Radius of the big ball.

    positions: dict
        Keys: position where simulate, Values: how many times

    t_max : float, optional
        Default 20, time of simulation

    downstream_distance : float, optional
        Default 5, downstream distance in ball radius where distribution is plotted

    limits : list
        minimal radius, maximal radius, minimal height, maximal height

    Returns
    --------
    list
        list of outcomes from pychastic for each position

    Example
    -------
    TODO
    """

    # # # calculate multiple trajectories
    collision_data = [
        traj.draw_trajectory_at_x(
            x,
            peclet,
            ball_radius,
            trials=amount,
            floor_h=downstream_distance,
            t_max=t_max,
        )
        for x, amount in positions.items()
    ]

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

    mesh, basis = fem.get_mesh(peclet)

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

    fontsize = 15
    x_min, x_max, z_min, z_max = limits
    plt.rcParams.update({"text.usetex": True, "font.family": "Times"})
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13 * 0.6, 10.5 * 0.6),
        sharey=True,
        gridspec_kw={"width_ratios": [1, 1]},
    )

    tric1 = axes[1].tripcolor(
        mesh.p[0],
        mesh.p[1],
        mesh.t.T,
        u,
        shading="gouraud",
        cmap="viridis",
        zorder=1,
    )

    tric2 = axes[1].tripcolor(
        -mesh.p[0],
        mesh.p[1],
        mesh.t.T,
        u,
        shading="gouraud",
        cmap="viridis",
        zorder=1,
    )

    tric1.set_clim(vmin=0, vmax=1)
    tric2.set_clim(vmin=0, vmax=1)

    for data in collision_data:
        trajectories = data["trajectories"]
        for i in range(len(trajectories)):

            r = (trajectories[i, :, 0] ** 2 + trajectories[i, :, 1] ** 2) ** 0.5
            z = trajectories[i, :, -1]
            when_hit = np.concatenate((np.where(r**2 + z**2 < 1)[0], [-1]))[0]

            r = r[:when_hit]
            z = z[:when_hit]

            if data["ball_hit"][i]:
                color = "C0"
            elif data["something_hit"][i]:
                color = "C1"
            else:
                color = "#a22"

            if i % 2:
                axes[0].plot(
                    r,
                    z,
                    color=color,
                    linewidth=0.4,
                    zorder=1,
                )
            else:
                axes[0].plot(
                    -r,
                    z,
                    color=color,
                    linewidth=0.4,
                    zorder=1,
                )

    for ax in axes:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(z_min, z_max)
        ax.set_xlabel(r"Radius $(\rho)$", fontsize=fontsize)
        ax.set_aspect(1)
        ax.tick_params(axis="x", labelsize=fontsize, top=True)
        ax.add_artist(
            plt.Circle(
                (0, 0),
                ball_radius,
                edgecolor="k",
                facecolor="#fff",
                hatch="///",
                zorder=2,
            )
        )
        ax.add_artist(
            Arc(
                (0, 0),
                2,
                2,
                color="k",
                linestyle="--",
                theta1=0,
                theta2=360,
                zorder=2,
                linewidth=2,
            )
        )

    cmap = plt.get_cmap("viridis")
    axes[1].add_artist(
        plt.Circle(
            (0, 0),
            1,
            edgecolor=None,
            facecolor=cmap(0),
            zorder=1,
        )
    )

    axes[0].tick_params(axis="y", labelsize=fontsize)
    axes[1].tick_params(axis="y", labelsize=fontsize, left=False)
    # plt.xlabel(
    #     r"Radius (negative values for better visbility) $(\rho)$", fontsize=fontsize
    # )

    axes[0].set_ylabel(r"Height $(z)$", fontsize=fontsize)

    for i, x in enumerate([r"(a)", r"(b)"]):
        axes[i].text(
            0.02,
            0.95,
            x,
            transform=axes[i].transAxes,
            fontsize=fontsize,
        )

    cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])  # [left, bottom, width, height]
    cbar = fig.colorbar(tric1, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.set_label(r"Concentration ($\varphi$)", fontsize=fontsize)
    plt.subplots_adjust(wspace=0.0)

    if save != "no":
        tosave = str(save)
        plt.savefig(
            tosave + "png",
            bbox_inches="tight",
            pad_inches=0.02,
            dpi=600,
        )

    plt.show()

    return None


from pathlib import Path

parent_dir = Path(__file__).parent

tosave = parent_dir / "graphics/some_trajectiories"
draw_pde_vs_sde(1000, 0.8, {0.1: 2, 0.2: 1}, save=tosave)
