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
import jax
import jax.numpy as jnp


def draw_pde_vs_sde(
    peclet,
    ball_radius,
    positions,
    limits=[-2.5, 2.5, -2.5, 5],
    t_max=30,
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

    def drift(q):
        return sf.stokes_around_sphere_jnp(q, ball_radius)

    _drift = jax.jit(drift)

    _diffusion_at_peclet = jax.jit(traj._diffusion_function(peclet=peclet))

    initial = jnp.array(
        [
            [
                [
                    x,
                    0,
                    -5,
                ]
                for i in range(amount)
            ]
            for x, amount in positions.items()
        ]
    )

    initial = jnp.concatenate(initial, axis=0)

    collision_data = collision_data = traj.simulate_trajectory(
        drift=_drift,
        noise=_diffusion_at_peclet,
        initial=initial,
        t_max=t_max,
        whole_trajectory=True,
    )

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

    #artefacts of past text size changes to fit text in paper
    fontsize = 15 * 1.3 * 11.5 / 16
    x_min, x_max, z_min, z_max = limits
    plt.rcParams.update(
        {"text.usetex": True, "font.family": "Times", "savefig.dpi": 300}
    )
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(19.5 * 0.6, 10.5 * 0.6),
        sharey=True,
        gridspec_kw={"width_ratios": [1, 1, 1]},
    )

    tric1 = axes[2].tripcolor(
        mesh.p[0],
        mesh.p[1],
        mesh.t.T,
        u,
        shading="gouraud",
        cmap="viridis",
        zorder=1,
        rasterized=True,
    )

    tric2 = axes[2].tripcolor(
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
    tric2.set_clim(vmin=0, vmax=1)

    trajectories = collision_data["trajectories"]
    for i in range(len(trajectories)):

        r = (trajectories[i, :, 0] ** 2 + trajectories[i, :, 1] ** 2) ** 0.5
        z = trajectories[i, :, -1]
        when_hit = np.concatenate((np.where(r**2 + z**2 < 1)[0], [-1]))[0]

        r = r[:when_hit]
        z = z[:when_hit]

        if collision_data["ball_hit"][i]:
            color = "C0"
        elif collision_data["something_hit"][i]:
            color = "C0"
        else:
            color = "#a22"

        if i % 2:
            axes[1].plot(r, z, color=color, linewidth=0.2, zorder=1, rasterized=True)
        else:
            axes[1].plot(-r, z, color=color, linewidth=0.2, zorder=1, rasterized=True)

    for ax in axes:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(z_min, z_max)
        ax.set_aspect(1)
        ax.tick_params(axis="x", labelsize=fontsize, top=True)

    axes[1].add_artist(
        plt.Circle(
            (0, 0),
            ball_radius,
            edgecolor="k",
            facecolor="#fff",
            hatch="///",
            zorder=2,
            linewidth=1,
        )
    )
    axes[1].add_artist(
        Arc(
            (0, 0),
            2,
            2,
            color="k",
            linestyle="--",
            theta1=0,
            theta2=360,
            zorder=2,
            linewidth=1,
        )
    )

    axes[2].add_artist(
        plt.Circle(
            (0, 0),
            ball_radius,
            edgecolor="k",
            facecolor="#fff",
            hatch="///",
            zorder=2,
            linewidth=1,
        )
    )
    axes[2].add_artist(
        Arc(
            (0, 0),
            2,
            2,
            color="w",
            linestyle="--",
            theta1=-90,
            theta2=90,
            zorder=2,
            linewidth=1,
        )
    )

    cmap = plt.get_cmap("viridis")
    axes[2].add_artist(
        plt.Circle(
            (0, 0), 1, edgecolor=None, facecolor=cmap(0), zorder=1, rasterized=True
        )
    )
    axes[0].tick_params(axis="y", labelsize=fontsize)
    axes[1].tick_params(axis="y", labelsize=fontsize, left=False)
    axes[2].tick_params(axis="y", labelsize=fontsize, left=False)
    # plt.xlabel(
    #     r"Radius (negative values for better visbility) $(\rho)$", fontsize=fontsize
    # )

    axes[0].set_ylabel(r"Along the flow $(z)$ [$a + b$]", fontsize=fontsize)
    axes[1].set_xlabel(r"Acros the flow $(\rho)$ [$a+b$]", fontsize=fontsize)

    for i, x in enumerate([r"(a)", r"(b)", r"(c)"]):
        axes[i].text(
            0.02,
            0.94,
            x,
            transform=axes[i].transAxes,
            fontsize=fontsize,
        )

    cbar_ax = fig.add_axes([0.92, 0.218, 0.02, 0.555])  # [left, bottom, width, height]
    cbar = fig.colorbar(tric1, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=fontsize)
    cbar.set_label(r"Concentration ($\varphi$)", fontsize=fontsize)
    plt.subplots_adjust(wspace=0.0)

    if save != "no":
        tosave = str(save)
        plt.savefig(
            tosave + ".pdf",
            bbox_inches="tight",
            pad_inches=0.02,
        )

    plt.show()

    return None


from pathlib import Path

parent_dir = Path(__file__).parent

tosave = parent_dir / "graphics/two_approaches"
draw_pde_vs_sde(
    500,
    0.7,
    {0: 4, 0.1: 4, 0.2: 4, 0.3: 4, 0.4: 4, 0.5: 4, 0.6: 4, 0.7: 4},
    limits=[-2.8, 2.8, -2.5, 4],
    save=tosave,
)
