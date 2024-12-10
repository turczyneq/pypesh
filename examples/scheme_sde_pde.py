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


def draw_scheme_sde_pde(
    peclet,
    ball_radius,
    positions,
    limits=[-2.5, 2.5, -2.5, 5],
    t_max=30,
    downstream_distance=3,
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

    # # # make scheme

    def stream(x, z):
        vx, vy, vz = sf.stokes_around_sphere_explicite(x, z, ball_radius)
        return vx, vz

    x_min, x_max, z_min, z_max = limits

    xlist = np.linspace(x_min - 1, 0, 400)
    zlist = np.linspace(-downstream_distance - 1, z_max, 400)

    X, Z = np.meshgrid(xlist, zlist)

    VX, VZ = stream(X, Z)

    speed = np.sqrt(VX**2 + VZ**2)

    def make_arrow(start, end, ax, scale=1, mutation_scale=2, tpercent=0.15):
        style = FenixArrowStyle(
            "fenix",
            head_length=2 * scale,
            head_width=2 * scale,
            tail_width=0.01 * scale,
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
        )
        ax.add_patch(arrow)
        return None

    def make_line(start, end, ax, ls='--'):
        ax.plot(
            start,
            end,
            zorder=3,
            c="k",
            linestyle=ls,
            linewidth=1,
        )
        return None

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

    # artefacts of past text size changes to fit text in paper
    fontsize = 15 * 1.3 * 11.5 / 16
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

    lw = 1.5 * speed
    many_arrows.streamplot_many_arrows(
        axes[0],
        X,
        Z,
        VX,
        VZ,
        density=5,
        broken_streamlines=False,
        start_points=[[x_val, -2.8] for x_val in np.linspace(-0.2, -x_max, 8)],
        num_arrows=18,
        integration_direction="forward",
        linewidth=lw,
        zorder=3,
        color="C1",
        arrowstyle=FenixArrowStyle(
            "fenix", head_length=0.4, head_width=0.4, tail_width=0.1
        ),
    )

    make_line([x_min, x_min], [z_min, z_max], axes[0], ls='-')
    make_line([x_min, 0], [z_max, z_max], axes[0], ls='-')
    make_line([x_min, 0], [z_min, z_min], axes[0], ls='-')

    make_arrow((0, -0.05), (0, 3.9), axes[0], scale = 0.6, mutation_scale=5, tpercent=0.05)
    make_arrow((-0.05, 0), (2.2, 0), axes[0], scale = 0.6, mutation_scale=5, tpercent=0.05)

    axes[0].text(
        0.53,
        0.95,
        r"$z$",
        transform=axes[0].transAxes,
        fontsize=fontsize,
    )
    axes[0].text(
        0.85,
        0.42,
        r"$\rho$",
        transform=axes[0].transAxes,
        fontsize=fontsize,
    )

    make_line([0, 0], [-3, 1], axes[0])
    make_line([0.7, 0.7], [0, 1.6], axes[0])
    make_line([1, 1], [0, 1.6], axes[0])

    make_arrow((0.5, 1.5), (0, 1.5), axes[0], tpercent=0.05)
    make_arrow((0.2, 1.5), (0.7, 1.5), axes[0], tpercent=0.05)

    make_arrow((0.2, 1.2), (0.7, 1.2), axes[0], tpercent=0.05)
    make_arrow((1.5, 1.2), (1, 1.2), axes[0], tpercent=0.05)

    axes[0].text(
        0.55,
        0.63,
        r"$a$",
        transform=axes[0].transAxes,
        fontsize=fontsize,
    )
    axes[0].text(
        0.72,
        0.59,
        r"$b$",
        transform=axes[0].transAxes,
        fontsize=fontsize,
    )

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

    axes[0].add_artist(
        plt.Circle(
            (0, 0),
            ball_radius,
            edgecolor="k",
            facecolor="#fff",
            hatch="////",
            zorder=2,
            linewidth=1,
        )
    )
    axes[0].add_artist(
        Arc(
            (0, 0),
            2,
            2,
            color="k",
            linestyle="--",
            theta1=0,
            theta2=360,
            zorder=2,
            linewidth=1.6,
        )
    )
    axes[0].add_artist(
        Wedge(
            (0.0, 0.0),
            1,
            -90,
            90,
            facecolor="C2",
            alpha=0.5,
        )
    )

    axes[1].add_artist(
        plt.Circle(
            (0.0, 0.0),
            ball_radius,
            edgecolor="k",
            facecolor="#fff",
            hatch="////",
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
            linewidth=1.6,
        )
    )

    axes[2].add_artist(
        plt.Circle(
            (0.0, 0.0),
            ball_radius,
            edgecolor="k",
            facecolor="#fff",
            hatch="////",
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
            linewidth=1.6,
        )
    )

    cmap = plt.get_cmap("viridis")
    axes[2].add_artist(
        plt.Circle(
            (0, 0), 1, edgecolor=None, facecolor=cmap(0), zorder=1, rasterized=True
        )
    )

    for ax in axes:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(z_min, z_max)
        ax.set_aspect(1)
        ax.tick_params(axis="x", labelsize=fontsize, top=True)

    axes[0].tick_params(axis="y", labelsize=fontsize)
    axes[1].tick_params(axis="y", labelsize=fontsize, left=False)
    axes[2].tick_params(axis="y", labelsize=fontsize, left=False)
    # plt.xlabel(
    #     r"Radius (negative values for better visbility) $(\rho)$", fontsize=fontsize
    # )

    axes[0].set_ylabel(r"Along the flow $(z)$ [$a + b$]", fontsize=fontsize)
    axes[1].set_xlabel(r"Acros the flow $(\rho)$ [$a+b$]", fontsize=fontsize)

    axes[0].text(
        0.023,
        0.9425,
        r"(a)",
        transform=axes[0].transAxes,
        fontsize=fontsize,
        backgroundcolor="#fff",
    )

    for i, x in enumerate([r"(b)", r"(c)"]):
        axes[i + 1].text(
            0.023,
            0.9425,
            x,
            transform=axes[i + 1].transAxes,
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
draw_scheme_sde_pde(
    500,
    0.7,
    {0: 4, 0.1: 4, 0.2: 4, 0.3: 4, 0.4: 4, 0.5: 4, 0.6: 4, 0.7: 4},
    limits=[-2.8, 2.8, -2.5, 4],
    save=tosave,
)
