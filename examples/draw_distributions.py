import pypesh.fem as fem
import pypesh.stokes_flow as sf
import pypesh.trajectories as traj
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Arc
from skfem import (
    BilinearForm,
)
from pathlib import Path


from skfem import asm, solve, condense
from skfem.helpers import grad, dot
import numpy as np

parent_dir = Path(__file__).parent


def draw_distributions_fem(
    peclet, ball_radius, limits=[-2.5, 2.5, -2.5, 5], draw_streamline=False
):
    """
    Draws distribution of concentration

    Parameters
    ----------
    peclet : float
        Peclet number defined as R u / D.

    ball_radius : float
        Radius of the big ball.

    limits : list
        minimal radius, maximal radius, minimal height, maximal height

    draw_streamline : boole, optional
        Default False, if True adds a streamline for `pe -> \infty`

    Returns
    --------
    None

    Example
    -------
    >>> import pypesh.visualisation as visual
    >>> visual.draw_distributions_fem(1000, 0.9)
    """

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

    x_min, x_max, z_min, z_max = limits
    fontsize = 20
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Cambria",
        "axes.titlesize": fontsize,
        "axes.labelsize": fontsize,
        "xtick.labelsize": fontsize,
        "ytick.labelsize": fontsize,
        "legend.fontsize": fontsize
    })

    scale = 0.002
    ratio = (scale * 2129, scale * 2839)

    plt.figure(dpi=200, figsize=ratio)

    plt.tripcolor(
        mesh.p[0],
        mesh.p[1],
        mesh.t.T,
        u,
        shading="gouraud",
        cmap="viridis",
        zorder=1,
    )
    plt.clim(vmin=0, vmax=1)  # Set color range
    plt.tripcolor(
        -mesh.p[0],
        mesh.p[1],
        mesh.t.T,
        u,
        shading="gouraud",
        cmap="viridis",
        zorder=1,
    )

    plt.clim(vmin=0, vmax=1)  # Set color range
    viridis = plt.get_cmap("viridis")

    plt.gca().add_artist(
        plt.Circle(
            (0, 0),
            1,
            edgecolor=viridis(0.0),
            facecolor=viridis(0.0),
            zorder=1,
            linewidth=0.5
        )
    )
    plt.gca().add_artist(
        plt.Circle(
            (0, 0),
            ball_radius,
            edgecolor="k",
            facecolor="#fff",
            hatch="/////",
            zorder=2,
            linewidth=0.5
        )
    )
    plt.gca().add_artist(
        Arc(
            (0, 0),
            2,
            2,
            color="w",
            linestyle="--",
            theta1=-90,
            theta2=90,
            zorder=2,
            linewidth=0.8,
        )
    )

    if draw_streamline:
        z_list = np.linspace(0, z_max, 100)
        rho_list = np.array([sf.streamline_radius(z, ball_radius) for z in z_list])

        plt.plot(rho_list, z_list, color="w", linestyle="--", zorder=3, linewidth=1)
        plt.plot(-rho_list, z_list, color="w", linestyle="--", zorder=3, linewidth=1)

    plt.gca().set_aspect(
        "equal", "box"
    )  # 'equal' ensures that one unit in x is equal to one unit in y
    plt.tight_layout()
    plt.xlim(x_min, x_max)
    plt.ylim(z_min, z_max)
    tosave = parent_dir/ "graphics/fem_pe500_rsyf_03.pdf"

    x_labels = np.linspace(-3, 3, 7)  # Adjust as needed
    y_labels = np.linspace(-2, 5, 8)

    plt.xticks(x_labels, [f"${int(x)}$" for x in x_labels], fontsize=fontsize)
    plt.yticks(y_labels, [f"${round(y)}$" for y in y_labels], fontsize=fontsize)


    plt.savefig(tosave, bbox_inches="tight")

    plt.show()

    return None


draw_distributions_fem(500, 0.7, [-3, 3, -2.5, 5])
