import pypesh.fem as fem
import pypesh.stokes_flow as sf
import pypesh.trajectories as traj
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Arc
from skfem import (
    BilinearForm,
)

from skfem import asm, solve, condense
from skfem.helpers import grad, dot
import numpy as np


def draw_cross_section_fem(
    peclet,
    ball_radius,
    downstream_distance=5,
    show=False,
    density=400,
    maximal_radius=1,
):
    """
    Draws cross section of hitting probability at selected height using scikit-fem.

    Parameters
    ----------
    peclet : float
        Peclet number defined as R u / D.

    ball_radius : float
        Radius of the big ball.

    downstream_distance : float, optional
        Default 5, downstream distance in ball radius where distribution is plotted

    show : bool, optional
        Default False, if True plot is shown

    denstity : int, optional
        Default 400, how many points in result

    maximal_radius : float, optional
        Default 1, maximal radius to place in result

    Returns
    --------
    np.array
        `density` by 2 array with location probability pairs.

    Example
    -------
    >>> import pypesh.visualisation as visual
    >>> visual.draw_cross_section_fem(1000, 0.9, density = 10)
    array([[ 0.00000000e+00,  7.96836968e-01],
        [ 1.11111111e-01,  6.95108509e-01],
        [ 2.22222222e-01,  4.52912665e-01],
        [ 3.33333333e-01,  2.11422004e-01],
        [ 4.44444444e-01,  6.74661539e-02],
        [ 5.55555556e-01,  1.42087363e-02],
        [ 6.66666667e-01,  1.95186128e-03],
        [ 7.77777778e-01,  1.57053994e-04],
        [ 8.88888889e-01,  5.67654858e-06],
        [ 1.00000000e+00, -2.40005303e-07]])
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

    if downstream_distance == 10:
        dofs = basis.get_dofs("top")
        vals = [
            [x, 1 - concenctration]
            for x, concenctration in zip(
                mesh.p[0, dofs.nodal["u"]], u[basis.get_dofs("top")]
            )
        ]
        vals = sorted(vals, key=lambda pair: pair[0])
        xargs, yargs = zip(*vals)
    else:
        # define where to probe function
        N_query_pts = density
        z_value = downstream_distance
        query_pts = np.vstack(
            [
                np.linspace(0, maximal_radius, N_query_pts),  # x[0] coordinate values
                z_value * np.ones(N_query_pts),  # x[1] coordinate values
            ]
        )
        # create the operator that will act on solution object to get function value
        probes = basis.probes(query_pts)

        xargs = np.linspace(0, maximal_radius, N_query_pts)
        yargs = 1 - probes @ u

    if show:
        plt.figure(figsize=(10, 6))
        plt.plot(xargs, yargs, "bo-", ms=2)

        # Add labels and title
        plt.xlabel("radius")
        plt.ylabel("propability")

        plt.vlines(
            [sf.streamline_radius(downstream_distance, ball_radius)],
            [-0.05],
            [1.2],
            color="0.1",
            linestyles="--",
        )

        plt.xlim(0, maximal_radius)
        plt.ylim(-0.05, 1.1)

        # Show the plot
        plt.show()

    return np.vstack((xargs, yargs)).T


def draw_cross_section_traj(
    peclet,
    ball_radius,
    downstream_distance=5,
    show=False,
    mesh_out=4,
    mesh_jump=6,
    spread=4,
    trials=200,
    partition=1,
    t_max=40,
):
    """
    Draws cross section of hitting probability at selected height using pychastic.

    Parameters
    ----------
    peclet : float
        Peclet number defined as R u / D.

    ball_radius : float
        Radius of the big ball.

    downstream_distance : float, optional
        Default 5, downstream distance in ball radius where distribution is plotted

    show : bool, optional
        Default False, if True plot is shown

    denstity : int, optional
        Default 10, how many points in result

    maximal_radius : float, optional
        Default 1, maximal radius to place in result

    mesh_out : int, optional
        Amount of samples outside the region of highest slope

    mesh_jump : int, optional
        Amount of samples in the region of highest slope

    spread: int, optional
        How far in sqrt(1/peclet), mesh_out will reach

    trials : int, optional
        Default 200, Number of trajectories.

    parition : int, optional
        Default 1, if to expensive in RAM partition into 'partition' parts.

    t_max: float, optional
        Default 40.0, select time of simmulation

    Returns
    --------
    np.array
        `density` by 2 array with location probability pairs.

    Example
    -------
    >>> import pypesh.visualisation as visual
    >>> visual.draw_cross_section_traj(1000, 0.9)
    array([[0.        , 0.77499998],
       [0.09137468, 0.67000002],
       [0.18274937, 0.47999999],
       [0.27412405, 0.315     ],
       [0.36549873, 0.155     ],
       [0.45687341, 0.04      ],
       [0.77310118, 0.        ],
       [1.08932895, 0.        ],
       [1.40555671, 0.        ]])
    """

    r_syf = sf.streamline_radius(downstream_distance, ball_radius)

    dispersion = 10 * (1 / peclet) ** (1 / 2)

    # generate the mesh to calculate the probability distribution
    if r_syf - dispersion > 0:
        x_probs = list(
            np.linspace(
                max(r_syf - spread * dispersion, 0), r_syf - dispersion, mesh_out
            )
        )
    else:
        x_probs = [0]
    x_probs = x_probs + list(
        np.linspace(max(r_syf - dispersion, 0), r_syf + dispersion, mesh_jump)
    )
    x_probs = x_probs + list(
        np.linspace(r_syf + dispersion, r_syf + spread * dispersion, mesh_out)
    )

    x_probs = list(dict.fromkeys(x_probs))

    to_return = np.array(
        [
            np.array(
                [
                    x,
                    traj.hitting_propability_at_x(
                        x,
                        peclet,
                        ball_radius,
                        trials=trials,
                        floor_h=downstream_distance,
                        partition=partition,
                        t_max=t_max,
                    ),
                ]
            )
            for x in x_probs
        ]
    )

    if show:
        plt.figure(figsize=(10, 6))
        plt.plot(to_return[:, 0], to_return[:, 1], "bo-", ms=2)

        # Add labels and title
        plt.xlabel("radius")
        plt.ylabel("propability")

        plt.vlines(
            [sf.streamline_radius(downstream_distance, ball_radius)],
            [-0.05],
            [1.2],
            color="0.1",
            linestyles="--",
        )

        plt.xlim(0, x_probs[-1])
        plt.ylim(-0.05, 1.1)

        # Show the plot
        plt.show()

    return to_return


def visualise_trajectories(
    peclet,
    ball_radius,
    positions,
    t_max=20,
    downstream_distance=5,
    show=False,
):
    """
    Draws trajectories simulated by pychastic.

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

    show : bool, optional
        Default False, if True plot is shown

    Returns
    --------
    list
        list of outcomes from pychastic for each position

    Example
    -------
    >>> visual.visualise_trajectories(1000, 0.9, {0.1: 2, 0.2: 1}, t_max=0.05)
    [{'ball_hit': Array([False, False], dtype=bool), 'roof_hit': Array([False, False], dtype=bool), 'something_hit': Array([False, False], dtype=bool), 'trajectories': Array([[[ 9.74556133e-02,  3.57697834e-03, -4.99383068e+00],
        [ 1.03875704e-01,  1.23539870e-03, -4.99430752e+00],
        [ 1.00019522e-01,  2.17465451e-03, -4.99006510e+00],
        [ 9.64764059e-02,  3.37144663e-03, -4.98079920e+00],
        [ 9.23867375e-02,  9.56971012e-03, -4.97192192e+00]],

       [[ 1.00915655e-01, -3.44433682e-03, -4.99829912e+00],
        [ 9.91058275e-02, -7.94267934e-03, -4.98968840e+00],
        [ 1.03579685e-01, -5.71627077e-03, -4.98568869e+00],
        [ 1.08699612e-01, -5.76060778e-03, -4.97635365e+00],
        [ 1.07397184e-01, -8.04143026e-03, -4.96779823e+00]]],      dtype=float32)}, {'ball_hit': Array([False], dtype=bool), 'roof_hit': Array([False], dtype=bool), 'something_hit': Array([False], dtype=bool), 'trajectories': Array([[[ 1.9748163e-01,  3.5769783e-03, -4.9938278e+00],
        [ 2.0392781e-01,  1.2353971e-03, -4.9943018e+00],
        [ 2.0009771e-01,  2.1746524e-03, -4.9900560e+00],
        [ 1.9658074e-01,  3.3714436e-03, -4.9807873e+00],
        [ 1.9251731e-01,  9.5697045e-03, -4.9719067e+00]]], dtype=float32)}]
    """

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

    fontsize = 15
    if show:
        plt.rcParams.update({"text.usetex": True, "font.family": "Cambria"})
        plt.figure(figsize=(12, 8))

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
                    plt.plot(r, z, color=color, linewidth=0.4)
                else:
                    plt.plot(-r, z, color=color, linewidth=0.4)

        plt.gca().add_artist(
            plt.Circle(
                (0, 0), ball_radius, edgecolor="k", facecolor="#fff", hatch="///"
            )
        )
        plt.gca().add_artist(
            Arc((0, 0), 2, 2, color="k", linestyle="--", theta1=-90, theta2=90)
        )

        plt.gca().set_aspect(
            "equal", "box"
        )  # 'equal' ensures that one unit in x is equal to one unit in y
        plt.tight_layout()
        plt.xlim(-2, 2)
        plt.ylim(-downstream_distance, 2)
        plt.xlabel(
            r"Radius (negative values for better visbility) $(\rho)$", fontsize=fontsize
        )
        plt.ylabel(r"Height $(z)$", fontsize=fontsize)

        plt.show()

    return collision_data


def visualise_trajectories_with_streamplot(
    peclet,
    ball_radius,
    positions,
    t_max=20,
    downstream_distance=5,
    show=False,
    save="no",
):
    """
    Draws trajectories simulated by pychastic.

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

    show : bool, optional
        Default False, if True plot is shown

    Returns
    --------
    list
        list of outcomes from pychastic for each position

    Example
    -------
    >>> visual.visualise_trajectories_with_streamplot(1000, 0.9, {0.1: 2, 0.2: 1}, t_max=0.05)
    [{'ball_hit': Array([False, False], dtype=bool), 'roof_hit': Array([False, False], dtype=bool), 'something_hit': Array([False, False], dtype=bool), 'trajectories': Array([[[ 9.74556133e-02,  3.57697834e-03, -4.99383068e+00],
        [ 1.03875704e-01,  1.23539870e-03, -4.99430752e+00],
        [ 1.00019522e-01,  2.17465451e-03, -4.99006510e+00],
        [ 9.64764059e-02,  3.37144663e-03, -4.98079920e+00],
        [ 9.23867375e-02,  9.56971012e-03, -4.97192192e+00]],

       [[ 1.00915655e-01, -3.44433682e-03, -4.99829912e+00],
        [ 9.91058275e-02, -7.94267934e-03, -4.98968840e+00],
        [ 1.03579685e-01, -5.71627077e-03, -4.98568869e+00],
        [ 1.08699612e-01, -5.76060778e-03, -4.97635365e+00],
        [ 1.07397184e-01, -8.04143026e-03, -4.96779823e+00]]],      dtype=float32)}, {'ball_hit': Array([False], dtype=bool), 'roof_hit': Array([False], dtype=bool), 'something_hit': Array([False], dtype=bool), 'trajectories': Array([[[ 1.9748163e-01,  3.5769783e-03, -4.9938278e+00],
        [ 2.0392781e-01,  1.2353971e-03, -4.9943018e+00],
        [ 2.0009771e-01,  2.1746524e-03, -4.9900560e+00],
        [ 1.9658074e-01,  3.3714436e-03, -4.9807873e+00],
        [ 1.9251731e-01,  9.5697045e-03, -4.9719067e+00]]], dtype=float32)}]
    """

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

    fontsize = 15
    if show:

        import pypesh.stokes_flow as sf

        r_list = np.linspace(-2, 2, 100)
        z_list = np.linspace(-downstream_distance, 2, 100)

        RR, ZZ = np.meshgrid(r_list, z_list)

        psi_list = sf.psi(RR, ZZ, ball_radius)
        mask = RR**2 + ZZ**2 < (ball_radius) ** 2
        psi_list = np.ma.masked_where(mask, psi_list)

        plt.rcParams.update({"text.usetex": True, "font.family": "Cambria"})
        plt.figure(figsize=(12, 8))

        plt.contour(
            RR,
            ZZ,
            psi_list,
            levels=[10 ** (-3), 10 ** (-2), 0.1, 0.5],
            alpha=0.3,
            linewidths=3,
            colors="k",
        )

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
                    plt.plot(r, z, color=color, linewidth=0.4)
                else:
                    plt.plot(-r, z, color=color, linewidth=0.4)

        plt.gca().add_artist(
            plt.Circle(
                (0, 0), ball_radius, edgecolor="k", facecolor="#fff", hatch="///"
            )
        )
        plt.gca().add_artist(
            Arc((0, 0), 2, 2, color="k", linestyle="--", theta1=-180, theta2=180)
        )

        plt.gca().set_aspect(
            "equal", "box"
        )  # 'equal' ensures that one unit in x is equal to one unit in y
        plt.xlim(-2, 2)
        plt.ylim(-downstream_distance, 2)
        plt.xlabel(
            r"Radius (negative values for better visbility) $(\rho)$", fontsize=fontsize
        )
        plt.ylabel(r"Height $(z)$", fontsize=fontsize)
        plt.tight_layout()

        if save != "no":
            plt.savefig(save)

        plt.show()

    return collision_data


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

    fontsize = 15
    x_min, x_max, z_min, z_max = limits
    # cmap = mpl.colormaps['viridis']
    plt.rcParams.update({"text.usetex": True, "font.family": "Cambria"})

    plt.figure(figsize=(12, 8))

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
    plt.gca().add_artist(
        plt.Circle(
            (0, 0),
            ball_radius,
            edgecolor="k",
            facecolor="#fff",
            hatch="///",
            zorder=2,
        )
    )
    plt.gca().add_artist(
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
    plt.xlabel(
        r"Radius (negative values for better visbility) $(\rho)$", fontsize=fontsize
    )
    plt.ylabel(r"Height $(z)$", fontsize=fontsize)

    plt.show()

    return None
