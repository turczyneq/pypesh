def all_sherwood(
    peclet,
    ball_radius,
    mesh_out=4,
    mesh_jump=6,
    trials=10**2,
    floor_h=5,
    spread=4,
    t_max=40.0,
    partition=1,
):
    """
    Calculates the sherwood number using clift approximation, fem approach and trajectories approach.

    Parameters
    ----------
    peclet : float, optional
        Peclet number defined as R u / D.

    ball_radius : float
        Radius of the big ball.

    mesh_out : int, optional
        For trajectories, amount of samples outside the region of highest slope

    mesh_jump : int, optional
        For trajectories, amount of samples in the region of highest slope

    trials : int, optional
        For trajectories, number of trajectories per position, uncertainty of propability estimation is sqrt(trials)/trials.

    floor_h : int, optional
        For trajectories, initial depth for simulation

    spread: int, optional
        For trajectories, how far in sqrt(1/peclet), mesh_out will reach

    Returns
    -------
    tuple
        float - Clift et. al., float - our approximation, float - fem approach, float - fem approach different integral , float - trajectories approach


    Example
    --------
    >>> import pypesh.pesh as pesh
    >>> pesh.all_sherwood(1000, 0.9)
    (6.800655008742168, 10.425655008742165, np.float64(12.033892568100546), np.float64(11.720084145681978), 14.194223139015682)
    """

    import pypesh.fem as fem
    import pypesh.analytic as analytic
    import pypesh.trajectories as traj

    if peclet <= 1e3:
        sherwood_traj = 0
    else:
        sherwood_traj, xargs, yagrs = traj.sherwood_trajectories(
            peclet,
            ball_radius,
            mesh_out=mesh_out,
            mesh_jump=mesh_jump,
            trials=trials,
            floor_h=floor_h,
            spread=spread,
            t_max=t_max,
            partition=partition,
        )

    sherwood_fem = fem.sherwood_fem(peclet, ball_radius)

    sherwood_fem_sphere = fem._sherwood_fem_different_integral(peclet, ball_radius)

    sherwood_clift = analytic.clift_approximation(peclet)

    sherwood_us = analytic.our_approximation(peclet, ball_radius)

    return sherwood_clift, sherwood_us, sherwood_fem, sherwood_fem_sphere, sherwood_traj


def _sherwood(
    peclet,
    ball_radius,
):
    """
    Interpolate the sherwood number for given peclet and ball_radius.

    Parameters
    ----------
    peclet : float, optional
        Peclet number defined as R u / D.

    ball_radius : float
        Radius of the big ball.

    Returns
    -------
    float
        Sherwood number for given peclet and ball_radius


    Example
    --------
    >>> import pypesh.pesh as pesh
    >>> pesh._sherwood(1000, 0.9)
    12.033892568100546
    """

    from pathlib import Path
    import numpy as np
    from scipy.interpolate import interpn

    parent_dir = Path(__file__).parent

    to_interpolate = parent_dir / "data/pesh.csv"
    to_interpolate = np.loadtxt(to_interpolate, delimiter=",")

    sherwood_dict = {(pe, ball_radius): sh for pe, ball_radius, sh in to_interpolate}
    pe_list = np.unique(to_interpolate[:, [0]])
    ball_list = np.unique(to_interpolate[:, [1]])

    if (
        pe_list[0] > peclet
        or peclet > pe_list[-1]
        or ball_list[0] > ball_radius
        or ball_radius > ball_list[-1]
    ):
        print(
            "requested vaule is outside so far calculated regime, you can calculate value yourself using all_sherwood\n or contact package admins with requested expansion of region"
        )
        return None

    points = (pe_list, ball_list)
    sherwood_matrix = [
        [sherwood_dict[(pe, ball_radius)] for ball_radius in ball_list]
        for pe in pe_list
    ]

    return interpn(points, sherwood_matrix, np.array([peclet, ball_radius]))[0]


def sherwood_heatmap():
    """
    Interpolate the sherwood number for given peclet and ball_radius and returns a function.

    Parameters
    ----------

    Returns
    -------
    function
        Gives sherwood number for given peclet and ball_radius

    Example
    --------
    >>> import pypesh.pesh as pesh
    >>> import numpy as np
    >>> interp = pesh.sherwood_heatmap()
    >>> to_call = np.array([1000,0.9])
    >>> interp(to_call)
    array([12.03389257])
    """

    from pathlib import Path
    import numpy as np
    from scipy.interpolate import RegularGridInterpolator

    parent_dir = Path(__file__).parent

    to_interpolate = parent_dir / "data/pesh.csv"
    to_interpolate = np.loadtxt(to_interpolate, delimiter=",")

    sherwood_dict = {(pe, ball_radius): sh for pe, ball_radius, sh in to_interpolate}
    pe_list = np.unique(to_interpolate[:, [0]])
    ball_list = np.unique(to_interpolate[:, [1]])

    points = (pe_list, ball_list)
    sherwood_matrix = [
        [sherwood_dict[(pe, ball_radius)] for ball_radius in ball_list]
        for pe in pe_list
    ]

    return RegularGridInterpolator(points, sherwood_matrix)


# def sherwood(
#     peclet,
#     ball_radius,
#     mesh_out=4,
#     mesh_jump=6,
#     trials=10**2,
#     floor_h=5,
#     spread=4,
# ):
#     """
#     Calculates the sherwood number for given peclet and ball_radius basing on recomended approach. (WARNING) trajectories approach takes long to calculate and default mesh and trajectories are not supporting accurate values, recomended increasing trials, mesh_jump and mesh_out.

#     Parameters
#     ----------
#     peclet : float, optional
#         Peclet number defined as R u / D.

#     ball_radius : float
#         Radius of the big ball.

#     mesh_out : int, optional
#         For trajectories, amount of samples outside the region of highest slope.

#     mesh_jump : int, optional
#         For trajectories, amount of samples in the region of highest slope.

#     trials : int, optional
#         For trajectories, number of trajectories per position, uncertainty of propability estimation is sqrt(trials)/trials.

#     floor_h : int, optional
#         For trajectories, initial depth for simulation.

#     spread: int, optional
#         For trajectories, how far in sqrt(1/peclet), mesh_out will reach.

#     Returns
#     -------
#     float
#         Sherwood number for given peclet and ball_radius


#     Example
#     --------
#     >>> import pypesh.pesh as pesh
#     >>> pesh.sherwood(1000, 0.9)
#     12.033892568100546
#     """

#     import pypesh.fem as fem
#     import pypesh.analytic as analytic
#     import pypesh.trajectories as traj

#     # TODO: only peclet is poor approach, should be on basis of pe*\beta**2
#     if peclet < 5:
#         sherwood = analytic.clift_approximation(peclet)
#     elif peclet < 10**6:
#         sherwood = fem.sherwood_fem(peclet, ball_radius)
#     else:
#         sherwood, xargs, yagrs = traj.sherwood_trajectories(
#             peclet,
#             ball_radius,
#             mesh_out=mesh_out,
#             mesh_jump=mesh_jump,
#             trials=trials,
#             floor_h=floor_h,
#             spread=spread,
#         )

#     return sherwood


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Calculates Sherwood Number for sphere with:\n"
            + "given Peclet number and absorptive diameter\n"
        )
    )

    parser.add_argument(
        "--peclet",
        type=float,
        required=True,
        help="Peclet number of a problem",
    )

    parser.add_argument(
        "--ball_radius",
        type=float,
        required=True,
        help="Radius of ball that affects the stokes flow",
    )

    args = parser.parse_args()

    result = _sherwood(
        peclet=args.peclet,
        ball_radius=args.ball_radius,
    )

    print(f"Sherwood for given parameters is {result}")


if __name__ == "__main__":
    main()
