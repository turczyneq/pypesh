import pypesh.fem as fem
import pypesh.analytic as analytic
import pypesh.trajectories as traj


def all_sherwood(
    peclet,
    ball_radius,
    mesh_out=4,
    mesh_jump=6,
    trials=10**2,
    floor_h=5,
    spread=4,
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
        float - Clift et. al., float - fem approach, float - trajectories approach


    Example
    --------
    >>> import pypesh.pesh as pesh
    >>> pesh.all_sherwood(1000, 0.9)
    (6.800655008742168, 12.033892568100546, 14.19422279233235)
    """
    sherwood_traj, xargs, yagrs = traj.sherwood_trajectories(
        peclet,
        ball_radius,
        mesh_out=mesh_out,
        mesh_jump=mesh_jump,
        trials=trials,
        floor_h=floor_h,
        spread=spread,
    )

    sherwood_fem = fem.sherwood_fem(peclet, ball_radius)

    sherwood_clift = analytic.clift_approximation(peclet)

    return sherwood_clift, sherwood_fem, sherwood_traj


def sherwood(
    peclet,
    ball_radius,
    mesh_out=4,
    mesh_jump=6,
    trials=10**2,
    floor_h=5,
    spread=4,
):
    """
    Calculates the sherwood number for given peclet and ball_radius basing on recomended approach. (WARNING) trajectories approach takes long to calculate and default mesh and trajectories are not supporting accurate values, recomended increasing trials, mesh_jump and mesh_out.

    Parameters
    ----------
    peclet : float, optional
        Peclet number defined as R u / D.

    ball_radius : float
        Radius of the big ball.

    mesh_out : int, optional
        For trajectories, amount of samples outside the region of highest slope.

    mesh_jump : int, optional
        For trajectories, amount of samples in the region of highest slope.

    trials : int, optional
        For trajectories, number of trajectories per position, uncertainty of propability estimation is sqrt(trials)/trials.

    floor_h : int, optional
        For trajectories, initial depth for simulation.

    spread: int, optional
        For trajectories, how far in sqrt(1/peclet), mesh_out will reach.

    Returns
    -------
    float
        Sherwood number for given peclet and ball_radius


    Example
    --------
    >>> import pypesh.pesh as pesh
    >>> pesh.sherwood(1000, 0.9)
    12.033892568100546
    """

    # TODO: only peclet is poor approach, should be on basis of pe*\beta**2
    if peclet < 5:
        sherwood = analytic.clift_approximation(peclet)
    elif peclet < 10**6:
        sherwood = fem.sherwood_fem(peclet, ball_radius)
    else:
        sherwood, xargs, yagrs = traj.sherwood_trajectories(
            peclet,
            ball_radius,
            mesh_out=mesh_out,
            mesh_jump=mesh_jump,
            trials=trials,
            floor_h=floor_h,
            spread=spread,
        )

    return sherwood
