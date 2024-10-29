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

    sherwood_us = analytic.our_approximation(peclet, ball_radius)

    return sherwood_clift, sherwood_us, sherwood_fem, sherwood_traj


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

def main():
    import argparse

    parser = argparse.ArgumentParser(
    description=(
        "Calculates Sherwood Number for sphere with:\n"
        + "given Peclet number and absorptive diameter\n"))

    parser.add_argument(
    "--peclet",
    type=float,
    required=True,
    help="Peclet number of a problem")
    
    parser.add_argument(
    "--ball_radius",
    type=float,
    required=True,
    help="Radius of ball that affects the stokes flow",)

    parser.add_argument(
    "--mesh_out",
    type=int,
    default=4,
    help="For trajectories, amount of samples outside the region of highest slope.",)

    parser.add_argument(
    "--mesh_jump",
    type=int,
    default=6,
    help="For trajectories, amount of samples in the region of highest slope.",)

    parser.add_argument(
    "--trials",
    type=int,
    default=10**2,
    help="For trajectories, number of trajectories per position, uncertainty of propability estimation is sqrt(trials)/trials.")

    parser.add_argument(
    "--floor_h",
    type=float,
    default=5,
    help="For trajectories, initial depth for simulation.",)

    parser.add_argument(
    "--spread",
    type=float,
    default=4,
    help="For trajectories, how far in sqrt(1/peclet), mesh_out will reach.",)

    args = parser.parse_args()

    result = sherwood(
    peclet = args.peclet,
    ball_radius = args.ball_radius,
    mesh_out = args.mesh_out,
    mesh_jump = args.mesh_jump,
    trials = args.trials,
    floor_h = args.floor_h,
    spread = args.spread,)

    print(f"Sherwood for given parameters is {result}")

if __name__ == "__main__":
    main()