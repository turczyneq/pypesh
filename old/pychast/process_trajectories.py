import numpy as np
import udajki as loc


def effective_radius(trajectory_array, ball_radius, floor_h):
    """
    Compute effective area of the particle from collision data
    ASSUMES UNIFORM SAMPLING!

    Parameters
    ----------
    trajectory_array : np.array
        2 by N array of tuples initial radius and outcome (1 - hit, 0 - miss)

    Returns
    -------
    float
        Effective radius of the particle
    """

    # Compute outcome as E[2\pi r hit(r)]
    radial_distance = trajectory_array[:, 0]
    is_hit = trajectory_array[:, 1]
    ver, vez = loc.velocities(radial_distance, floor_h, ball_radius)

    effective_area = np.max(radial_distance) * np.mean(
        2 * np.pi * radial_distance * is_hit * vez
    )

    effective_radius = (effective_area / np.pi) ** 0.5

    return effective_radius


def sherwood(
    trajectory_array,
    peclet, 
    ball_radius, 
    floor_h
):
    """
    Compute Sherwood number from collision data

    Parameters
    ----------
    trajectory_array : np.array
        2 by N array of tuples initial radius and outcome (1 - hit, 0 - miss)

    Returns
    -------
    float
        Sherwood number

    """
    r_eff = effective_radius(trajectory_array, ball_radius, floor_h)

    return (peclet / 4) * ((r_eff) ** 2)
