import numpy as np


def sherwood_from_flux(flux, peclet):
    """
    sherwood is defined as flux_dimesional/(4 pi D R) = U R^2 flux / (4 pi D R) = flux*(Pe/4 pi)

    Parameters
    ----------
    flux: float
        Flux passing the ceiling

    peclet: float
        Peclet number defined as R u / D.

    Returns
    --------
    float
        sherwood from dimensionless flux

    Example
    -------
    >>> import pypesh.analytic as analytic
    >>> analytic.sherwood_from_flux(0.1, 1000)
    7.957747154594768
    """
    return flux * (peclet / (4 * np.pi))


def clift_approximation(pe):
    """
    Analytic approximation to numerical solution of Clift et. al.

    Parameters
    ----------
    pe : float
        Peclet number defined as R u / D.

    Returns
    -------
    float
        Sherwod calculated from Clift et. al. approximation

    Example
    --------
    >>> import pypesh.analytic as analytic
    >>> analytic.clift_approximation(1000)
    6.800655008742168
    """
    return (1 / 2) * (1 + (1 + 2 * pe) ** (1 / 3))


def our_approximation(peclet, ball_radius):
    """
    Our approximation of Sherwood number for `peclet` and `ball_radius`

    Parameters
    ----------
    peclet : float
        Peclet number defined as R u / D.

    ball_radius : float
        Radius of big ball

    Returns
    -------
    float
        Sherwod calculated our approximation

    Example
    --------
    >>> import pypesh.analytic as analytic
    >>> analytic.our_approximation(1000, 1)
    6.800655008742168
    >>> analytic.our_approximation(1000, 0.9)
    10.425655008742165 
    """

    beta = 1 - ball_radius

    return clift_approximation(peclet) + (peclet / 4) * (
        (beta**2) * (3 - beta) / 2
    )
