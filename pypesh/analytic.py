import numpy as np

def sherwood_from_flux(flux, peclet):
    '''
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
    '''
    return flux*(peclet/(4*np.pi))


def clift_approximation(pe):
    '''
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
    '''
    return (1 / 2) * (1 + (1 + 2 * pe) ** (1 / 3))
