import pypesh.stokes_flow as sf
import numpy as np
import jax.numpy as jnp
import itertools


def test_no_slip():
    sphere_positions = np.array(
        [[np.cos(theta), np.sin(theta)] for theta in np.linspace(0, np.pi, 10)]
    )
    val = [sf.psi(r, z, 1) for r, z in sphere_positions]
    random_val = np.random.permutation(val)
    assert np.allclose(
        val, random_val
    ), "test if streamline is constant on surface of a sphere"


def inf_radius(ball_radius, r_start):
    """
    psi at z -> infinity has to be r**2/2, for z = 0, psi = (2*r_start + R)*(r_start - R)**2/(4*r_start), then the radius at infinty is sqrt((2 * r_start + R) * (r_start - R) ** 2 / (2 * r_start))

    Compare:
    https://en.wikipedia.org/wiki/Stokes%27_law#Transversal_flow_around_a_sphere

    Parameters
    ---------
    ball_radius: float
        Radius of ball.
    r_start: float
        Radius of searched streamline for z = 0

    Returns
    --------
    float
        radius of streamline at infinity
    """

    R = ball_radius
    return np.sqrt((2 * r_start + R) * (r_start - R) ** 2 / (2 * r_start))


def test_infinity():
    pars = itertools.product(np.linspace(0.1, 1, 10), np.linspace(0, 0.5, 10))
    positions = np.array([[R, R + r_syf] for R, r_syf in pars])

    heigh_inf = 10**8

    for pos in positions:
        assert np.isclose(
            sf.streamline_radius(heigh_inf, pos[0], pos[1]),
            inf_radius(pos[0], pos[1]),
            rtol=1e-05,
            atol=1e-08,
        ), "test if numerical radius of streamline is good at infinity"


# """
# TODO: write more tests for streamline and streamline_radius
# """
