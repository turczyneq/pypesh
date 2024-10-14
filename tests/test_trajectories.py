import pypesh.stokes_flow as sf
import pypesh.trajectories as traj
import numpy as np
import jax.numpy as jnp
import itertools
from scipy.integrate import quad


def test_integrating():
    z = 5
    ball_radius = 0.9

    def vz(x):
        velocities = sf.stokes_around_sphere_np(np.array([x, 0, z]), ball_radius)
        return velocities[2]

    def weight(x):
        return 2 * np.pi * x * vz(x)

    xargs = np.linspace(0, np.pi / 4, 20)
    def square(x):
        return 1 - (x/2)**2

    def tanh_expression(x):
        x0 = 0.1
        a = 0.1
        return 0.5 * (-np.tanh((x - x0) / a) + np.tanh((x + x0) / a))
    
    function_list = [np.cos, np.tanh, square, tanh_expression]

    for fun in function_list:
        dictionary = {x: fun(x) for x in xargs}

        def to_check(x):
            return weight(x) * fun(x)

        assert np.isclose(
            traj.weighted_trapezoidal(dictionary, ball_radius, z),
            quad(to_check, 0, np.pi / 4)[0],
            rtol=1e-05,
            atol=5*1e-04,
        ), f"test if implemented integration of {str(fun)} is correct"


# def inf_radius(ball_radius, r_start):
#     """
#     psi at z -> infinity has to be r**2/2, for z = 0, psi = (2*r_start + R)*(r_start - R)**2/(4*r_start), then the radius at infinty is sqrt((2 * r_start + R) * (r_start - R) ** 2 / (2 * r_start))

#     Compare:
#     https://en.wikipedia.org/wiki/Stokes%27_law#Transversal_flow_around_a_sphere

#     Parameters
#     ---------
#     ball_radius: float
#         Radius of ball.
#     r_start: float
#         Radius of searched streamline for z = 0

#     Returns
#     --------
#     float
#         radius of streamline at infinity
#     """

#     R = ball_radius
#     return np.sqrt((2 * r_start + R) * (r_start - R) ** 2 / (2 * r_start))


# def test_infinity():
#     pars = itertools.product(np.linspace(0.1, 1, 10), np.linspace(0, 0.5, 10))
#     positions = np.array([[R, R + r_syf] for R, r_syf in pars])

#     heigh_inf = 10**8

#     for pos in positions:
#         assert np.isclose(
#             sf.streamline_radius(heigh_inf, pos[0], pos[1]),
#             inf_radius(pos[0], pos[1]),
#             rtol=1e-05,
#             atol=1e-08,
#         ), "test if numerical radius of streamline is good at infinity"
