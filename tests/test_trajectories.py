import pypesh.stokes_flow as sf
import pypesh.trajectories as traj
import numpy as np
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
        return 1 - (x / 2) ** 2

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
            traj._weighted_trapezoidal(dictionary, ball_radius, z),
            quad(to_check, 0, np.pi / 4)[0],
            rtol=1e-05,
            atol=5 * 1e-04,
        ), f"test if implemented integration of {str(fun)} is correct"


def test_probability_at_x():
    ball_list = [0.8, 0.9, 0.95]
    peclet = 10**9
    for ball_radius in ball_list:
        r_psi = sf.streamline_radius(5, ball_radius)
        yargs = [
            traj.hitting_propability_at_x(x_pos, peclet, ball_radius, trials=10**3)
            for x_pos in [
                r_psi - (1 - ball_radius) / 10,
                r_psi + (1 - ball_radius) / 10,
            ]
        ]
        assert np.allclose(
            yargs,
            [1, 0],
            rtol=1e-10,
            atol=1e-08,
        ), "test if for very heigh peclets inside stream radius is 1 and outside is 0"