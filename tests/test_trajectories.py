import pypesh.stokes_flow as sf
import pypesh.trajectories as traj
import pypesh.analytic as analytic
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
            traj.weighted_trapezoidal(dictionary, ball_radius, z),
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

def test_traj_vs_clift_pe_1():
    peclet = 10**1
    trials = 10**3
    sherwood, xargs, yargs = traj.distribution(
        peclet, 1, trials=trials, mesh_jump=10, mesh_out=10, spread = 2,
    )
    assert np.isclose(
        sherwood,
        analytic.clift_approximation(peclet),
        rtol=np.sqrt(trials) / trials,
        atol=0.5,
    ), f"test if for ball_radius = 1, peclet = {peclet} pychastic is consistent with Clift et. al."

def test_traj_vs_clift_pe_2():
    peclet = 10**2
    trials = 10**3
    sherwood, xargs, yargs = traj.distribution(
        peclet, 1, trials=trials, mesh_jump=10, mesh_out=10, spread = 2,
    )
    assert np.isclose(
        sherwood,
        analytic.clift_approximation(peclet),
        rtol=np.sqrt(trials) / trials,
        atol=0.5,
    ), f"test if for ball_radius = 1, peclet = {peclet} pychastic is consistent with Clift et. al."

def test_traj_vs_clift_pe_3():
    peclet = 10**3
    trials = 10**3
    sherwood, xargs, yargs = traj.distribution(
        peclet, 1, trials=trials, mesh_jump=10, mesh_out=10, spread = 2,
    )
    assert np.isclose(
        sherwood,
        analytic.clift_approximation(peclet),
        rtol=np.sqrt(trials) / trials,
        atol=0.5,
    ), f"test if for ball_radius = 1, peclet = {peclet} pychastic is consistent with Clift et. al."

def test_traj_vs_clift_pe_4():
    peclet = 10**4
    trials = 10**3
    sherwood, xargs, yargs = traj.distribution(
        peclet, 1, trials=trials, mesh_jump=10, mesh_out=10, spread = 2,
    )
    assert np.isclose(
        sherwood,
        analytic.clift_approximation(peclet),
        rtol=np.sqrt(trials) / trials,
        atol=0.5,
    ), f"test if for ball_radius = 1, peclet = {peclet} pychastic is consistent with Clift et. al."

# def test_traj_vs_clift_pe_2():
#     pelist = [10**i for i in range(1, 5)]
#     for peclet in pelist:
#         trials = 10**3
#         sherwood, xargs, yargs = traj.distribution(
#             peclet, 1, trials=trials, mesh_jump=10, mesh_out=10, spread = 2,
#         )
#         assert np.isclose(
#             sherwood,
#             analytic.clift_approximation(peclet),
#             rtol=np.sqrt(trials) / trials,
#             atol=0.5,
#         ), "test if for ball_radius = 1 pychastic is consistent with Clift et. al."
