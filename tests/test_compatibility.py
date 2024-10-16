import pypesh.fem as fem
import pypesh.trajectories as traj
import pypesh.analytic as analytic
import numpy as np


def test_compatibility():
    peclet = 10**4
    ball_radius = 0.9
    trials = 10**3
    pychastic, xargs, yargs = traj.sherwood_trajectories(
        peclet,
        ball_radius,
        trials=trials,
        mesh_jump=5,
        mesh_out=5,
        spread=2,
    )
    fem_sherwood = fem.sherwood_fem(peclet, ball_radius)

    assert np.isclose(pychastic, fem_sherwood, rtol=0.03), 'if fem and trajetories give simmilar sherwood'


# @pytest.mark.parametrize(
#     "peclet, expected",
#     [(pe, analytic.clift_approximation(pe)) for pe in np.logspace(1, 4.5, 10)],
# )
# def test_traj_vs_clift(peclet, expected):
#     sherwood = fem.sherwood_fem(peclet, 1)
#     assert np.isclose(
#         sherwood,
#         expected,
#         rtol=1e-8,
#         atol=0.5,
#     ), f"test if for ball_radius = 1, peclet = {peclet} fem is consistent with Clift et. al."
