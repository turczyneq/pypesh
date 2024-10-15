import pytest
import pypesh.trajectories as traj
import pypesh.analytic as analytic
import numpy as np

@pytest.mark.parametrize(
    "peclet, expected",
    [(10**i, analytic.clift_approximation(10**i)) for i in range(1, 5)],
)
def test_traj_vs_clift(peclet, expected):
    trials = 10**3
    sherwood, xargs, yargs = traj.sherwood_trajectories(
        peclet,
        1,
        trials=trials,
        mesh_jump=10,
        mesh_out=5,
        spread=2,
    )
    assert np.isclose(
        sherwood,
        expected,
        rtol=np.sqrt(trials) / trials,
        atol=0.5,
    ), f"test if for ball_radius = 1, peclet = {peclet} pychastic is consistent with Clift et. al."
