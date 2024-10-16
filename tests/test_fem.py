import pytest
import pypesh.fem as fem
import pypesh.analytic as analytic
import numpy as np


@pytest.mark.parametrize(
    "peclet, expected",
    [(pe, analytic.clift_approximation(pe)) for pe in np.logspace(1, 4.5, 10)],
)
def test_traj_vs_clift(peclet, expected):
    sherwood = fem.sherwood_fem(peclet, 1)
    assert np.isclose(
        sherwood,
        expected,
        rtol=1e-8,
        atol=0.5,
    ), f"test if for ball_radius = 1, peclet = {peclet} fem is consistent with Clift et. al."
