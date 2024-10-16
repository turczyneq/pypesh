import pypesh.stokes_flow as sf
import numpy as np
import jax.numpy as jnp
import itertools


def test_np_jnp():
    angles = itertools.product(np.linspace(0, 2 * np.pi, 10), np.linspace(0, np.pi, 10))
    radius = 10
    np_positions = radius * np.array(
        [
            [np.cos(phi) * np.cos(theta), np.sin(phi) * np.cos(theta), np.sin(theta)]
            for phi, theta in angles
        ]
    )
    for pos in np_positions:
        assert np.allclose(
            sf.stokes_around_sphere_np(pos, 1),
            sf.stokes_around_sphere_jnp(jnp.array(pos), 1),
            rtol=1e-05,
            atol=1e-08,
        ), "test if function works for np and jnp"


def test_np_explicite():
    angles = np.linspace(0, np.pi, 10)
    radius = 10
    np_positions = radius * np.array(
        [
            [np.cos(theta), 0, np.sin(theta)]
            for theta in angles
        ]
    )
    for pos in np_positions:
        print(sf.stokes_around_sphere_np(pos, 1))
        print(sf.stokes_around_sphere_explicite(pos[0], pos[2], 1))
        assert np.allclose(
            sf.stokes_around_sphere_np(pos, 1),
            sf.stokes_around_sphere_explicite(pos[0], pos[2], 1),
            rtol=1e-05,
            atol=1e-08,
        ), "test if function works for np and jnp"


def test_no_slip():
    angles = itertools.product(np.linspace(0, 2 * np.pi, 10), np.linspace(0, np.pi, 10))
    sphere_positions = np.array(
        [
            [np.cos(phi) * np.cos(theta), np.sin(phi) * np.cos(theta), np.sin(theta)]
            for phi, theta in angles
        ]
    )
    for pos in sphere_positions:
        assert np.allclose(
            sf.stokes_around_sphere_np(pos, 1), np.array([0, 0, 0])
        ), "test if velocity on surface of sphere is 0"


def test_far_field():
    angles = itertools.product(np.linspace(0, 2 * np.pi, 10), np.linspace(0, np.pi, 10))
    far_radius = 10**8
    far_positions = far_radius * np.array(
        [
            [np.cos(phi) * np.cos(theta), np.sin(phi) * np.cos(theta), np.sin(theta)]
            for phi, theta in angles
        ]
    )
    for pos in far_positions:
        assert np.allclose(
            sf.stokes_around_sphere_np(pos, 1),
            np.array([0, 0, 1]),
            rtol=1e-05,
            atol=1e-08,
        ), "test if velocity far away is [0,0,1]"


def test_cylindrical_symmetry():
    phis = np.linspace(0, 2 * np.pi, 10)
    positions = np.array([[np.cos(phi), np.sin(phi), 1] for phi in phis])
    testing_value = sf.stokes_around_sphere_np(positions[0], 1)[2]
    for pos in positions[1:]:
        assert np.allclose(
            sf.stokes_around_sphere_np(pos, 1)[2], testing_value, rtol=1e-05, atol=1e-08
        ), "test if velocity is independent on phi"
