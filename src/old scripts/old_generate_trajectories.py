import pychastic
import jax.numpy as jnp
import numpy as np
import time


def generate_trajectories(
    peclet=1,
    small_r=0.05,
    trials=100,
    r_mesh=0.1,
    floor_r=None,
    floor_h=5,
    t_max=None,
):
    """
    Generate trajectories of particles in a simulation.

    Parameters
    ----------
    peclet : float, optional
        Peclet number defined as R u / D.

    small_r : float, optional
        Radius of the smal ball.

    trials : int, optional
        Number of trajectories per initial condition.

    floor_r : int, optional
        Radius of the floor of trial cyllinder.

    floor_h : int, optional
        Vertical distance from initial condition to big ball centre.

    t_max : float, optional
        Max simulation time. (Remember dimensionless!)

    Returns
    -------
    np.array
        2 by `total_trials` array containing initial radial distance in
        first coord and outcome 1 - ball hit, 0 - ball missed.

    """

    if floor_r == None:
        floor_r = max((6 * floor_h) / (3 + 2 * peclet), 1)

    initial = construct_initial_condition(floor_r, floor_h, r_mesh, trials)

    collision_data = simulate_until_collides(
        drift=stokes_around_unit_sphere,
        noise=diffusion_function(peclet=peclet),
        initial=initial,
        small_r=small_r,
        floor_h=floor_h,
        t_max=t_max,
    )

    ret = np.vstack((initial[:, 0], collision_data["ball_hit"])).T

    return ret[collision_data["something_hit"]]


def calculate_probability(
    x_position,
    trials,
    peclet,
    small_r,
    floor_h=5,
    t_max=None,
):
    """
    Generate trajectories of particles in a simulation at a single position and calculate hitting propability

    Parameters
    ----------

    x_position: float
        position to calculate probability

    trials : int
        Number of trajectories to calculate pobability.

    peclet : float
        Peclet number defined as R u / D.

    floor_r : int, optional
        Radius of the floor of trial cyllinder.

    floor_h : int, optional
        Vertical distance from initial condition to big ball centre.

    t_max : float, optional
        Max simulation time. (Remember dimensionless!)

    Returns
    -------
    propability of hitting , amout of sucsesfull runs

    """

    initial = construct_initial_trials_at_x(floor_h = floor_h, x_position = x_position, trials = trials)

    collision_data = simulate_until_collides(
        drift=stokes_around_unit_sphere,
        noise=diffusion_function(peclet=peclet),
        initial=initial,
        small_r=small_r,
        floor_h=floor_h,
        t_max=t_max,
    )

    # print(collision_data)

    return (
        np.sum(collision_data["ball_hit"]) / np.sum(collision_data["something_hit"]),
        np.sum(collision_data["something_hit"]),
    )


def diffusion_function(peclet):
    def diffusion(q):
        return ((1 / peclet) ** 0.5) * jnp.eye(3)

    return diffusion


def stokes_around_unit_sphere(q):
    """
    Given location of the tracer find drift velocity -- Stokes flow around sphere of size big_r (stationary) and ambient flow u_inf.

    Locaiton is measured from the centre of the sphere.

    Compare:
    https://en.wikipedia.org/wiki/Stokes%27_law#Transversal_flow_around_a_sphere
    """

    big_r = 1
    u_inf = jnp.array([0, 0, 1])

    abs_x = jnp.sum(q * q) ** 0.5

    xx_scale = (3 * big_r**3) / (4 * abs_x**5) - (3 * big_r) / (4 * abs_x**3)
    id_scale = -(big_r**3) / (4 * abs_x**3) - (3 * big_r) / (4 * abs_x) + 1

    xx_tensor = q[:, jnp.newaxis] * q[jnp.newaxis, :]
    id_tensor = jnp.eye(3)

    return (xx_scale * xx_tensor + id_scale * id_tensor) @ u_inf


def construct_initial_condition(floor_r, floor_h, r_mesh, trials):
    # TODO: RW 2024-01-27
    # TODO: Uniform along the radius is a terible strategy

    initial_x = np.tile(np.arange(0, floor_r, r_mesh), trials)

    initial_y = np.zeros_like(initial_x)
    initial_z = np.zeros_like(initial_x) - floor_h
    return np.vstack((initial_x, initial_y, initial_z)).T


def construct_initial_trials_at_x(floor_h, x_position, trials):
    # TODO: RW 2024-01-27
    # TODO: Uniform along the radius is a terible strategy

    initial_x = x_position * np.ones(trials)

    initial_y = np.zeros_like(initial_x)
    initial_z = np.zeros_like(initial_x) - floor_h
    return np.vstack((initial_x, initial_y, initial_z)).T


def simulate_until_collides(drift, noise, initial, small_r, floor_h, t_max=None):
    """
    Simulate trajectories until they collide with roof or ball
    """

    if t_max == None:
        t_max = 10 * floor_h

    # TODO: RW 2024-01-27
    # TODO: Better implementation possible: compute part of trajectory
    # TODO: and drop finalized trajectories as soon as possible.
    # TODO: Some initial conditions will typically take longer.

    problem = pychastic.sde_problem.SDEProblem(
        drift,
        noise,
        x0=initial,
        tmax=t_max,
    )

    solver = pychastic.sde_solver.SDESolver(dt=0.01)
    solution = solver.solve_many(problem, None, progress_bar=None)
    trajectories = solution["solution_values"]

    ball_distances = jnp.linalg.norm(trajectories, axis=2)
    ball_hit = jnp.min(ball_distances, axis=1) < (1 + small_r)

    roof_hit = jnp.max(trajectories[:, :, -1], axis=1) > floor_h

    something_hit = np.logical_or(ball_hit, roof_hit)

    return {
        "ball_hit": ball_hit,
        "roof_hit": roof_hit,
        "something_hit": something_hit,
        "trajectories": trajectories,  # for debug only
    }


if __name__ == "__main__":
    generate_trajectories()
