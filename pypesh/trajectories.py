import pypesh.stokes_flow as sf
import jax.numpy as jnp
import pychastic


def construct_initial_trials_at_x(x_position, floor_h, trials):
    """
    Constructs initial conditions for slected x position at the bottom

    Parameters
    ----------
    floor_h : float
        Initial depth for simulation

    x_postition : float
        Position to generate initial conditions

    trials : int, optional
        Number of trajectories per initial condition.

    Returns
    --------
    jnp.array
        Array [x_position, 0, -floor_h] trials times

    Example
    -------
    >>> construct_initial_trials_at_x(2, 5, 3)
    jnp.array([[2, 0, -5], [2, 0, -5], [2, 0, -5]])
    """

    initial_x = x_position * jnp.ones(trials)

    initial_y = jnp.zeros_like(initial_x)
    initial_z = jnp.zeros_like(initial_x) - floor_h
    return jnp.vstack((initial_x, initial_y, initial_z)).T


def diffusion_function(peclet):
    """
    Simple noice operator describing diffusion

    Compare:
    https://en.wikipedia.org/wiki/Fokker%E2%80%93Planck_equation

    Parameters
    ----------
    peclet : float
        dimesnionless number defined as U R / D

    Returns
    --------
    jnp.array
        Array [x_position, 0, -floor_h] trials times

    Example
    -------
    >>> construct_initial_trials_at_x(2, 5, 3)
    jnp.array([[2, 0, -5], [2, 0, -5], [2, 0, -5]])
    """
    def diffusion(q):
        return ((2 / peclet) ** 0.5) * jnp.eye(3)

    return diffusion


def simulate_trajectory(drift, noise, initial, t_max):
    """
    Simulates trajectories starting from initial conditions, affected by noise and moved via drift.

    Parameters
    ----------
    derift : callable
        Function describing drift term of the equation, should return np.array of length ``dimension``.

    noise : callable
        Function describing noise term of the equation, should return np.array of size ``(dimension,noiseterms)``.

    initial : jnp.array
        Array of positions where to start simulating

    t_max : float
        Time of calculation

    Returns
    --------
    dict
        ``ball_hit`` - jnp.array() of 1 and 0, if 1 trajectory within radius 1, 0 miss
        ``roof_hit`` - jnp.array() of 1 and 0, if 1 trajectory at the end above height 2, 0 miss
        ``something_hit`` - jnp.array() of 1 and 0, union of ``ball_hit`` and ``roof_hit``

    Example
    -------
    TODO
    """

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
    ball_hit = jnp.min(ball_distances, axis=1) < 1

    roof_hit = jnp.max(trajectories[:, :, -1], axis=1) > 2

    something_hit = jnp.logical_or(ball_hit, roof_hit)

    return {
        "ball_hit": ball_hit,
        "roof_hit": roof_hit,
        "something_hit": something_hit,
        # "trajectories": trajectories,  # for debug only
    }


def hitting_propability_at_x(
    x_position,
    peclet,
    ball_radius,
    trials=100,
    floor_h=5,
):
    """
    Generate trajectories of particles in a simulation for certain x position. Than calculates the propability of hitting for this position.

    Parameters
    ----------
    x_postition : float
        Radius where probability is evaluated (simulation initiation point)

    peclet : float, optional
        Peclet number defined as R u / D.

    ball_radius : float
        Radius of the big ball.

    trials : int, optional
        Number of trajectories.

    floor_h : int, optional
        Initial depth for simulation

    Returns
    -------
    float - value of propability


    Example
    --------
    >>> hitting_propability_at_x(0.1, 10**4, 0.9)
    TODO
    """

    def drift(q):
        return sf.stokes_around_sphere_jnp(q, ball_radius)

    # begin with short time and test the outcome
    t_max = floor_h / 2

    ratio = 1
    initial = construct_initial_trials_at_x(floor_h, x_position, 100)
    while ratio > 0.01:
        """
        loops increasing time until almost all of particles hit either ball or roof
        """
        t_max = t_max * 2

        collision_data = simulate_trajectory(
            drift=drift,
            noise=diffusion_function(peclet=peclet),
            initial=initial,
            floor_h=floor_h,
            t_max=t_max,
        )
        ratio = (100 - sum(collision_data["something_hit"])) / 100

    initial = construct_initial_trials_at_x(floor_h, x_position, trials)

    def drift(q):
        return stokes_around_sphere(q, ball_radius)

    collision_data = simulate_until_collides(
        drift=drift,
        noise=diffusion_function(peclet=peclet),
        initial=initial,
        floor_h=floor_h,
        t_max=t_max,
    )

    trajectory_outcome = [int(val) for val in collision_data["ball_hit"]]
    propab = sum(trajectory_outcome) / trials

    return propab
