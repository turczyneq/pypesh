import pypesh.stokes_flow as sf
import pypesh.analytic as analytic
import jax.numpy as jnp
import numpy as np
import pychastic
from scipy.integrate import quad


def construct_initial_trials_at_x(x_position, floor_h, trials):
    """
    Constructs initial conditions for slected x position at the bottom

    Parameters
    ----------
    floor_h : float
        Initial depth for simulation

    x_postition : float
        Position to generate initial conditions

    trials : int
        Number of trajectories per initial condition.

    Returns
    --------
    jnp.array
        Array [x_position, 0, -floor_h] trials times

    Example
    -------
    >>> import pypesh.trajectories as traj
    >>> traj.construct_initial_trials_at_x(2, 5, 3)
    Array([[ 2.,  0., -5.],
        [ 2.,  0., -5.],
        [ 2.,  0., -5.]], dtype=float32
    """

    initial_x = x_position * np.ones(trials)

    initial_y = np.zeros_like(initial_x)
    initial_z = np.zeros_like(initial_x) - floor_h
    return np.vstack((initial_x, initial_y, initial_z)).T


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
    callable
        jnp.array, Identity matrix times drift coeficient, dimensionalized sqrt(2 D)*dW, where D - diffusion, dW - wiener process

    Example
    -------
    >>> import pypesh.trajectories as traj
    >>> fun = traj.diffusion_function(100)
    >>> fun([0,0,1])
    Array([[0.14142136, 0.        , 0.        ],
        [0.        , 0.14142136, 0.        ],
        [0.        , 0.        , 0.14142136]], dtype=float32)
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
    >>> import pypesh.trajectories as traj
    >>> import pypesh.stokes_flow as sf
    >>> def drift(q):
    ...     return sf.stokes_around_sphere_jnp(q, 0.9)
    >>> noise = traj.diffusion_function(100)
    >>> traj.simulate_trajectory(drift, noise, init, 20)
    {'ball_hit': Array([ True, False, False], dtype=bool), 'roof_hit': Array([False,  True,  True], dtype=bool), 'something_hit': Array([ True,  True,  True], dtype=bool)}
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

    peclet : float
        Peclet number defined as R u / D.

    ball_radius : float
        Radius of the big ball.

    trials : int, optional
        Number of trajectories.

    floor_h : int, optional
        Initial depth for simulation

    Returns
    -------
    float
        value of propability

    Example
    --------
    >>> import pypesh.trajectories as traj
    >>> traj.hitting_propability_at_x(0.1, 10**4, 0.9)
    0.91
    """

    def drift(q):
        return sf.stokes_around_sphere_jnp(q, ball_radius)

    # begin with short time and test the outcome
    t_max = floor_h / 2

    ratio = 1
    initial = construct_initial_trials_at_x(x_position, floor_h, 100)
    while ratio > 0.01:
        """
        loops increasing time until almost all of particles hit either ball or roof
        """
        t_max = t_max * 2

        collision_data = simulate_trajectory(
            drift=drift,
            noise=diffusion_function(peclet=peclet),
            initial=initial,
            t_max=t_max,
        )
        ratio = (100 - sum(collision_data["something_hit"])) / 100

    initial = construct_initial_trials_at_x(x_position, floor_h, trials)

    collision_data = simulate_trajectory(
        drift=drift,
        noise=diffusion_function(peclet=peclet),
        initial=initial,
        t_max=t_max,
    )

    trajectory_outcome = [int(val) for val in collision_data["ball_hit"]]
    propab = sum(trajectory_outcome) / trials

    return propab


def weighted_trapezoidal(function, ball_radius, z):
    """
    Calculate integral of function with weight, expresion to integrate is: 2*pi*r*vz(r)*function(r). On each step assume function(r) is a*r + b and then integrate.

    Parameters
    ----------
    function : dict
        Keys are xargs and values are yargs of function to integrate

    ball_radius : float
        Radius of the big ball.

    z : float
        Height where integration is taking place

    Returns
    -------
    float
        value of integral

    Example
    --------
    >>> import pypesh.trajectories as traj
    >>> import numpy as np
    >>> dict = {x: np.cos(x) for x in np.linspace(0, np.pi/2, 10)}
    >>> traj.weighted_trapezoidal(dict, 1, 5)
    2.5493506321093182
    """

    def vz(x):
        velocities = sf.stokes_around_sphere_np(np.array([x, 0, z]), ball_radius)
        return velocities[2]

    def weight(x):
        return 2 * np.pi * x * vz(x)

    xargs = list(function.keys())

    integral, error = quad(weight, 0, xargs[0])
    for i in range(len(xargs) - 1):
        x0 = xargs[i]
        x1 = xargs[i + 1]
        a = (function[x1] - function[x0]) / (x1 - x0)
        b = (function[x0] * x1 - function[x1] * x0) / (x1 - x0)

        def function_to_integrate(x):
            return weight(x) * (a * x + b)

        element, error = quad(function_to_integrate, x0, x1)

        integral = integral + element

    return integral


def sherwood_trajectories(
    peclet,
    ball_radius,
    mesh_out=4,
    mesh_jump=6,
    trials=10**2,
    floor_h=5,
    spread=4,
):
    """
    Calculates the distribution of probability of hitting as a function of radius, at depth floor_h. Addaptive sampling is implemented to ensure effective calculation. Position of greatest slope is assumed to be at streamline that pass position [1, 0, 0], then spread is scaled as sqrt(1/peclet) \sim sqrt(D). Then integrates with weigth and finds the 

    Parameters
    ----------
    peclet : float, optional
        Peclet number defined as R u / D.

    ball_radius : float
        Radius of the big ball.

    mesh_out : int, optional
        Amount of samples outside the region of highest slope

    mesh_jump : int, optional
        Amount of samples in the region of highest slope

    trials : int, optional
        Number of trajectories per position, uncertainty of propability estimation is sqrt(trials)/trials.

    floor_h : int, optional
        Initial depth for simulation

    spread: int, optional
        How far in sqrt(1/peclet), mesh_out will reach

    Returns
    -------
    float
        Estimation of resulting sherwood number (flux/advective_flux)

    list
        Radius samples

    list
        Probability values ordered as radius samples


    Example
    --------
    >>> traj.distribution(10**6, 0.9)
    (18557.92284339992, [0.10064564758776434, 0.11064564758776434, 0.12064564758776435, 0.13064564758776434, 0.13464564758776434, 0.13864564758776435, 0.14264564758776435, 0.14664564758776436, 0.15064564758776436, 0.16064564758776437, 0.17064564758776435, 0.18064564758776436], [1.0, 1.0, 1.0, 0.99, 0.94, 0.76, 0.39, 0.19, 0.07, 0.0, 0.0, 0.0])
    """

    # define the spread of testing range
    r_syf = sf.streamline_radius(floor_h, ball_radius)

    dispersion = 10 * (1 / peclet) ** (1 / 2)

    # generate the mesh to calculate the probability distribution
    if r_syf - dispersion > 0:
        x_probs = list(
            np.linspace(max(r_syf - spread * dispersion, 0), r_syf - dispersion, mesh_out)
        )
    else:
        x_probs = [0]
    x_probs = x_probs + list(np.linspace(max(r_syf - dispersion, 0), r_syf + dispersion, mesh_jump))
    x_probs = x_probs + list(np.linspace(r_syf + dispersion, r_syf + spread * dispersion, mesh_out))

    def fun(x):
        return hitting_propability_at_x(x, peclet, ball_radius, trials=trials)

    # sol_dict = {x: fun(x) for x in tqdm.tqdm(x_probs)}
    sol_dict = {x: fun(x) for x in x_probs}

    integral = weighted_trapezoidal(sol_dict, ball_radius, floor_h)

    return (
        analytic.sherwood_from_flux(integral, peclet),
        list(sol_dict.keys()),
        list(sol_dict.values()),
    )