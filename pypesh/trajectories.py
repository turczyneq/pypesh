import pypesh.stokes_flow as sf
import pypesh.analytic as analytic
import jax.numpy as jnp
import jax
import numpy as np
import pychastic
from scipy.integrate import quad


def _construct_initial_trials_at_x(x_position, floor_h, trials):
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

    initial_x = x_position * jnp.ones(trials)

    initial_y = jnp.zeros_like(initial_x)
    initial_z = jnp.zeros_like(initial_x) - floor_h
    return jnp.vstack((initial_x, initial_y, initial_z)).T


def _diffusion_function(peclet):
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


def simulate_trajectory(drift, noise, initial, t_max, whole_trajectory=False):
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

    whole_trajectory : boole, optional
        Deafult False, if True than also returns whole trajectory (Warning expensive in memory)

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

    if whole_trajectory:
        return {
            "ball_hit": ball_hit,
            "roof_hit": roof_hit,
            "something_hit": something_hit,
            "trajectories": trajectories,
        }
    else:
        return {
            "ball_hit": ball_hit,
            "roof_hit": roof_hit,
            "something_hit": something_hit,
        }


def draw_trajectory_at_x(
    x_position,
    peclet,
    ball_radius,
    trials=5,
    floor_h=5,
    t_max=10,
):
    """
    Generate trajectories of particles in a simulation for certain x position amd returns whole trajectory.

    Parameters
    ----------
    x_postition : float
        Radius where probability is evaluated (simulation initiation point)

    peclet : float
        Peclet number defined as R u / D.

    ball_radius : float
        Radius of the big ball.

    trials : int, optional
        Default 5, Number of trajectories.

    floor_h : int, optional
        Default 5, Initial depth for simulation

    t_max : float, optional
        Default 5, time of simulation

    Returns
    -------
    dict
        ``ball_hit`` - jnp.array() of 1 and 0, if 1 trajectory within radius 1, 0 miss
        ``roof_hit`` - jnp.array() of 1 and 0, if 1 trajectory at the end above height 2, 0 miss
        ``something_hit`` - jnp.array() of 1 and 0, union of ``ball_hit`` and ``roof_hit``
        ``trajectories`` - jnp.array() `trials` by `100*t_max` by 3 with `x(t), y(t), z(t)` positions for each trial.

    Example
    --------
    >>> import pypesh.trajectories as traj
    >>> traj.draw_trajectory_at_x(0.1, 1000, 0.9, trials = 1, t_max = .1)
    {'ball_hit': Array([False], dtype=bool), 'roof_hit': Array([False], dtype=bool), 'something_hit': Array([False], dtype=bool), 'trajectories': Array([[[ 1.01513535e-01,  1.41892245e-03, -4.98940468e+00],
            [ 1.03897616e-01,  9.31997492e-04, -4.97806931e+00],
            [ 1.04288198e-01, -1.80142978e-03, -4.97903204e+00],
            [ 9.97019112e-02, -3.12771578e-03, -4.96664095e+00],
            [ 1.02413967e-01, -2.95094796e-03, -4.96111631e+00],
            [ 1.01300426e-01,  6.96127070e-04, -4.94909143e+00],
            [ 1.03424884e-01, -6.30148593e-03, -4.94255972e+00],
            [ 1.03292465e-01, -5.54783177e-03, -4.93507099e+00],
            [ 9.94927734e-02, -4.31211060e-03, -4.92812681e+00],
            [ 9.83759165e-02, -7.30469078e-03, -4.92086792e+00]]],      dtype=float32)}
    >>> traj.draw_trajectory_at_x(0.1, 1000, 0.9, trials = 2, t_max = .05)
    {'ball_hit': Array([False, False], dtype=bool), 'roof_hit': Array([False, False], dtype=bool), 'something_hit': Array([False, False], dtype=bool), 'trajectories': Array([[[ 9.8736316e-02, -7.5448438e-04, -4.9963560e+00],
            [ 9.6689783e-02, -4.3136943e-03, -4.9961877e+00],
            [ 9.7639665e-02, -5.3856885e-03, -4.9852118e+00],
            [ 9.8702230e-02, -7.6030511e-03, -4.9785647e+00],
            [ 9.1684863e-02,  7.3614251e-04, -4.9669151e+00]],

        [[ 1.0136999e-01, -1.1048245e-02, -4.9944925e+00],
            [ 1.0074968e-01, -3.2547791e-03, -4.9900651e+00],
            [ 1.0723757e-01, -6.1367834e-03, -4.9799914e+00],
            [ 1.1393108e-01, -1.7973720e-03, -4.9786887e+00],
            [ 1.1445464e-01, -7.9164971e-03, -4.9739923e+00]]], dtype=float32)}
    """

    def drift(q):
        return sf.stokes_around_sphere_jnp(q, ball_radius)

    initial = _construct_initial_trials_at_x(x_position, floor_h, trials)

    collision_data = simulate_trajectory(
        drift=drift,
        noise=_diffusion_function(peclet=peclet),
        initial=initial,
        t_max=t_max,
        whole_trajectory=True,
    )

    return collision_data


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

    _drift = jax.jit(drift)

    _diffusion_at_peclet = jax.jit(_diffusion_function(peclet=peclet))

    # begin with short time and test the outcome
    # t_max = floor_h / 2

    # ratio = 1
    # initial = _construct_initial_trials_at_x(x_position, floor_h, 100)
    # while ratio > 0.01:
    #     """
    #     loops increasing time until almost all of particles hit either ball or roof
    #     """
    #     t_max = t_max * 2

    #     collision_data = simulate_trajectory(
    #         drift=drift,
    #         noise=_diffusion_function(peclet=peclet),
    #         initial=initial,
    #         t_max=t_max,
    #     )
    #     ratio = (100 - sum(collision_data["something_hit"])) / 100

    t_max = 40.0

    initial = _construct_initial_trials_at_x(x_position, floor_h, trials)

    collision_data = simulate_trajectory(
        drift=_drift,
        noise=_diffusion_at_peclet,
        initial=initial,
        t_max=t_max,
    )

    propab = sum(1.0 * collision_data["ball_hit"]) / trials

    return propab


def _weighted_trapezoidal(function, ball_radius, z):
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
    >>> dictionary = {x: np.cos(x) for x in np.linspace(0, np.pi/2, 10)}
    >>> traj.weighted_trapezoidal(dictionary, 1, 5)
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
    Calculates the distribution of probability of hitting as a function of radius, at depth floor_h. Addaptive sampling is implemented to ensure effective calculation. Position of greatest slope is assumed to be at streamline that pass position [1, 0, 0], then spread is scaled as sqrt(1/peclet) \sim sqrt(D). Then integrates with weigth and finds the Sherwood number

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
    tuple
        float - estimation of resulting sherwood number (flux/advective_flux), list - Radius samples, list - Probability values ordered as radius samples


    Example
    --------
    >>> import pypesh.trajectories as traj
    >>> traj.distribution(10**6, 0.9)
    (18557.92284339992, [0.10064564758776434, 0.11064564758776434, 0.12064564758776435, 0.13064564758776434, 0.13464564758776434, 0.13864564758776435, 0.14264564758776435, 0.14664564758776436, 0.15064564758776436, 0.16064564758776437, 0.17064564758776435, 0.18064564758776436], [1.0, 1.0, 1.0, 0.99, 0.94, 0.76, 0.39, 0.19, 0.07, 0.0, 0.0, 0.0])
    """

    # define the spread of testing range
    r_syf = sf.streamline_radius(floor_h, ball_radius)

    dispersion = 10 * (1 / peclet) ** (1 / 2)

    # generate the mesh to calculate the probability distribution
    if r_syf - dispersion > 0:
        x_probs = list(
            np.linspace(
                max(r_syf - spread * dispersion, 0), r_syf - dispersion, mesh_out
            )
        )
    else:
        x_probs = [0]
    x_probs = x_probs + list(
        np.linspace(max(r_syf - dispersion, 0), r_syf + dispersion, mesh_jump)
    )
    x_probs = x_probs + list(
        np.linspace(r_syf + dispersion, r_syf + spread * dispersion, mesh_out)
    )

    # delete duplicates
    x_probs = list(dict.fromkeys(x_probs))

    def fun(x):
        return hitting_propability_at_x(x, peclet, ball_radius, trials=trials)
    
    _fun = jax.jit(fun)

    # sol_dict = {x: fun(x) for x in tqdm.tqdm(x_probs)}
    sol_dict = {x: _fun(x) for x in x_probs}

    integral = _weighted_trapezoidal(sol_dict, ball_radius, floor_h)

    return (
        analytic.sherwood_from_flux(integral, peclet),
        list(sol_dict.keys()),
        list(sol_dict.values()),
    )
