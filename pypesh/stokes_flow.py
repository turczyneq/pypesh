import jax.numpy as jnp
import numpy as np
from scipy.optimize import fsolve


def stokes_around_sphere_jnp(q, ball_radius):
    """
    Given location of the tracer find drift velocity -- Stokes flow around sphere of size big_r (stationary) and ambient flow u_inf = [0,0,1].

    Location is measured from the centre of the sphere.

    Compare:
    https://en.wikipedia.org/wiki/Stokes%27_law#Transversal_flow_around_a_sphere

    Parameters
    ---------
    q: jnp.array
        Lenght 3, cartesian position from centre of the sphere.
    ball_radius: float
        Radius of ball.


    Returns
    --------
    jnp.array
        Flow velocity in cartesian coordinates at q


    Example
    --------
    >>> import jax.numpy as jnp
    >>> import pypesh.stokes_flow as sf
    >>> posjnp = jnp.array([1,1,1])
    >>> sf.stokes_around_sphere_jnp(posjnp, 0.9) 
    Array([-0.0948298, -0.0948298,  0.4803847], dtype=float32, weak_type=True)
    """

    big_r = ball_radius
    u_inf = jnp.array([0, 0, 1])

    abs_x = (q[0] ** 2 + q[1] ** 2 + q[2] ** 2) ** 0.5

    xx_scale = (3 * big_r**3) / (4 * abs_x**5) - (3 * big_r) / (4 * abs_x**3)
    id_scale = -(big_r**3) / (4 * abs_x**3) - (3 * big_r) / (4 * abs_x) + 1

    return xx_scale * q[2] * q + id_scale * u_inf


def stokes_around_sphere_np(q, ball_radius):
    """
    Given location of the tracer find drift velocity -- Stokes flow around sphere of size big_r (stationary) and ambient flow u_inf = [0,0,1].

    Location is measured from the centre of the sphere.

    Compare:
    https://en.wikipedia.org/wiki/Stokes%27_law#Transversal_flow_around_a_sphere

    Parameters
    ---------
    q: np.array
        Lenght 3, cartesian position from centre of the sphere.
    ball_radius: float
        Radius of ball.


    Returns
    --------
    np.array
        Flow velocity in cartesian coordinates at q


    Example
    --------
    >>> import pypesh.stokes_flow as sf
    >>> import numpy as np
    >>> posnp = np.array([1,1,1])
    >>> sf.stokes_around_sphere_np(posnp, 0.9)
    array([-0.09482978, -0.09482978,  0.48038476])
    """

    big_r = ball_radius
    u_inf = np.array([0, 0, 1])

    abs_x = (q[0] ** 2 + q[1] ** 2 + q[2] ** 2) ** 0.5

    xx_scale = (3 * big_r**3) / (4 * abs_x**5) - (3 * big_r) / (4 * abs_x**3)
    id_scale = -(big_r**3) / (4 * abs_x**3) - (3 * big_r) / (4 * abs_x) + 1

    return xx_scale * q[2] * q + id_scale * u_inf


def stokes_around_sphere_explicite(r, z, ball_radius):
    """
    Given location of the tracer find drift velocity -- Stokes flow around sphere of size big_r (stationary) and ambient flow u_inf = [0,0,1].

    Location is measured from the centre of the sphere.

    Compare:
    https://en.wikipedia.org/wiki/Stokes%27_law#Transversal_flow_around_a_sphere

    Parameters
    ---------
    r: any
        Radius

    z: any
        Height

    ball_radius: float
        Radius of ball.


    Returns
    --------
    tuple
        Velocity in rho direction, Velocity in phi direction, Velocity in z direction


    Example
    --------
    >>> import pypesh.stokes_flow as sf
    >>> sf.stokes_around_sphere_explicite(1, 1, 0.7)
    (-0.14013972519640885, 0, 0.4583120114372806)
    """

    u = 1  # velocity scale
    a = ball_radius  # ball size

    w = r**2 + z**2
    v_r = ((3 * a * r * z * u) / (4 * w**0.5)) * ((a / w) ** 2 - (1 / w))
    v_z = u + ((3 * a * u) / (4 * w**0.5)) * (
        (2 * a**2 + 3 * r**2) / (3 * w) - ((a * r) / w) ** 2 - 2
    )

    return v_r, 0, v_z


def psi(r, z, ball_radius):
    """
    Given location of the tracer find streamfunction -- Stokes flow around sphere of size ball_radius (stationary) and ambient flow u_inf = [0,0,1].

    Location is measured from the centre in cylindrical coordinates [r, phi, z], z is parallel to ambient flow.

    Compare:
    https://en.wikipedia.org/wiki/Stokes%27_law#Transversal_flow_around_a_sphere

    Parameters
    ---------
    r: float
        distance from central axis of ball.
    z: float
        height above the plane perpendicular to ambient flow, containing centre of sphere
    ball_radius: float
        Radius of ball.

    Returns
    --------
    float
        value of streamline at [r, 0, z]


    Example
    --------
    >>> import pypesh.stokes_flow as sf
    >>> sf.psi(1, 1, 1)
    0.05805826175840782
    """

    R = ball_radius

    return (
        (1 / 2)
        * r**2
        * (
            1
            - (3 / 2) * R / np.sqrt(r**2 + z**2)
            + (1 / 2) * (R / np.sqrt(r**2 + z**2)) ** 3
        )
    )


def streamline_radius(z, ball_radius, r_start=1):
    """
    Find the radius of streamline at heigh z, that goes through the position [r_start, 0, 0]

    Location is measured from the centre in cylindrical coordinates [r, phi, z], z is parallel to ambient flow.

    Parameters
    ---------
    z: float
        height above the plane perpendicular to ambient flow, containing centre of sphere
    ball_radius: float
        Radius of ball.
    r_start: float
        radius of searched streamline for z = 0

    Returns
    --------
    float
        radius of streamline that pases [r_start, 0, 0]


    Example
    --------
    >>> import pypesh.stokes_flow as sf
    >>> sf.streamline_radius(5, 0.8)
    0.2710224007612492
    """

    R = ball_radius

    # find the difference between streamfunction at [r_start, 0, 0] and [r, 0, z]
    def difference(r):
        return psi(r, z, R) - psi(r_start, 0, R)

    # initial guess is the distance between r_start and ball_radius
    r_guess = r_start - ball_radius

    # Numerical solver for r
    r_solution = fsolve(difference, 1)
    return r_solution[0]
