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
    >>> position = jnp.array([1,1,1])
    >>> stokes_around_sphere(position, 0.7)
    [-0.0845337019138477, -0.0845337019138477, 0.595854811567262]
    """

    big_r = ball_radius
    u_inf = np.array([0, 0, 1])

    abs_x = (q[0] ** 2 + q[1] ** 2 + q[2] ** 2) ** 0.5

    xx_scale = (3 * big_r**3) / (4 * abs_x**5) - (3 * big_r) / (4 * abs_x**3)
    id_scale = -(big_r**3) / (4 * abs_x**3) - (3 * big_r) / (4 * abs_x) + 1

    xx_tensor = q.reshape(-1, 1) * q.reshape(1, -1)
    id_tensor = np.eye(3)

    return (xx_scale * xx_tensor + id_scale * id_tensor) @ u_inf


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
    >>> position = np.array([1,1,1])
    >>> stokes_around_sphere(position, 0.7)
    [-0.0845337019138477, -0.0845337019138477, 0.595854811567262]
    """

    big_r = ball_radius
    u_inf = jnp.array([0, 0, 1])

    abs_x = (q[0] ** 2 + q[1] ** 2 + q[2] ** 2) ** 0.5

    xx_scale = (3 * big_r**3) / (4 * abs_x**5) - (3 * big_r) / (4 * abs_x**3)
    id_scale = -(big_r**3) / (4 * abs_x**3) - (3 * big_r) / (4 * abs_x) + 1

    xx_tensor = q.reshape(-1, 1) * q.reshape(1, -1)
    id_tensor = jnp.eye(3)

    return (xx_scale * xx_tensor + id_scale * id_tensor) @ u_inf


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
    >>> psi(1, 1, 1)
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

def streamline_radius(z, ball_radius, r_start = 1):
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
    >>> streamline_radius(5, 0.8)
    0.2710224007612492
    """

    R = ball_radius

    #find the difference between streamfunction at [r_start, 0, 0] and [r, 0, z]
    def difference(r):
        return psi(r, z, R) - psi(r_start, 0, R)

    #initial guess is the distance between r_start and ball_radius
    r_guess = r_start - ball_radius

    # Numerical solver for r
    r_solution = fsolve(difference, r_guess)
    return r_solution[0]