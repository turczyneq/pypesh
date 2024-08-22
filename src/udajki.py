import pygmsh
import numpy as np

from pathlib import Path
from skfem import (
    MeshTri,
    
)

def binary_search(fun, target, left, right, tol=1e-15, max_iter=10000):
    """
    Perform a binary search on a monotonically increasing function to find the input value that
    makes the function output close to the target value.
    
    Parameters:
    - fun: A monotonically increasing function.
    - target: The target output value to search for.
    - left: The lower bound of the input interval.
    - right: The upper bound of the input interval.
    - tol: Tolerance for convergence. The search stops when the function value is within this
      tolerance of the target.
    - max_iter: Maximum number of iterations to prevent infinite loops.
    
    Returns:
    - A value x in the interval [left, right] where fun(x) is approximately equal to target,
      or None if the target is not reachable within the given interval.
    """
    iteration = 0
    while left <= right and iteration < max_iter:
        mid = left + (right - left) / 2
        mid_val = fun(mid)

        # Check if the function value is within the specified tolerance
        if np.abs(mid_val - target) <= tol:
            return mid

        # If the middle value is less than the target, move to the right half
        if mid_val < target:
            left = mid - tol
        else:
            right = mid + tol

        iteration += 1

    # If no value is found within the tolerance after max_iter iterations, return None
    return None


def psi(r, z, ball_radius):
    u = 1
    R = ball_radius
    return (1/2) * u * r**2 * (1 - (3/2) * R / (r**2 + z**2)**0.5 + (1/2) * (R / (r**2 + z**2)**0.5)**3)

def streamline(r, z, ball_radius):
    '''
    Value of streamline minus stramline at z = 0, r = 1 - when particle touches the ball
        - lower then zero = inside
        - greater then zero = outside
    '''
    u = 1
    return psi(r,z,ball_radius) - psi(1,0,ball_radius)

def velocities(r, z, ball_radius):

    u = 1  # velocity scale
    a = ball_radius  # ball size

    # Stokes flow around a sphere of size `a`
    w = r**2 + z**2
    v_r = ((3 * a * r * z * u) / (4 * w**0.5)) * ((a / w) ** 2 - (1 / w))
    v_z = u + ((3 * a * u) / (4 * w**0.5)) * (
        (2 * a**2 + 3 * r**2) / (3 * w) - ((a * r) / w) ** 2 - 2
    )
    return v_r, v_z

def generate_mesh(ball_radius, mesh = 0.01, far_mesh = 0.5, cell_size = 1.0, width = 10.0, ceiling = 10.0, floor = 10.0, filename = "test.msh", quiet = "False"):
    "Advection diffusion in stokes flow"


    floor_depth = floor*cell_size
    ceiling_depth = ceiling*cell_size
    floor_width = width*cell_size
    ball_size = 1.0
    ball_segments = int(1/mesh)
    mesh_size = mesh
    far_mesh = far_mesh

    box_points = [
            ([0, -floor_depth], far_mesh),
            ([floor_width, -floor_depth], far_mesh),
            ([floor_width, ceiling_depth], far_mesh),
            ([floor_width/5, ceiling_depth], mesh_size),
            ([0, ceiling_depth], mesh_size),
        ]

    phi_values = np.linspace(0, np.pi, ball_segments)
    ball_points = ball_size * np.column_stack((np.sin(phi_values), np.cos(phi_values)))
    mesh_boundary = np.vstack((
        np.array([p for p,s  in box_points])
        , ball_points))

    # Create the geometry and mesh using pygmsh
    with pygmsh.occ.Geometry() as geom:
        poly = geom.add_polygon(
            mesh_boundary,
            mesh_size=([s for p,s in box_points]) + ([mesh_size] * len(ball_points)),
        )
        inside = geom.add_point(
            [7,-7],
            mesh_size = 0.1
        )
        geom.boolean_fragments(inside, poly)

        raw_mesh = geom.generate_mesh(dim=2)


    ball_radius = ball_radius

    # Convert the mesh to a skfem MeshTri object and define boundaries
    mesh = MeshTri(
        raw_mesh.points[:, :2].T, raw_mesh.cells_dict["triangle"].T
    )

    mesh = mesh.with_boundaries(
        boundaries = {
            "left": lambda x: np.isclose(x[0], 0),  # Left boundary condition
            "right": lambda x: np.isclose(x[0], floor_width),  # Right boundary condition
            "top": lambda x: np.isclose(x[1], ceiling_depth),  
            "bottom": lambda x: np.isclose(x[1], -floor_depth), 
            "ball": lambda x: x[0] ** 2 + x[1] ** 2 < 1.01 * ball_size**2,
        }
    )

    # mesh = mesh.with_subdomains(
    #     {
    #         "internal": lambda x: streamline(x[0],x[1],ball_radius) < 0,
    #         "external": lambda x: streamline(x[0],x[1],ball_radius) > 0,
    #     }
    # )

    # mesh.save("meshes/subdomains/"+filename)

    if not quiet:

        mesh.draw(boundaries=True).show()
    
    return mesh