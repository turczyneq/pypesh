import pygmsh
import numpy as np
from skfem import *
from skfem.models.poisson import laplace
from skfem.visuals.matplotlib import plot, show
import matplotlib.pyplot as plt

# Define the Peclet number
peclet = 100

floor_depth = 5.0
floor_width = 5.0
ball_size = 1.0
ball_segments = 100
mesh_size = 0.1

box_points = np.array(
    [
        [0, -floor_depth],
        [floor_width, -floor_depth],
        [floor_width, floor_depth],
        [0, floor_depth],
    ]
)
phi_values = np.linspace(0, np.pi, ball_segments)
ball_points = ball_size * np.column_stack((np.sin(phi_values), np.cos(phi_values)))
mesh_boundary = np.vstack((box_points, ball_points))

# Create the geometry and mesh using pygmsh
with pygmsh.geo.Geometry() as geom:
    poly = geom.add_polygon(
        mesh_boundary,
        mesh_size=mesh_size,  # Mesh size
    )

    raw_mesh = geom.generate_mesh()

# Convert the mesh to a skfem MeshTri object and define boundaries
mesh = MeshTri(
    raw_mesh.points[:, :2].T, raw_mesh.cells_dict["triangle"].T
).with_boundaries(
    {
        "left": lambda x: np.isclose(x[0], 0),  # Left boundary condition
        "bottom": lambda x: np.isclose(x[1], -floor_depth),  # Bottom boundary condition
        "ball": lambda x: x[0] ** 2 + x[1] ** 2 < 1.1 * ball_size**2,
    }
)


# Define the basis for the finite element method
basis = Basis(mesh, ElementTriP1())

left_nodes = mesh.p[:, basis.get_dofs("left").flatten()]
bottom_nodes = mesh.p[:, basis.get_dofs("bottom").flatten()]
ball_nodes = mesh.p[:, basis.get_dofs("ball").flatten()]

# Draw the mesh
mesh.draw()
plt.plot(left_nodes[0], left_nodes[1], "x")
plt.plot(bottom_nodes[0], bottom_nodes[1], "x")
plt.plot(ball_nodes[0], ball_nodes[1], "o")
show()


@BilinearForm
def advection(u, v, w):
    """Advection bilinear form."""
    from skfem.helpers import grad

    U = 1 # velocity scale
    a = 1 # ball size
    r, z = w.x
    
    # r_squared_plus_z_squared = r**2 + z**2
    
    # # Calculate the common term (r^2 + z^2)^(3/2) and (r^2 + z^2)^(5/2)
    # r_squared_plus_z_squared_3_2 = np.power(r_squared_plus_z_squared, 3/2)
    # r_squared_plus_z_squared_5_2 = np.power(r_squared_plus_z_squared, 5/2)
    
    # # Calculate the velocity components v_r and v_z
    # v_r = (U * a**3 * r * z) / r_squared_plus_z_squared_5_2 - (U * a * r * z) / (2 * r_squared_plus_z_squared_3_2)
    # v_z = (2 * U * a**3 * z**2) / r_squared_plus_z_squared_5_2 - (U * a * z**2) / r_squared_plus_z_squared_3_2 + (U * a * r**2) / (2 * r_squared_plus_z_squared_3_2)
    v_r = (-3 * z * r * (-1 + z**2 + r**2)) / (4. * (z**2 + r**2)**2.5)
    v_z = -(2 * z**2 - 6 * z**4 - r**2 - 9 * z**2 * r**2 - 3 * r**4 + 4 * (z**2 + r**2)**2.5) / (4. * (z**2 + r**2)**2.5)
    
    advection_velocity_x = v_r
    advection_velocity_y = -v_z
    return v * advection_velocity_x * grad(u)[0] + v * advection_velocity_y * grad(u)[1]


# Identify the interior degrees of freedom
interior = basis.complement_dofs(basis.get_dofs({"bottom", "ball"}))

# Assemble the system matrix
A = asm(laplace, basis) + peclet * asm(advection, basis)

# Initialize the solution vector with boundary conditions
u = basis.zeros()

# u[basis.get_dofs("left")] = 1.0  # Left boundary condition
u[basis.get_dofs("bottom")] = 1.0  # Bottom boundary condition
u[basis.get_dofs("ball")] = 0.0  # Bottom boundary condition

# Solve the system
u = solve(*condense(A, x=u, I=interior))

if __name__ == "__main__":
    # Plot the solution

    # mesh.draw()

    plt.tripcolor(mesh.p[0], mesh.p[1], mesh.t.T, u, shading="gouraud", cmap="viridis")
    plt.colorbar()
    plt.clim(vmin=0, vmax=1)  # Set color range
    plt.gca().set_aspect('equal', 'box')  # 'equal' ensures that one unit in x is equal to one unit in y
    plt.tight_layout()

    plt.show()

    # plot(basis, u, shading="gouraud", colorbar=True, cmap="viridis")
    # show()
