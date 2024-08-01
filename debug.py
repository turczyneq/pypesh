# Copyright (C) 2024 Radost Waszkiewicz and Jan Turczynowicz
# This software is published under BSD-3-clause license

# Folowing code solves advection-diffusion problem for
# temperature distribution around a cold sphere in warm
# liquid. Liquid flow is modelled using Stokes flow field.
# Thermal difusivity to advection ratio is controlled by
# Peclet number.

import numpy as np
from tqdm import tqdm

from pathlib import Path
from skfem import (
    MeshTri,
    Basis,
    ElementTriP1,
    BilinearForm,
    LinearForm,
    FacetBasis,
    Functional,
)
from skfem import asm, solve, condense
from skfem.helpers import grad, dot

# Define the Peclet number
peclet = 30

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=("Advection diffusion in stokes flow"))
    parser.add_argument(
        "--peclet",
        type=float,
        default=30,
        help="value of Peclet number",
    )
    parser.add_argument(
        "--mesh",
        type=str,
        default="cylinder_stokes_fine.msh",
        help="select if specyfic mesh is downloaded",
    )

    parser.add_argument(
        "--propab_dist",
        type=str,
        default="test.txt",
        help="select where to sve propabiity distribution",
    )

    parser.add_argument(
        "--ball",
        type=float,
        default=1,
        help="radius of the ball in absorbing radius",
    )

    parser.add_argument(
        "--dist2d",
        type=str,
        default="no",
        help="where to export 2d contoruplot",
    )

    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    peclet = args.peclet

#
# Code for creating mesh which we load
#

# floor_depth = 5.0
# floor_width = 5.0
# ball_size = 1.0
# ball_segments = 100
# mesh_size = 0.01
# far_mesh = 0.5

# box_points = [
#         ([0, -floor_depth], far_mesh),
#         ([floor_width, -floor_depth], far_mesh),
#         ([floor_width, floor_depth], far_mesh),
#         ([0, floor_depth], mesh_size),
#     ]

# phi_values = np.linspace(0, np.pi, ball_segments)
# ball_points = ball_size * np.column_stack((np.sin(phi_values), np.cos(phi_values)))
# mesh_boundary = np.vstack((
#     np.array([p for p,s  in box_points])
#     , ball_points))

# # Create the geometry and mesh using pygmsh
# with pygmsh.geo.Geometry() as geom:
#     poly = geom.add_polygon(
#         mesh_boundary,
#         mesh_size=([s for p,s in box_points]) + ([mesh_size] * len(ball_points)),
#     )

#     raw_mesh = geom.generate_mesh()

# # Convert the mesh to a skfem MeshTri object and define boundaries
# mesh = MeshTri(
#     raw_mesh.points[:, :2].T, raw_mesh.cells_dict["triangle"].T
# ).with_boundaries(
#     {s
#         "left": lambda x: np.isclose(x[0], 0),  # Left boundary condition
#         "bottom": lambda x: np.isclose(x[1], -floor_depth),  # Bottom boundary condition
#         "ball": lambda x: x[0] ** 2 + x[1] ** 2 < 1.1 * ball_size**2,
#     }
# )

# mesh = MeshTri.load(Path(__file__).parent / "meshes" / "cylinder_stokes.msh")
print(f"loading mesh from {Path(__file__).parent}/meshes/{args.mesh}")
mesh = MeshTri.load(Path(__file__).parent / "meshes" / args.mesh)
print("mesh loaded")
# Define the basis for the finite element method
basis = Basis(mesh, ElementTriP1())


@BilinearForm
def advection(k, l, m):
    """Advection bilinear form."""

    # Coordinate fields
    r, z = m.x

    u = 1  # velocity scale
    a = args.ball  # ball size

    # Stokes flow around a sphere of size `a`
    w = r**2 + z**2
    v_r = ((3 * a * r * z * u) / (4 * w**0.5)) * ((a / w) ** 2 - (1 / w))
    v_z = u + ((3 * a * u) / (4 * w**0.5)) * (
        (2 * a**2 + 3 * r**2) / (3 * w) - ((a * r) / w) ** 2 - 2
    )

    return (l * v_r * grad(k)[0] + l * v_z * grad(k)[1]) * 2 * np.pi * r


@BilinearForm
def claplace(u, v, m):
    """Laplace operator in cylindrical coordinates."""
    r, z = m.x
    return dot(grad(u), grad(v)) * 2 * np.pi * r

# Assemble the system matrix
A = asm(claplace, basis) + peclet * asm(advection, basis)

# Identify the interior degrees of freedom
interior = basis.complement_dofs(basis.get_dofs({"bottom", "ball"}))

# Boundary condition
u = basis.zeros()
u[basis.get_dofs("bottom")] = 1.0
u[basis.get_dofs("ball")] = 0.0

print("solving problem")
u = solve(*condense(A, x=u, I=interior))
print("solution calculated")


'''
Exporting porpability distribution on heigt 10
REQUIRED: to export results it is required to support --dist2d
'''
# pos = np.transpose(mesh.p)
# pos_val = np.zeros((len(pos),3))
# print("loading solution")
# for n in tqdm(range(len(pos))):
#     pos_val[n] = np.array([[pos[n,0],pos[n,1],1-u[n]]])
# output_file = "numerical_results/dist_10/"+args.dist2d
# mask = (10 <= pos_val[:, 1]) & (pos_val[:, 1] < 10.05)
# toexp = pos_val[mask]
# with open(output_file, 'w') as f:
#     for x, y, z in toexp:
#         f.write(f"{x}\t{y}\t{z}\n")


''' 
Exporting porpability distribution on different heights
REQUIRED: to calculate on mesh: ./tests/depth_size200.msh (otherwise no points for height e.g. 199)
'''
# pos = np.transpose(mesh.p)
# pos_val = np.zeros((len(pos),3))
# print("loading solution")
# for n in tqdm(range(len(pos))):
#     pos_val[n] = np.array([[pos[n,0],pos[n,1],1-u[n]]])

# print("decreasing amount of data")
# mask = (pos_val[:, 1] > 1)
# pos_val = pos_val[mask]

# print("exporting different heights")
# for heigh in tqdm([2, 5, 10, 20, 50, 80, 100, 199]):
#     output_file = f"numerical_results/dist_chage_heights/on_height_{heigh}.txt"
#     mask = (heigh <= pos_val[:, 1]) & (pos_val[:, 1] < heigh + 0.02)
#     toexp = pos_val[mask]
#     with open(output_file, 'w') as f:
#         for x, y, z in toexp:
#             f.write(f"{x}\t{y}\t{z}\n")


if __name__ == "__main__" and not args.quiet:
    import matplotlib.pyplot as plt

    mesh.draw(boundaries=True).show()

    # dofs = basis.get_dofs("top")
    # vals = [[x, 1-el] for x, el in zip(mesh.p[0, dofs.nodal["u"]],u[basis.get_dofs("top")])]
    # vals = sorted(vals, key=lambda pair: pair[0])
    # xargs, yargs = zip(*vals)
    # print((yargs[1]-yargs[0])/(xargs[1]-xargs[0]))

    # plt.figure(figsize=(10, 6))
    # plt.plot(xargs, yargs, color='b', marker = 'o')

    # # Add labels and title
    # plt.xlabel('radius')
    # plt.ylabel('propability')

    # # Show the plot
    # plt.show()

    basis.plot(u, shading="gouraud", cmap="viridis").show()

fbasis = FacetBasis(mesh, ElementTriP1(), facets="top")

'''
Calculating flux on "top"
'''

@Functional
def intercepted(m):
    r, z = m.x
    phi = m["u"]

    return (1 - phi) * 2 * np.pi * r


result = asm(intercepted, fbasis, u=u)

print(f"pe,{peclet},intercepted,{result},")