# Copyright (C) 2024 Radost Waszkiewicz and Jan Turczynowicz
# This software is published under BSD-3-clause license

# Folowing code solves advection-diffusion problem for
# temperature distribution around a cold sphere in warm
# liquid. Liquid flow is modelled using Stokes flow field.
# Thermal difusivity to advection ratio is controlled by
# Peclet number.

import numpy as np

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
import os
import pychast.collision_kernels as coll_ker

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=("Advection diffusion in stokes flow"))
    parser.add_argument(
        "--peclet",
        type=float,
        default=100,
        help="value of Peclet number",
    )
    parser.add_argument(
        "--mesh",
        type=str,
        default="cylinder_stokes_fine.msh",
        help="select if specyfic mesh is downloaded",
    )

    parser.add_argument(
        "--ball",
        type=float,
        default=1,
        help="radius of the ball in absorbing radius",
    )

    parser.add_argument("--showgraph", action="store_false")
    args = parser.parse_args()

    peclet = args.peclet

#
# Code for creating mesh which we load
#

def gen_mesh( mesh = 0.01, far_mesh = 0.5, cell_size = 1, width = 10, ceiling = 10, floor = 10,):
    import pygmsh
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
    with pygmsh.geo.Geometry() as geom:
        poly = geom.add_polygon(
            mesh_boundary,
            mesh_size=([s for p,s in box_points]) + ([mesh_size] * len(ball_points)),
        )

        raw_mesh = geom.generate_mesh()

    # Convert the mesh to a skfem MeshTri object and define boundaries
    mesh = MeshTri(
        raw_mesh.points[:, :2].T, raw_mesh.cells_dict["triangle"].T
    ).with_boundaries(
        {
            "left": lambda x: np.isclose(x[0], 0),  # Left boundary condition
            "right": lambda x: np.isclose(x[0], floor_width),  # Right boundary condition
            "top": lambda x: np.isclose(x[1], ceiling_depth),  
            "bottom": lambda x: np.isclose(x[1], -floor_depth), 
            "ball": lambda x: x[0] ** 2 + x[1] ** 2 < 1.01 * ball_size**2,
        }
    )

    return mesh

mesh_fine_path = Path(__file__).parent.parent / "meshes/cylinder_stokes_fine_turbo.msh"

mesh_default_path = Path(__file__).parent.parent / "meshes/cylinder_stokes_fine.msh"

mesh_wide_path = Path(__file__).parent.parent / "meshes/cylinder_stokes_fine_wide.msh"


if mesh_fine_path.exists():
    mesh_fine = MeshTri.load(mesh_fine_path)
else:
    mesh_fine = gen_mesh(mesh = 0.001)

if mesh_default_path.exists():
    mesh_default = MeshTri.load(mesh_default_path)
else:
    mesh_default = gen_mesh()

if mesh_wide_path.exists():
    mesh_wide = MeshTri.load(mesh_wide_path)
else:
    mesh_wide = gen_mesh(width = 20, mesh = 0.05)

# mesh_default = MeshTri.load(Path(__file__).parent.parent / "meshes/cylinder_stokes_fine.msh")

# mesh_wide = MeshTri.load(Path(__file__).parent.parent / "meshes/cylinder_stokes_fine_wide.msh")

# Define the basis for the finite element method
basis_fine = Basis(mesh_fine, ElementTriP1())

basis_default = Basis(mesh_default, ElementTriP1())

basis_wide = Basis(mesh_wide, ElementTriP1())


def sherwood(peclet, ball_radius):
    
    @BilinearForm
    def advection(k, l, m):
        """Advection bilinear form."""

        # Coordinate fields
        r, z = m.x

        u = 1  # velocity scale
        a = ball_radius  # ball size

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

    if peclet > 50000:
        '''
        For big peclets use finer mesh
        '''
        mesh = mesh_fine
        basis = basis_fine
        #print(f"peclet {peclet}, using finer mesh")

    elif peclet < 5:
        '''
        For small peclets use wider base with bigger mesh
        '''
        mesh = mesh_wide
        basis = basis_wide
        #print(f"peclet {peclet}, using wider mesh")

    else:
        '''
        For regural peclets use default mesh
        '''
        mesh = mesh_default
        basis = basis_default
        #print(f"peclet {peclet}, using default mesh")

    # Assemble the system matrix
    A =  asm(claplace, basis) + peclet * asm(advection, basis)

    # Identify the interior degrees of freedom
    interior = basis.complement_dofs(basis.get_dofs({"bottom", "ball"}))

    # Boundary condition
    u = basis.zeros()
    u[basis.get_dofs("bottom")] = 1.0
    u[basis.get_dofs("ball")] = 0.0

    u = solve(*condense(A, x=u, I=interior))

    pos = np.transpose(mesh.p)
    pos_val = np.zeros((len(pos),3))
    print("loading solution")
    for n in range(len(pos)):
        pos_val[n] = np.array([[pos[n,0],pos[n,1],1-u[n]]])
    mask = (5 <= pos_val[:, 1]) & (pos_val[:, 1] < 5.05)
    toexp = pos_val[mask]
    pe_exp = f"{peclet}".replace('.', '_')
    exp_rad = f"{ball_radius}".replace('.', '_')
    output_file = f"numerical_results/distributions/ball{exp_rad}_peclet{pe_exp}.txt"
    with open(output_file, 'w') as f:
        for x, y, z in toexp:
            f.write(f"{x}\t{y}\t{z}\n")
    
    return toexp

for i in range(2,9):
    sherwood(10**i, 0.98)

# pe_list = []
# for i in range(3,10):
#     pe_list = pe_list + [round((10**i)*float(round(xi,1)),2) for xi in np.logspace(0, 1, 3)[:-2]]
# # pe_list = [0.1, 0.2, 0.5] + pe_list

# ball_list = []
# for i in range(-3,0):
#     ball_list = ball_list + [(10**i)*ball for ball in [1, 2, 5]]

# ball_list.reverse()
# ball_list = ball_list[1:]

# print(ball_list)
# print(pe_list)

# output_file = f"numerical_results/pytest.txt"
# with open(output_file, 'w') as f:
#     f.write("Peclet\tball_radius\tSherwood_fem\tSherwood_pychast\txargs(list)\tsolutions(list)\n")

# for j in range(len(ball_list)):
#     for n in range(len(pe_list)):
#         peclet = pe_list[n]
#         ball_radius = 1 - ball_list[j]
#         # print(f"radius = {ball_radius}, peclet = {peclet}")
#         femsol = sherwood(peclet, ball_radius)
#         integral, xargs, sol = coll_ker.distribution(peclet, ball_radius, trials = 10**3, mesh_out = 6, mesh_jump = 10)
#         with open(output_file, 'a') as f:
#             f.write(f"{peclet}\t{ball_radius}\t{femsol}\t{integral}")
#             for arg in xargs:
#                 f.write(f"\t{arg}")
#             for arg in sol:
#                 f.write(f"\t{arg}")
#             f.write(f"\n")