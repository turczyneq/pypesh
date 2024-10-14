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
mesh = MeshTri.load(Path(__file__).parent / "meshes" / args.mesh)

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

def sherwood(peclet):
    # Assemble the system matrix
    A =  asm(claplace, basis) + peclet * asm(advection, basis)

    # Identify the interior degrees of freedom
    interior = basis.complement_dofs(basis.get_dofs({"bottom", "ball"}))

    # Boundary condition
    u = basis.zeros()
    u[basis.get_dofs("bottom")] = 1.0
    u[basis.get_dofs("ball")] = 0.0

    u = solve(*condense(A, x=u, I=interior))

    dofs = basis.get_dofs("top")
    vals = [[x, 1-el] for x, el in zip(mesh.p[0, dofs.nodal["u"]],u[basis.get_dofs("top")])]
    vals = sorted(vals, key=lambda pair: pair[0])
    xargs, yargs = zip(*vals)
    pe_exp = f"{peclet}".replace('.', '_')
    output_file = f"numerical_results/sh_vs_pe/ball{args.ball}/peclet" + pe_exp + ".txt"
    with open(output_file, 'w') as f:
        f.write("radius\tpropability of hit\n")
        for x, y in vals:
            f.write(f"{x}\t{y}\n")

    if __name__ == "__main__" and not args.quiet:
        import matplotlib.pyplot as plt

        # mesh.draw(boundaries=True).show()

        plt.figure(figsize=(10, 6))
        plt.plot(xargs, yargs, color='b', marker = 'o')

        # Add labels and title
        plt.xlabel('radius')
        plt.ylabel('propability')

        # Show the plot
        plt.show()

        # basis.plot(u, shading="gouraud", cmap="viridis").show()

    fbasis = FacetBasis(mesh, ElementTriP1(), facets="top")

    @Functional
    def intercepted(m):
        # Coordinate fields
        r, z = m.x

        u = 1  # velocity scale
        a = args.ball  # ball size

        w = r**2 + z**2

        v_z = u + ((3 * a * u) / (4 * w**0.5)) * (
        (2 * a**2 + 3 * r**2) / (3 * w) - ((a * r) / w) ** 2 - 2
        )
        phi = m["u"]

        '''
        calculation of effective surface: 
            1-phi - propability of hitting
            2*pi*r - measure from cylindrical integration
            v_z - flux is v.n so effective surface is dependent on value of v_z for selected r
        '''

        return (1 - phi) * 2 * np.pi * r * v_z

    '''
    Calculating Sherwood
    '''
    result = peclet*asm(intercepted, fbasis, u=u)/(4*np.pi)

    return result

    return result

sherwood(10**6)

pe_list = np.logspace(0, 1, 10)


from tqdm import tqdm
try:  
    os.mkdir(f"numerical_results/sh_vs_pe/ball{args.ball}")
except OSError as error:  
    pass

result = np.zeros((len(pe_list),2))
for n in tqdm(range(len(pe_list))):
    result[n,0] = pe_list[n]
    result[n,1] = sherwood(pe_list[n]) 

output_file = f"numerical_results/sh_vs_pe/ball{args.ball}.txt"
with open(output_file, 'w') as f:
    f.write(f"Peclet\tSherwood\n")    
    for x, y in result:
        f.write(f"{x}\t{y}\n")