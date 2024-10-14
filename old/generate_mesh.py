import pygmsh
import numpy as np

from pathlib import Path
from skfem import (
    MeshTri,
    
)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=("Advection diffusion in stokes flow"))
    parser.add_argument(
        "--mesh",
        type=float,
        default=0.01,
        help="size of mesh",
    )

    parser.add_argument(
        "--far_mesh",
        type=float,
        default=0.5,
        help="size of mesh far far away",
    )

    parser.add_argument(
        "--cell_size",
        type=float,
        default=1.0,
        help="the size of the floor",
    )

    parser.add_argument(
        "--width",
        type=float,
        default=10.0,
        help="the width of the cell",
    )

    parser.add_argument(
        "--ceiling",
        type=float,
        default=10.0,
        help="the height of ceiling",
    )

    parser.add_argument(
        "--floor",
        type=float,
        default=10.0,
        help="the distance between floor and centre of ball",
    )

    parser.add_argument(
        "--filename",
        type=str,
        default="test.msh",
        help="where to save the mesh",
    )

    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()


    floor_depth = args.floor*args.cell_size
    ceiling_depth = args.ceiling*args.cell_size
    floor_width = args.width*args.cell_size
    ball_size = 1.0
    ball_segments = int(1/args.mesh)
    mesh_size = args.mesh
    far_mesh = args.far_mesh

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

    mesh.save("meshes/"+args.filename)

    if not args.quiet:

        mesh.draw(boundaries=True).show()