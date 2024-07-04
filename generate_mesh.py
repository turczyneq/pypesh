import pygmsh
import numpy as np

from pathlib import Path
from skfem import (
    MeshTri,
    
)

floor_depth = 10.0
floor_width = 10.0
ball_size = 1.0
ball_segments = 100
mesh_size = 0.01
far_mesh = 0.5

box_points = [
        ([0, -floor_depth], far_mesh),
        ([floor_width, -floor_depth], far_mesh),
        ([floor_width, floor_depth], far_mesh),
        ([0, floor_depth], mesh_size),
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
        "top": lambda x: np.isclose(x[1], floor_depth),  
        "bottom": lambda x: np.isclose(x[1], -floor_depth), 
        "ball": lambda x: x[0] ** 2 + x[1] ** 2 < 1.1 * ball_size**2,
    }
)

if __name__ == "__main__":

    mesh.draw(boundaries=True).show()


mesh.save("meshes/cylinder_stokes_fine.msh")