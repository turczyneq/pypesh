from skfem import MeshTri
import numpy as np
import pygmsh


def gen_mesh(
    mesh=0.01,
    far_mesh=0.5,
    cell_size=1,
    width=10,
    ceiling=10,
    floor=10,
    save=False,
    show_mesh=False,
):
    """
    Generates mesh used then by sciki-fem to solve advection-diffusion in Stokes Flow.

    Parameters
    ----------
    mesh: float, optional
        Density of triangles in most interesting region

    far mesh: float, optional
        Density of triangles in region far away from axis, ceiling and sphere

    cell_size: float, optional
        Scales cell size, keeping ratio of walls to ceiling

    width: float, optional
        Width of cell

    ceiling: float, optional
        Distance from 0 to upper boundary of cell

    floor: float, optional
        Distance from 0 to lower boundary of cell

    save: Bool, optional
        Default False, if True than mesh will be saved in /meshes/

    show : bool, optional
        Default False, if True mesh is shown

    Returns
    -------
    <skfem MeshTri1 object>
        skfem object, used by scikit-fem

    Example
    --------
    >>> import pypesh.generate_mesh as msh
    >>> msh.gen_mesh()
    Transforming over 1000 vertices to C_CONTIGUOUS.
    Transforming over 1000 elements to C_CONTIGUOUS.
    <skfem MeshTri1 object>
    Number of elements: 90756
    Number of vertices: 46225
    Number of nodes: 46225
    Named boundaries [# facets]: left [972], right [40], top [264], bottom [20], ball [396]
    """
    floor_depth = floor * cell_size
    ceiling_depth = ceiling * cell_size
    floor_width = width * cell_size
    ball_size = 1.0
    ball_segments = int(1 / mesh)
    mesh_size = mesh
    far_mesh = far_mesh

    box_points = [
        ([0, -floor_depth], far_mesh),
        ([floor_width, -floor_depth], far_mesh),
        ([floor_width, ceiling_depth], far_mesh),
        ([floor_width / 5, ceiling_depth], mesh_size),
        ([0, ceiling_depth], mesh_size),
    ]

    phi_values = np.linspace(0, np.pi, ball_segments)
    ball_points = ball_size * np.column_stack((np.sin(phi_values), np.cos(phi_values)))
    mesh_boundary = np.vstack((np.array([p for p, s in box_points]), ball_points))

    # Create the geometry and mesh using pygmsh
    with pygmsh.geo.Geometry() as geom:
        poly = geom.add_polygon(
            mesh_boundary,
            mesh_size=([s for p, s in box_points]) + ([mesh_size] * len(ball_points)),
        )

        raw_mesh = geom.generate_mesh()

    # Convert the mesh to a skfem MeshTri object and define boundaries
    triangural_mesh = MeshTri(
        raw_mesh.points[:, :2].T, raw_mesh.cells_dict["triangle"].T
    ).with_boundaries(
        {
            "left": lambda x: np.isclose(x[0], 0),  # Left boundary condition
            "right": lambda x: np.isclose(
                x[0], floor_width
            ),  # Right boundary condition
            "top": lambda x: np.isclose(x[1], ceiling_depth),
            "bottom": lambda x: np.isclose(x[1], -floor_depth),
            "ball": lambda x: x[0] ** 2 + x[1] ** 2 < 1.01 * ball_size**2,
        }
    )

    if save:
        import os
        import sys
        from pathlib import Path

        base_dir = (
            Path(__file__).parent.parent if "__file__" in globals() else Path.cwd()
        )

        meshes_dir = base_dir / "meshes"
        meshes_dir.mkdir(parents=True, exist_ok=True)

        filname = (
            base_dir
            / f"meshes/mesh_{str(mesh).replace('.', '_')}__width_{str(width).replace('.', '_')}.msh"
        )
        if filname.exists():
            print("file exists, not saving")
        else:
            triangural_mesh.save(filname)

    if show_mesh:
        triangural_mesh.draw(boundaries=True).show()

    return triangural_mesh
