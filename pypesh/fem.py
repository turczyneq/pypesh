import pypesh.generate_mesh as msh
import pypesh.stokes_flow as sf
import pypesh.analytic as analytic
import numpy as np
from functools import lru_cache
from pathlib import Path
from skfem import (
    MeshTri,
    Basis,
    ElementTriP1,
    FacetBasis,
    Functional,
    BilinearForm,
)

from skfem import asm, solve, condense
from skfem.helpers import grad, dot


@lru_cache(maxsize=None)
def _get_mesh(mesh_type):
    """
    Loads mesh defined by mesh_type and caches the mesh to avoid repetative loading.

    Parameters
    ---------
    mesh_type: {'default', 'fine', 'coarse'}
        'default' for medium peclet, 'fine' for big peclets and 'coarse' for small peclets

    Returns
    --------
    tuple
        skfem object <skfem MeshTri1 object>, used by scikit-fem and <skfem CellBasis(MeshTri1, ElementTriP1) object> boundaries of mesh

    Example
    -------
    >>> import pypesh.fem as fem
    >>> fem._get_mesh('default')
    (<skfem MeshTri1 object>
    Number of elements: 90756
    Number of vertices: 46225
    Number of nodes: 46225
    Named boundaries [# facets]: left [972], right [40], top [264], bottom [20], ball [396], <skfem CellBasis(MeshTri1, ElementTriP1) object>
    Number of elements: 90756
    Number of DOFs: 46225
    Size: 19603296 B)
    """

    parent_dir = Path(__file__).parent.parent if "__file__" in globals() else Path.cwd()

    mesh_fine_path = (
        parent_dir
        / f"meshes/mesh_{str(0.001).replace('.', '_')}__width_{10}.msh"
    )

    mesh_default_path = (
        parent_dir
        / f"meshes/mesh_{str(0.01).replace('.', '_')}__width_{10}.msh"
    )

    mesh_wide_path = (
        parent_dir
        / f"meshes/mesh_{str(0.05).replace('.', '_')}__width_{20}.msh"
    )

    if mesh_type == "default":
        if mesh_default_path.exists():
            mesh = MeshTri.load(mesh_default_path)
        else:
            mesh = msh.gen_mesh(save=True)

        basis = Basis(mesh, ElementTriP1())

    elif mesh_type == "fine":
        if mesh_fine_path.exists():
            mesh = MeshTri.load(mesh_fine_path)
        else:
            mesh = msh.gen_mesh(mesh=0.001, save=True)

        basis = Basis(mesh, ElementTriP1())

    elif mesh_type == "coarse":
        if mesh_wide_path.exists():
            mesh = MeshTri.load(mesh_wide_path)
        else:
            mesh = msh.gen_mesh(width=20, mesh=0.05, save=True)

        basis = Basis(mesh, ElementTriP1())

    else:
        raise NotImplementedError

    return mesh, basis


def get_mesh(peclet):
    """
    Loads mesh adequate to Peclet number

    Parameters
    ---------
    peclet: float
        Peclet number defined as R u / D.

    Returns
    --------
    tuple
        skfem object <skfem MeshTri1 object> used by scikit-fem, <skfem CellBasis(MeshTri1, ElementTriP1) object> boundaries of mesh


    Example
    -------
    >>> import pypesh.fem as fem
    >>> fem.get_mesh(1000)
    (<skfem MeshTri1 object>
    Number of elements: 90756
    Number of vertices: 46225
    Number of nodes: 46225
    Named boundaries [# facets]: left [972], right [40], top [264], bottom [20], ball [396], <skfem CellBasis(MeshTri1, ElementTriP1) object>
    Number of elements: 90756
    Number of DOFs: 46225
    Size: 19603296 B)
    """
    if peclet > 50000:
        # For big peclets use finer mesh
        mesh_type = "fine"

    elif peclet < 5:
        # For small peclets use wider base, with coarse  mesh
        mesh_type = "coarse"

    else:
        # For regural peclets use default mesh
        mesh_type = "default"

    mesh, basis = _get_mesh(mesh_type)

    return mesh, basis


def sherwood_fem(peclet, ball_radius):
    """
    sherwood is defined as flux_dimesional/(4 pi D R) = U R^2 flux / (4 pi D R) = flux*(Pe/4 pi)

    Parameters
    ----------
    peclet: float
        Peclet number defined as R u / D.

    ball_radius : float
        Radius of the big ball.

    Returns
    --------
    float
        Sherwood calcualted for selected peclet and ball radius using fem approach.

    Example
    -------
    >>> import pypesh.fem as fem
    >>> fem.sherwood_fem(10000, 0.9)
    54.93467214954524
    """

    @BilinearForm
    def advection(k, l, m):
        # Coordinate fields
        r, z = m.x

        v_r, v_y, v_z = sf.stokes_around_sphere_explicite(r, z, ball_radius)

        return (l * v_r * grad(k)[0] + l * v_z * grad(k)[1]) * 2 * np.pi * r

    @BilinearForm
    def claplace(u, v, m):
        """Laplace operator in cylindrical coordinates."""
        r, z = m.x
        return dot(grad(u), grad(v)) * 2 * np.pi * r

    mesh, basis = get_mesh(peclet)

    # Assemble the system matrix
    A = asm(claplace, basis) + peclet * asm(advection, basis)
    # Identify the interior degrees of freedom
    interior = basis.complement_dofs(basis.get_dofs({"bottom", "ball"}))
    # Boundary condition
    u = basis.zeros()
    u[basis.get_dofs("bottom")] = 1.0
    u[basis.get_dofs("ball")] = 0.0
    # Solve the problem
    u = solve(*condense(A, x=u, I=interior))

    fbasis = FacetBasis(mesh, ElementTriP1(), facets="top")

    @Functional
    def intercepted(m):
        # Coordinate fields
        r, z = m.x

        v_r, v_y, v_z = sf.stokes_around_sphere_explicite(r, z, ball_radius)

        phi = m["u"]

        """
        calculation of effective surface: 
            1-phi - propability of hitting
            2*pi*r - measure from cylindrical integration
            v_z - flux is v.n so effective surface is dependent on value of v_z for selected r
        """

        return (1 - phi) * 2 * np.pi * r * v_z

    """
    Calculating Sherwood
    """
    result = analytic.sherwood_from_flux(asm(intercepted, fbasis, u=u), peclet)

    return result
