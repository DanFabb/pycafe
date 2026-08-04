# source_volume.py
#
# Volume operations for acoustic source terms.
#
# The volumetric source distribution q(x) [1/s] enters the dynamic
# system through the load vector
#
#     Q_a = j rho0 omega int_V N_a q dV .
#
# Two geometric quantities are needed and provided here:
#
#   * the volume load vector g_a = int_V N_a dV, for a source density
#     that is constant over (part of) the domain;
#   * the shape function values N_a(x_s) of the element containing an
#     arbitrary point x_s, for a point monopole q(x) = q_s delta(x-x_s).
#
# The point evaluation inverts the isoparametric map x(xi) = N(xi) x_e
# with a Newton iteration, which converges in a handful of steps on any
# non-degenerate element.
import numpy as np

from .element_cquad4 import cquad4_shape
from .element_cquad8 import cquad8_shape
from .element_hex8 import hex8_shape


def _quad4(pt):
    N, dN_dxi, dN_deta = cquad4_shape(pt[0], pt[1])
    return N, np.column_stack([dN_dxi, dN_deta])


def _quad8(pt):
    N, dN_dxi, dN_deta = cquad8_shape(pt[0], pt[1])
    return N, np.column_stack([dN_dxi, dN_deta])


def _hex8(pt):
    N, dN_dxi, dN_deta, dN_dzeta = hex8_shape(pt[0], pt[1], pt[2])
    return N, np.column_stack([dN_dxi, dN_deta, dN_dzeta])


def _gauss_tensor(dim, n=2):
    """Tensor-product Gauss rule on [-1, 1]^dim."""
    xi, w = np.polynomial.legendre.leggauss(n)
    grids = np.meshgrid(*([xi] * dim), indexing="ij")
    wgrids = np.meshgrid(*([w] * dim), indexing="ij")
    pts = np.column_stack([g.ravel() for g in grids])
    weights = np.ones(pts.shape[0])
    for wg in wgrids:
        weights *= wg.ravel()
    return pts, weights


# Acoustic domain elements: shape routine, topological dimension and
# quadrature order for the load vector int N dV.
VOLUME_ELEMENTS = {
    "Quadrilateral 4": dict(n_nodes=4, dim=2, shape=_quad4, quad_n=2),
    "Quadrilateral 8": dict(n_nodes=8, dim=2, shape=_quad8, quad_n=3),
    "Quadrangle 4": dict(n_nodes=4, dim=2, shape=_quad4, quad_n=2),
    "Quadrangle 8": dict(n_nodes=8, dim=2, shape=_quad8, quad_n=3),
    "Hexahedron 8": dict(n_nodes=8, dim=3, shape=_hex8, quad_n=2),
}


def volume_load_vector(nodes, elements, n_dof):
    """
    Integrate the shape functions over the acoustic domain.

    Parameters
    ----------
    nodes : ndarray of shape (N, 2) or (N, 3)
        Nodal coordinates.
    elements : dict
        ``{gmsh_element_name: (n_elem, n_nodes) array}``, 1-based.
        Only registered volume element groups are integrated; boundary
        groups (lines on a 2D mesh, faces on a 3D one) are skipped.
    n_dof : int
        Total number of acoustic degrees of freedom.

    Returns
    -------
    g : ndarray of shape (n_dof,)
        ``g_a = int_V N_a dV``. Its sum is the measure (area or
        volume) of the integrated domain.

    Notes
    -----
    When a mesh holds both volume and boundary groups (a HEXA8 cavity
    with its Quadrilateral 4 wall faces), only the groups of highest
    topological dimension are the domain; the rest are skipped.

    See Also
    --------
    point_source_shape : Shape functions at an arbitrary point.
    """
    nodes = np.asarray(nodes, dtype=float)

    present = {
        name: spec for name, spec in VOLUME_ELEMENTS.items()
        if name in elements
    }
    if not present:
        raise ValueError(
            "No registered acoustic domain elements in the mesh. "
            f"Registered: {sorted(set(VOLUME_ELEMENTS))}; "
            f"mesh groups: {list(elements)}"
        )
    domain_dim = max(spec["dim"] for spec in present.values())

    g = np.zeros(n_dof, dtype=float)

    for name, spec in present.items():
        if spec["dim"] != domain_dim:
            continue                      # boundary group, not the domain
        conn = np.asarray(elements[name], dtype=int)
        pts, weights = _gauss_tensor(spec["dim"], spec["quad_n"])
        N_at_gp = [spec["shape"](p) for p in pts]

        for elem in conn:
            idx = elem - 1
            x_e = nodes[idx, :spec["dim"]] if spec["dim"] == 2 \
                else nodes[idx, :3]
            g_e = np.zeros(spec["n_nodes"])
            for (N, dN), w in zip(N_at_gp, weights):
                J = dN.T @ x_e
                g_e += N * abs(np.linalg.det(J)) * w
            g[idx] += g_e

    return g


def _invert_isoparametric(shape, x_e, x_s, dim, tol=1e-10, max_iter=30):
    """
    Newton inversion of the isoparametric map: find xi with x(xi) = x_s.

    Returns (xi, N) on convergence, or None if the iteration diverges.
    """
    xi = np.zeros(dim)
    for _ in range(max_iter):
        N, dN = shape(xi)
        residual = x_s - N @ x_e
        if np.linalg.norm(residual) < tol:
            return xi, N
        J = dN.T @ x_e                    # (dim, dim)
        try:
            xi = xi + np.linalg.solve(J.T, residual)
        except np.linalg.LinAlgError:
            return None
        if np.any(np.abs(xi) > 10.0):     # hopeless: far outside
            return None
    return None


def point_source_shape(nodes, elements, x_s, tol=1e-8):
    """
    Locate the element containing a point and evaluate its shape functions.

    This is the geometric half of a point monopole: for a source
    ``q(x) = q_s delta(x - x_s)`` the load vector is
    ``Q_a = j rho0 omega q_s N_a(x_s)``, non-zero only on the nodes of
    the element containing ``x_s``.

    Parameters
    ----------
    nodes : ndarray of shape (N, 2) or (N, 3)
        Nodal coordinates.
    elements : dict
        Mesh connectivity (1-based), as returned by the mesh loader.
    x_s : array-like
        Source position: (x, y) on a 2D mesh, (x, y, z) on a 3D one.
    tol : float, optional
        Tolerance on the natural coordinates for the containment test
        (``|xi| <= 1 + tol``). Default 1e-8.

    Returns
    -------
    node_idx : ndarray of int
        0-based global indices of the nodes of the containing element.
    N : ndarray of float
        Shape function values at the source point; they sum to 1.

    Raises
    ------
    ValueError
        If no element contains the point.

    Notes
    -----
    A source lying exactly on a node yields ``N`` equal to 1 on that
    node and 0 elsewhere, recovering the pure nodal source
    ``Q_i = j rho0 omega q_s``.

    See Also
    --------
    volume_load_vector : Load vector of a distributed source.
    """
    nodes = np.asarray(nodes, dtype=float)
    x_s = np.asarray(x_s, dtype=float).ravel()

    present = {
        name: spec for name, spec in VOLUME_ELEMENTS.items()
        if name in elements
    }
    if not present:
        raise ValueError("No registered acoustic domain elements in the mesh.")
    domain_dim = max(spec["dim"] for spec in present.values())

    if x_s.size < domain_dim:
        raise ValueError(
            f"Source position has {x_s.size} coordinates but the domain "
            f"is {domain_dim}D."
        )
    x_s = x_s[:domain_dim]

    for name, spec in present.items():
        if spec["dim"] != domain_dim:
            continue
        conn = np.asarray(elements[name], dtype=int)

        for elem in conn:
            idx = elem - 1
            x_e = nodes[idx, :domain_dim]

            # Cheap bounding-box rejection before the Newton iteration.
            pad = tol * (1.0 + np.abs(x_s).max())
            if np.any(x_s < x_e.min(axis=0) - pad) or \
               np.any(x_s > x_e.max(axis=0) + pad):
                continue

            result = _invert_isoparametric(spec["shape"], x_e, x_s, domain_dim)
            if result is None:
                continue
            xi, N = result
            if np.all(np.abs(xi) <= 1.0 + tol):
                return idx, N

    raise ValueError(
        f"No element contains the source point {x_s.tolist()}."
    )
