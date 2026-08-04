# bc_surface.py
#
# Boundary integration for acoustic boundary conditions.
#
# Every non-essential acoustic boundary condition needs the same two
# integrals over a portion of the fluid boundary:
#
#     g_a  = int_Omega  N_a          dOmega     (load vector)
#     S_ab = int_Omega  N_a N_b      dOmega     (boundary mass matrix)
#
# The prescribed normal velocity (Neumann) uses g, the impedance
# condition (Robin) uses S:
#
#     V_n,a = -j rho0 omega * vbar_n * g_a
#     C_ab  =  rho0 * Abar * S_ab
#
# This module integrates them on the actual boundary *elements* of the
# mesh -- edges for a 2D fluid, faces for a 3D one -- so the same code
# path serves CQUAD4/CQUAD8 models and HEXA8 cavities. The previous
# implementation sorted the boundary node list along its dominant
# direction and integrated consecutive pairs as line segments, which is
# only meaningful for a straight 2D edge.
import numpy as np
from scipy.sparse import coo_matrix

from .element_cquad4 import cquad4_shape
from .element_cquad8 import cquad8_shape


# ------------------------------------------------------------
#  QUADRATURE RULES ON THE REFERENCE BOUNDARY ELEMENT
# ------------------------------------------------------------
def _gauss_line(n_points):
    """Gauss-Legendre rule on the reference segment xi in [-1, 1]."""
    xi, w = np.polynomial.legendre.leggauss(n_points)
    return xi.reshape(-1, 1), w


def _gauss_quad(n_per_direction):
    """Tensor-product Gauss rule on the reference square [-1, 1]^2."""
    xi, w = np.polynomial.legendre.leggauss(n_per_direction)
    XI, ETA = np.meshgrid(xi, xi, indexing="ij")
    WI, WJ = np.meshgrid(w, w, indexing="ij")
    pts = np.column_stack([XI.ravel(), ETA.ravel()])
    return pts, (WI * WJ).ravel()


def _gauss_tri_degree2():
    """3-point symmetric rule on the reference triangle (exact to deg. 2)."""
    pts = np.array([
        [1.0 / 6.0, 1.0 / 6.0],
        [2.0 / 3.0, 1.0 / 6.0],
        [1.0 / 6.0, 2.0 / 3.0],
    ])
    w = np.full(3, 1.0 / 6.0)
    return pts, w


def _gauss_tri_degree4():
    """6-point symmetric rule on the reference triangle (exact to deg. 4)."""
    a1, w1 = 0.445948490915965, 0.111690794839005
    a2, w2 = 0.091576213509771, 0.054975871827661
    pts = np.array([
        [a1, a1], [1.0 - 2.0 * a1, a1], [a1, 1.0 - 2.0 * a1],
        [a2, a2], [1.0 - 2.0 * a2, a2], [a2, 1.0 - 2.0 * a2],
    ])
    w = np.array([w1, w1, w1, w2, w2, w2])
    return pts, w


# ------------------------------------------------------------
#  SHAPE FUNCTIONS ON THE REFERENCE BOUNDARY ELEMENT
# ------------------------------------------------------------
# Every shape routine takes the natural coordinates of one integration
# point and returns (N, dN) with N of shape (n_nodes,) and dN of shape
# (n_nodes, dim) -- dim being the topological dimension of the boundary
# element, 1 for an edge and 2 for a face.
def _line2_shape(pt):
    xi = pt[0]
    N = 0.5 * np.array([1.0 - xi, 1.0 + xi])
    dN = 0.5 * np.array([[-1.0], [1.0]])
    return N, dN


def _line3_shape(pt):
    # Gmsh "Line 3": end nodes at xi = -1, +1, mid node last.
    xi = pt[0]
    N = np.array([
        xi * (xi - 1.0) / 2.0,
        xi * (xi + 1.0) / 2.0,
        1.0 - xi * xi,
    ])
    dN = np.array([
        [(2.0 * xi - 1.0) / 2.0],
        [(2.0 * xi + 1.0) / 2.0],
        [-2.0 * xi],
    ])
    return N, dN


def _tri3_shape(pt):
    xi, eta = pt
    N = np.array([1.0 - xi - eta, xi, eta])
    dN = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])
    return N, dN


def _tri6_shape(pt):
    # Gmsh "Triangle 6": 3 corners, then the mid-edge nodes of the
    # edges (0-1), (1-2), (2-0).
    xi, eta = pt
    L = np.array([1.0 - xi - eta, xi, eta])
    dL = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])

    N = np.array([
        L[0] * (2.0 * L[0] - 1.0),
        L[1] * (2.0 * L[1] - 1.0),
        L[2] * (2.0 * L[2] - 1.0),
        4.0 * L[0] * L[1],
        4.0 * L[1] * L[2],
        4.0 * L[2] * L[0],
    ])
    dN = np.vstack([
        (4.0 * L[0] - 1.0) * dL[0],
        (4.0 * L[1] - 1.0) * dL[1],
        (4.0 * L[2] - 1.0) * dL[2],
        4.0 * (L[0] * dL[1] + L[1] * dL[0]),
        4.0 * (L[1] * dL[2] + L[2] * dL[1]),
        4.0 * (L[2] * dL[0] + L[0] * dL[2]),
    ])
    return N, dN


def _quad4_shape(pt):
    N, dN_dxi, dN_deta = cquad4_shape(pt[0], pt[1])
    return N, np.column_stack([dN_dxi, dN_deta])


def _quad8_shape(pt):
    N, dN_dxi, dN_deta = cquad8_shape(pt[0], pt[1])
    return N, np.column_stack([dN_dxi, dN_deta])


# ------------------------------------------------------------
#  BOUNDARY ELEMENT CATALOGUE
# ------------------------------------------------------------
# Keyed by Gmsh element name, as produced by the mesh loader. `dim` is
# the topological dimension of the boundary entity: an edge (1) bounds
# a 2D fluid, a face (2) bounds a 3D one.
BOUNDARY_ELEMENTS = {
    "Line 2": dict(n_nodes=2, dim=1, shape=_line2_shape,
                   rule=lambda: _gauss_line(2)),
    "Line 3": dict(n_nodes=3, dim=1, shape=_line3_shape,
                   rule=lambda: _gauss_line(3)),
    "Triangle 3": dict(n_nodes=3, dim=2, shape=_tri3_shape,
                       rule=_gauss_tri_degree2),
    "Triangle 6": dict(n_nodes=6, dim=2, shape=_tri6_shape,
                       rule=_gauss_tri_degree4),
    "Quadrilateral 4": dict(n_nodes=4, dim=2, shape=_quad4_shape,
                            rule=lambda: _gauss_quad(2)),
    "Quadrilateral 8": dict(n_nodes=8, dim=2, shape=_quad8_shape,
                            rule=lambda: _gauss_quad(3)),
}

# Aliases used by some Gmsh versions.
BOUNDARY_ELEMENT_ALIASES = {
    "Quadrangle 4": "Quadrilateral 4",
    "Quadrangle 8": "Quadrilateral 8",
}


def boundary_element_spec(name):
    """
    Return the catalogue entry for a Gmsh boundary element name.

    Returns ``None`` when the name is not a supported boundary element
    (e.g. a volume element group).
    """
    key = BOUNDARY_ELEMENT_ALIASES.get(name, name)
    return BOUNDARY_ELEMENTS.get(key)


# ------------------------------------------------------------
#  SURFACE JACOBIAN
# ------------------------------------------------------------
def _surface_jacobian(dN, x_e):
    """
    Differential measure of a boundary element embedded in 3D space.

    Parameters
    ----------
    dN : ndarray of shape (n_nodes, dim)
        Shape function derivatives w.r.t. the natural coordinates.
    x_e : ndarray of shape (n_nodes, 3)
        Nodal coordinates of the boundary element.

    Returns
    -------
    detJ : float
        ``|dx/dxi|`` for an edge, ``|dx/dxi x dx/deta|`` for a face.
        Both are the correct measure for an element embedded in 3D,
        so no local coordinate system has to be built.
    """
    # tangent[k] = dx/d(natural coordinate k), shape (dim, 3)
    tangent = dN.T @ x_e

    if tangent.shape[0] == 1:
        return float(np.linalg.norm(tangent[0]))

    return float(np.linalg.norm(np.cross(tangent[0], tangent[1])))


# ------------------------------------------------------------
#  BOUNDARY INTEGRALS
# ------------------------------------------------------------
def boundary_integrals(nodes, faces_by_type, n_dof):
    """
    Integrate the shape functions over a set of boundary elements.

    Parameters
    ----------
    nodes : ndarray of shape (N, 2) or (N, 3)
        Nodal coordinates. Two-dimensional meshes are padded with z = 0.
    faces_by_type : dict
        ``{gmsh_element_name: (n_faces, n_nodes) array}`` of boundary
        element connectivity, **1-based** (Gmsh convention), as
        returned by :func:`resolve_boundary_faces`.
    n_dof : int
        Total number of acoustic degrees of freedom (nodes) in the mesh.

    Returns
    -------
    g : ndarray of shape (n_dof,)
        Load vector ``g_a = int N_a dOmega``. Its sum is the measure
        (length or area) of the integrated boundary.
    S : scipy.sparse.csr_matrix of shape (n_dof, n_dof)
        Boundary mass matrix ``S_ab = int N_a N_b dOmega``.

    Raises
    ------
    ValueError
        If an element group is not a supported boundary element type.

    Notes
    -----
    Both integrals are computed in one sweep because they always come
    from the same quadrature loop and the same Jacobians.

    See Also
    --------
    resolve_boundary_faces : Turn boundary names into face connectivity.
    """
    nodes = np.asarray(nodes, dtype=float)
    if nodes.shape[1] == 2:
        nodes = np.column_stack([nodes, np.zeros(nodes.shape[0])])

    g = np.zeros(n_dof, dtype=float)
    rows, cols, vals = [], [], []

    for name, conn in faces_by_type.items():
        spec = boundary_element_spec(name)
        if spec is None:
            raise ValueError(
                f"'{name}' is not a supported boundary element type. "
                f"Supported: {sorted(BOUNDARY_ELEMENTS)}"
            )

        conn = np.asarray(conn, dtype=int).reshape(-1, spec["n_nodes"])
        pts, weights = spec["rule"]()
        shape = spec["shape"]

        # Shape functions and derivatives are the same for every
        # element of a given type: evaluate them once.
        N_at_gp = [shape(p) for p in pts]

        for elem in conn:
            idx = elem - 1                      # 1-based -> 0-based
            x_e = nodes[idx, :3]

            g_e = np.zeros(spec["n_nodes"])
            S_e = np.zeros((spec["n_nodes"], spec["n_nodes"]))

            for (N, dN), w in zip(N_at_gp, weights):
                detJ = _surface_jacobian(dN, x_e)
                g_e += N * detJ * w
                S_e += np.outer(N, N) * detJ * w

            g[idx] += g_e
            rows.append(np.repeat(idx, spec["n_nodes"]))
            cols.append(np.tile(idx, spec["n_nodes"]))
            vals.append(S_e.ravel())

    if rows:
        S = coo_matrix(
            (np.concatenate(vals),
             (np.concatenate(rows), np.concatenate(cols))),
            shape=(n_dof, n_dof),
        ).tocsr()
    else:
        S = coo_matrix((n_dof, n_dof)).tocsr()

    return g, S


# ------------------------------------------------------------
#  BOUNDARY RESOLUTION: NAMES / NODE LISTS -> FACE CONNECTIVITY
# ------------------------------------------------------------
def _node_set_from_selection(selection, boundaries):
    """
    Collect the 1-based node tags addressed by a boundary selection.

    ``selection`` is either a list of physical-group names or a list of
    1-based node tags; a bare string is accepted as a single name.
    """
    if isinstance(selection, str):
        selection = [selection]

    selection = list(selection)
    if len(selection) == 0:
        return set(), []

    if isinstance(selection[0], (str, np.str_)):
        if boundaries is None:
            raise ValueError(
                "Boundary names were given but 'boundaries' was not "
                "provided; cannot resolve them to nodes."
            )
        tags = []
        for name in selection:
            if name not in boundaries:
                raise ValueError(
                    f"Boundary '{name}' not found. "
                    f"Available: {list(boundaries.keys())}"
                )
            tags.extend(boundaries[name])
        return set(int(t) for t in tags), list(selection)

    return set(int(t) for t in selection), []


def infer_boundary_dim(elements):
    """
    Topological dimension of the boundary entities of a fluid mesh.

    The fluid domain is the highest-dimensional acoustic element group
    in the mesh, so its boundary is one dimension lower: edges (1) for a
    2D fluid, faces (2) for a 3D one. Returns ``None`` if the domain
    type cannot be identified, leaving the choice to the caller.
    """
    try:
        from .element_registry import find_acoustic_elements
        _, _, spec = find_acoustic_elements(elements)
    except Exception:
        return None
    return spec.dim - 1


def resolve_boundary_faces(
    selection,
    *,
    nodes=None,
    boundaries=None,
    groups=None,
    elements=None,
    boundary_dim=None,
    allow_polyline_fallback=True,
):
    """
    Resolve a boundary selection to boundary element connectivity.

    Boundary conditions are specified by physical-group name (or, for
    backwards compatibility, by an explicit node list), but the integrals
    of :func:`boundary_integrals` need *elements*, not nodes. This
    function bridges the two, trying in order:

    1. the physical groups with connectivity (``groups``), which is the
       authoritative source when available;
    2. the global ``elements`` dictionary, keeping the boundary elements
       whose nodes all belong to the selection;
    3. a polyline built from the sorted node list -- the legacy 2D
       behaviour, kept only as a last resort for straight edges.

    Parameters
    ----------
    selection : list of str or list of int
        Physical-group names, or 1-based node tags.
    nodes : ndarray, optional
        Nodal coordinates. Only used by the polyline fallback, which
        needs to sort the boundary nodes geometrically.
    boundaries : dict, optional
        ``{name: [1-based node tags]}`` from the mesh loader. Needed to
        resolve names when ``groups`` is not available.
    groups : dict, optional
        Physical groups with connectivity, as returned by
        ``load_mesh_with_groups``.
    elements : dict, optional
        ``{gmsh_element_name: connectivity}`` for the whole mesh.
    boundary_dim : int, optional
        Topological dimension of the boundary elements (1 for edges,
        2 for faces). Inferred from ``elements`` when omitted.
    allow_polyline_fallback : bool, optional
        Whether strategy 3 may be used. Default True.

    Returns
    -------
    faces_by_type : dict
        ``{gmsh_element_name: (n_faces, n_nodes) 1-based array}``.
        Empty if the selection is empty.

    Raises
    ------
    ValueError
        If a named boundary does not exist, or if no boundary element
        could be found for a non-empty selection.

    See Also
    --------
    boundary_integrals : Integration over the resolved elements.
    """
    node_set, names = _node_set_from_selection(selection, boundaries)
    if not node_set:
        return {}

    if boundary_dim is None and elements is not None:
        boundary_dim = infer_boundary_dim(elements)

    def _keep(name):
        spec = boundary_element_spec(name)
        if spec is None:
            return False
        return boundary_dim is None or spec["dim"] == boundary_dim

    # --- 1) physical groups with connectivity -----------------------
    if groups is not None and names:
        faces = {}
        for name in names:
            group = groups.get(name)
            if group is None:
                continue
            for etype, conn in group.get("elements", {}).items():
                if not _keep(etype):
                    continue
                conn = np.asarray(conn, dtype=int)
                faces[etype] = (
                    np.vstack([faces[etype], conn]) if etype in faces else conn
                )
        if faces:
            return {k: np.unique(v, axis=0) for k, v in faces.items()}

    # --- 2) boundary elements of the global mesh --------------------
    if elements is not None:
        faces = {}
        for etype, conn in elements.items():
            if not _keep(etype):
                continue
            conn = np.asarray(conn, dtype=int)
            if conn.ndim != 2:
                continue
            # An element belongs to the boundary only if *all* of its
            # nodes do. Gmsh includes the closure of each physical
            # group, so edge/corner nodes shared with a neighbouring
            # group are present and the end elements are not lost.
            keep = np.all(np.isin(conn, list(node_set)), axis=1)
            if np.any(keep):
                faces[etype] = conn[keep]
        if faces:
            return faces

    # --- 3) legacy fallback: polyline through the sorted nodes ------
    if allow_polyline_fallback and (boundary_dim in (None, 1)):
        return {"Line 2": _polyline_from_nodes(node_set, nodes)}

    raise ValueError(
        "Could not resolve any boundary element for the selection "
        f"{names or sorted(node_set)[:8]}. Provide the mesh 'elements' "
        "(or the physical 'groups' with connectivity) so that the "
        "boundary can be integrated on its faces."
    )


def _polyline_from_nodes(node_set, nodes):
    """
    Build a Line 2 connectivity by chaining the boundary nodes.

    Legacy path for 2D meshes loaded without their boundary elements:
    the nodes are sorted along the dominant direction of the edge and
    consecutive pairs become segments. Only valid for a single straight
    (or monotone) edge -- it cannot represent a closed or branching
    boundary, which is why the element-based strategies come first.
    """
    tags = np.array(sorted(node_set), dtype=int)
    if tags.size < 2:
        return np.zeros((0, 2), dtype=int)

    if nodes is not None:
        coords = np.asarray(nodes, dtype=float)[tags - 1, :2]
        spread = coords.max(axis=0) - coords.min(axis=0)
        tags = tags[np.argsort(coords[:, int(np.argmax(spread))])]

    return np.column_stack([tags[:-1], tags[1:]])
