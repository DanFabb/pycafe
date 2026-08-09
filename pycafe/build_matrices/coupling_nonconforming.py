r"""
Coupling across an interface whose two meshes do not share their nodes.

The conforming coupling of :mod:`pycafe.build_matrices.coupling` needs
every shell element to be a face of a fluid element: the pressure that
loads a structural node is *that* node's pressure. When the two meshes
are built independently — a hexahedral panel against a tetrahedral
fluid, two grids of different density, a curved interface meshed twice —
that identity is gone and the surface integral has no common node
numbering to assemble into.

This module implements the interpolation route of

    Y. Mi, H. Zheng, *An interpolation method for coupling
    non-conforming patches in isogeometric analysis of vibro-acoustic
    systems*, Comput. Methods Appl. Mech. Engrg. 338 (2018) 264-297,
    https://doi.org/10.1016/j.cma.2018.07.002

in the finite-element setting of pyCAFE. Their section 3.3 handles a
structural-acoustic interface in two moves:

1. a **virtual layer of acoustic nodes** is laid on the structural
   grid, conforming to it by construction, so the ordinary coupling
   integral :math:`\mathbf K_c^v = \int_S \mathbf N_s^T \mathbf n
   \mathbf N_a\, dS` can be assembled on the structural faces exactly as
   in the conforming case;

2. the pressures of that virtual layer are written in terms of the real
   fluid pressures by an interpolation matrix :math:`\mathbf A`
   (their equation 18 or 33), which reduces the structural-acoustic
   non-conformity to the acoustic-acoustic one they had already solved.

Their equation (35) then gives the coupling matrix of the
non-conforming interface as the product

.. math:: \mathbf K_c = \mathbf K_c^v\, \mathbf A .

The interpolation itself is built from Euclidean distances between the
two point sets only — no parametric map, no intersection of the two
surface meshes — which is what makes it work between element types that
have nothing in common (quadrilaterals against triangles). Two variants
are provided, both from the paper:

``"bim"``
    Basic interpolation method, section 3.1: a Gaussian radial basis
    interpolant is fitted through the fluid (master) pressures and
    sampled at the virtual (slave) nodes, giving
    :math:`\mathbf A = \mathbf A^s (\mathbf A^m)^{-1}`, equation (18).

``"mls"``
    The same Gaussian used as a moving-least-squares weight,
    section 3.2, so the interpolant is a local polynomial fit rather
    than a global one, equation (30). Their motivation is stated for
    interfaces with strong non-conformity or curved shape.

Master and slave
----------------
The paper takes the finer side as slave and the coarser as master. For a
structural-acoustic interface that choice is already made by the
construction above: the slave is the virtual layer, i.e. the structural
nodes, and the master is the wet boundary of the fluid mesh.

What is *not* claimed here
--------------------------
The interpolation is an approximation of the interface condition, and
what it costs in accuracy depends on the ratio of the two mesh sizes and
on the curvature — the paper's sections 4.1 and 4.2 are the reference
for that. What this module does check, and reports, is the identity that
must hold whatever the meshes are: a uniform pressure on the interface
must transfer the vector area :math:`\int_S \mathbf n\, dS` to the
structure. See :func:`build_nonconforming_coupling` and the
``consistency`` entry of its report.

See Also
--------
pycafe.build_matrices.coupling : Conforming coupling and sign convention.
"""

import warnings

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.spatial import cKDTree

# Weights below exp(-CUTOFF^2) are dropped from the moving least-squares
# fit: the Gaussian is globally supported in the paper, but at three
# shape lengths it contributes ~1e-4 of the peak weight and keeping it
# only makes every fit dense.
CUTOFF = 3.0

# Default multiplier of the shape parameter, lambda = SHAPE / d, with d
# the largest nearest-neighbour distance between centres. The paper
# scales lambda with that distance at every refinement (section 4.1);
# the multiplier itself is the "initial value" they leave open.
DEFAULT_SHAPE = 1.0

DEFAULT_METHOD = "mls"
DEFAULT_DEGREE = 1


# geometry of the two sides
def fluid_boundary_faces(fluid_conn0):
    """
    Faces of the fluid mesh that have a single element behind them.

    Parameters
    ----------
    fluid_conn0 : ndarray (n_elements, 4 or 8)
        Fluid volume connectivity, 0-based. Tetrahedra give triangular
        faces, hexahedra quadrilateral ones.

    Returns
    -------
    faces : list of tuple
        Node indices of each boundary face, in the element's own
        ordering.
    owners : ndarray (n_faces,)
        Index of the fluid element behind each face.
    """
    from .coupling import _VOLUME_FACES

    conn = np.asarray(fluid_conn0, dtype=int)
    local = _VOLUME_FACES.get(conn.shape[1])
    if local is None:
        raise ValueError(
            f"Fluid elements with {conn.shape[1]} nodes are not supported; "
            "expected 8 (hexa) or 4 (tetra)."
        )

    seen = {}
    for e, row in enumerate(conn):
        for face in local:
            nodes = tuple(row[list(face)].tolist())
            key = tuple(sorted(nodes))
            if key in seen:
                seen[key] = None            # interior: two owners
            else:
                seen[key] = (nodes, e)

    kept = [v for v in seen.values() if v is not None]
    faces = [v[0] for v in kept]
    owners = np.array([v[1] for v in kept], dtype=int)
    return faces, owners


def _face_centroids_normals(nodes, faces, owners, fluid_conn0):
    """Centroid and outward unit normal of each fluid boundary face."""
    conn = np.asarray(fluid_conn0, dtype=int)
    centroids = np.zeros((len(faces), 3))
    normals = np.zeros((len(faces), 3))
    for k, face in enumerate(faces):
        x = nodes[list(face)]
        centroids[k] = x.mean(axis=0)
        n = _newell_normal(x)
        # Point it away from the element behind the face.
        if np.dot(centroids[k] - nodes[conn[owners[k]]].mean(axis=0), n) < 0.0:
            n = -n
        normals[k] = n
    return centroids, normals


def _face_size(nodes, conn):
    """Representative size of a set of faces: mean of the bbox diagonals."""
    x = nodes[np.asarray(conn, dtype=int)]
    return float(np.linalg.norm(x.max(axis=1) - x.min(axis=1), axis=1).mean())


def _newell_normal(x):
    """Unit normal of a polygon, valid for triangles and warped quads."""
    n = np.zeros(3)
    for i in range(x.shape[0]):
        n += np.cross(x[i], x[(i + 1) % x.shape[0]])
    norm = np.linalg.norm(n)
    if norm <= 0.0:
        raise ValueError("Degenerate face: its normal vanishes.")
    return n / norm


def _point_triangle_distance(p, a, b, c):
    """Distance from a point to a triangle (Ericson, real-time collision)."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = ab @ ap, ac @ ap
    if d1 <= 0.0 and d2 <= 0.0:
        return np.linalg.norm(p - a)

    bp = p - b
    d3, d4 = ab @ bp, ac @ bp
    if d3 >= 0.0 and d4 <= d3:
        return np.linalg.norm(p - b)

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 <= d1 and d3 <= 0.0:
        return np.linalg.norm(p - (a + d1 / (d1 - d3) * ab))

    cp = p - c
    d5, d6 = ab @ cp, ac @ cp
    if d6 >= 0.0 and d5 <= d6:
        return np.linalg.norm(p - c)

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 <= d2 and d6 <= 0.0:
        return np.linalg.norm(p - (a + d2 / (d2 - d6) * ac))

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return np.linalg.norm(p - (b + w * (c - b)))

    denom = 1.0 / (va + vb + vc)
    return np.linalg.norm(p - (a + ab * (vb * denom) + ac * (vc * denom)))


def _distance_to_quad(p, x):
    """Distance from a point to a quadrilateral, split into two triangles."""
    return min(_point_triangle_distance(p, x[0], x[1], x[2]),
               _point_triangle_distance(p, x[0], x[2], x[3]))


def wet_boundary(nodes, fluid_conn0, interface_conn0, *, wet_factor=1.5,
                 align_min=0.5):
    """
    The part of the fluid boundary that faces the structure.

    The master point set of the interpolation is not the whole fluid
    boundary — that would let a node on the far wall of the cavity
    contribute to the pressure on the panel. A fluid boundary face is
    kept when it satisfies both conditions that make it the same
    surface as the structure:

    * it lies **close** to it — the distance from its centroid to the
      structural faces, measured in units of the local element size, so
      that the selection follows the mesh instead of an absolute
      tolerance;
    * it is **parallel** to it — its normal and the one of the nearest
      structural face point along the same axis. Without this the wall
      that meets the panel at its edge would be taken as part of the
      interface, being only one element away from it.

    Parameters
    ----------
    nodes : ndarray (N, 3)
    fluid_conn0 : ndarray
        Fluid volume connectivity, 0-based.
    interface_conn0 : ndarray (n_faces, 4)
        Structural faces, 0-based.
    wet_factor : float, optional
        Multiplier of ``(h_fluid + h_structure) / 2`` used as the
        selection radius.
    align_min : float, optional
        Smallest ``|n_fluid . n_structure|`` accepted, the sign being
        irrelevant because the structural faces are not oriented yet.
        Zero disables the test.

    Returns
    -------
    dict
        ``faces``, ``owners``, ``centroids``, ``normals`` of the
        selected fluid boundary faces, ``nodes0`` (the master node
        indices) and ``radius`` (the distance used).

    Raises
    ------
    ValueError
        If no fluid boundary face is found near the structure, which
        means the two surfaces are not the same interface at all.
    """
    nodes = np.asarray(nodes, dtype=float)
    faces, owners = fluid_boundary_faces(fluid_conn0)
    centroids, normals = _face_centroids_normals(nodes, faces, owners,
                                                 fluid_conn0)

    iface = np.asarray(interface_conn0, dtype=int)
    s_centroids = nodes[iface].mean(axis=1)
    s_normals = np.array([_newell_normal(nodes[face]) for face in iface])
    # Element size of the two sides, averaged. The fluid boundary faces
    # are ragged (triangles from tetrahedra, quadrilaterals from
    # hexahedra), so their size is measured face by face.
    h_fluid = float(np.mean([
        np.linalg.norm(nodes[list(f)].max(axis=0) - nodes[list(f)].min(axis=0))
        for f in faces
    ]))
    h = 0.5 * (h_fluid + _face_size(nodes, iface))
    radius = float(wet_factor) * h

    tree = cKDTree(s_centroids)
    n_candidates = min(8, iface.shape[0])
    keep = []
    for k, centre in enumerate(centroids):
        _, candidates = tree.query(centre, k=n_candidates)
        candidates = np.atleast_1d(candidates)
        distances = [_distance_to_quad(centre, nodes[iface[c]])
                     for c in candidates]
        nearest = int(candidates[int(np.argmin(distances))])
        if min(distances) > radius:
            continue
        if abs(float(normals[k] @ s_normals[nearest])) < float(align_min):
            continue
        keep.append(k)
    keep = np.array(keep, dtype=int)

    if keep.size == 0:
        raise ValueError(
            "No fluid boundary face lies within "
            f"{radius:.3e} m of the structural surface, parallel to it: the "
            "two meshes do not describe the same interface (check units, "
            "position, or whether the structure is buried inside the fluid)."
        )

    kept_faces = [faces[k] for k in keep]
    master = np.unique(np.concatenate([np.asarray(f, dtype=int)
                                       for f in kept_faces]))
    return {
        "faces": kept_faces,
        "owners": owners[keep],
        "centroids": centroids[keep],
        "normals": normals[keep],
        "nodes0": master,
        "radius": radius,
    }


def normals_from_wet_boundary(nodes, interface_conn0, wet):
    """
    Orient the structural faces with the fluid boundary they face.

    On a conforming mesh the outward direction comes from the fluid
    element behind the face. Here there is no element behind it, so each
    structural face takes the direction of the nearest wet fluid face,
    whose own normal is known to point out of the fluid.

    Parameters
    ----------
    nodes : ndarray (N, 3)
    interface_conn0 : ndarray (n_faces, 4)
    wet : dict
        Output of :func:`wet_boundary`.

    Returns
    -------
    normals : ndarray (n_faces, 3)
        Unit normals out of the fluid, one per structural face.
    """
    from .coupling import _quad4_shape

    nodes = np.asarray(nodes, dtype=float)
    conn = np.asarray(interface_conn0, dtype=int)
    tree = cKDTree(wet["centroids"])

    normals = np.zeros((conn.shape[0], 3))
    for k, face in enumerate(conn):
        x_e = nodes[face]
        _, dNdxi, dNdeta = _quad4_shape(0.0, 0.0)
        raw = np.cross(dNdxi @ x_e, dNdeta @ x_e)
        norm = np.linalg.norm(raw)
        if norm <= 0.0:
            raise ValueError(f"Interface face {k} is degenerate.")
        n = raw / norm
        _, nearest = tree.query(x_e.mean(axis=0))
        if np.dot(n, wet["normals"][nearest]) < 0.0:
            n = -n
        normals[k] = n
    return normals


# interpolation, sections 3.1 and 3.2 of the paper
def shape_parameter(x_master, shape=DEFAULT_SHAPE):
    """
    Gaussian shape parameter ``lambda`` of equation (19).

    The paper scales it with the distance between neighbouring centres
    at each mesh refinement (section 4.1). Here that distance is the
    largest nearest-neighbour distance of the master point set, so that
    the width of the basis follows the coarsest part of the master mesh:

    .. math:: \\lambda = \\frac{c}{\\max_i \\min_{j \\ne i}
              \\|x_i^m - x_j^m\\|}

    Parameters
    ----------
    x_master : ndarray (Nm, 3)
    shape : float, optional
        The multiplier ``c``.

    Returns
    -------
    float
    """
    x = np.asarray(x_master, dtype=float)
    if x.shape[0] < 2:
        raise ValueError("At least two centres are needed.")
    d, _ = cKDTree(x).query(x, k=2)
    spacing = float(d[:, 1].max())
    if spacing <= 0.0:
        raise ValueError("Two centres coincide; the interpolation matrix "
                         "would be singular.")
    return float(shape) / spacing


def _gaussian(r, lam):
    """Gaussian radial basis function, equation (19)."""
    return np.exp(-(lam * r) ** 2)


def _bim_matrix(x_slave, x_master, lam):
    """Basic interpolation method: ``A = A^s (A^m)^-1``, equation (18)."""
    A_m = _gaussian(
        np.linalg.norm(x_master[:, None, :] - x_master[None, :, :], axis=2),
        lam,
    )
    A_s = _gaussian(
        np.linalg.norm(x_slave[:, None, :] - x_master[None, :, :], axis=2),
        lam,
    )
    try:
        A = np.linalg.solve(A_m.T, A_s.T).T
    except np.linalg.LinAlgError:
        A = np.linalg.lstsq(A_m.T, A_s.T, rcond=None)[0].T
    return A


def _polynomial_basis(d, degree):
    """Polynomial basis of equation (20), evaluated on offsets ``d``."""
    ones = np.ones((d.shape[0], 1))
    if degree == 0:
        return ones
    if degree == 1:
        return np.hstack([ones, d])
    if degree == 2:
        x, y, z = d[:, 0:1], d[:, 1:2], d[:, 2:3]
        return np.hstack([ones, d, x * x, y * y, z * z, x * y, y * z, z * x])
    raise ValueError("degree must be 0, 1 or 2.")


def _mls_matrix(x_slave, x_master, lam, degree):
    """
    Moving least squares with a Gaussian weight, equation (30).

    The fit is local: only the centres inside ``CUTOFF / lambda`` of the
    evaluation point carry a non-negligible weight, and the radius is
    grown until there are enough of them to determine the polynomial.
    The moment matrix is inverted in the least-squares sense, because on
    a surface the centres are coplanar and a linear basis in three
    variables is rank-deficient by construction — the pseudo-inverse
    picks the solution that ignores the direction normal to the patch.
    """
    n_slave = x_slave.shape[0]
    n_master = x_master.shape[0]
    K = _polynomial_basis(np.zeros((1, 3)), degree).shape[1]
    tree = cKDTree(x_master)

    A = np.zeros((n_slave, n_master))
    phi0 = _polynomial_basis(np.zeros((1, 3)), degree)[0]

    for i in range(n_slave):
        radius = CUTOFF / lam
        for _ in range(6):
            idx = tree.query_ball_point(x_slave[i], r=radius)
            if len(idx) >= K + 1:
                break
            radius *= 2.0
        else:
            idx = list(range(n_master))
        idx = np.asarray(idx, dtype=int)

        d = x_master[idx] - x_slave[i]
        w = _gaussian(np.linalg.norm(d, axis=1), lam)
        Psi = _polynomial_basis(d, degree)

        M = Psi.T @ (w[:, None] * Psi)                  # equation (27)
        N = Psi.T * w                                   # equation (27)
        coeff = np.linalg.lstsq(M, N, rcond=None)[0]    # M^-1 N
        A[i, idx] = phi0 @ coeff                        # equation (30)
    return A


def interpolation_matrix(x_slave, x_master, *, method=DEFAULT_METHOD,
                         shape=DEFAULT_SHAPE, degree=DEFAULT_DEGREE,
                         normalize=True):
    """
    Matrix ``A`` mapping master pressures to slave pressures.

    Parameters
    ----------
    x_slave : ndarray (Ns, 3)
        Evaluation points — here the nodes of the virtual acoustic
        layer, i.e. the structural interface nodes.
    x_master : ndarray (Nm, 3)
        Centres — here the nodes of the wet fluid boundary.
    method : {"mls", "bim"}, optional
        Section 3.2 or section 3.1 of the paper.
    shape : float, optional
        Multiplier of the shape parameter, see :func:`shape_parameter`.
    degree : int, optional
        Polynomial degree of the moving least squares fit (0 gives
        Shepard's function, equation 31). Ignored by ``"bim"``.
    normalize : bool, optional
        Scale every row to sum to one. A constant pressure is then
        transferred exactly, which is what makes the vector-area
        identity of the coupling matrix hold. The moving least squares
        fit already reproduces constants; the basic interpolation method
        does so only to the accuracy of the fit.

    Returns
    -------
    A : ndarray (Ns, Nm)
    info : dict
        ``lambda``, ``method``, ``degree``, ``row_sum_error`` — the
        largest deviation of a row sum from one *before* any
        normalization, i.e. how far the interpolant is from reproducing
        a uniform pressure.
    """
    x_slave = np.atleast_2d(np.asarray(x_slave, dtype=float))
    x_master = np.atleast_2d(np.asarray(x_master, dtype=float))
    lam = shape_parameter(x_master, shape)

    if method == "bim":
        A = _bim_matrix(x_slave, x_master, lam)
    elif method == "mls":
        A = _mls_matrix(x_slave, x_master, lam, int(degree))
    else:
        raise ValueError(f"method must be 'mls' or 'bim', not {method!r}.")

    row_sums = A.sum(axis=1)
    row_sum_error = float(np.abs(row_sums - 1.0).max())
    if normalize:
        bad = np.abs(row_sums) < 1e-12
        if bad.any():
            raise ValueError(
                f"{int(bad.sum())} slave node(s) receive a vanishing "
                "interpolation: the centres are too far for the shape "
                "parameter in use. Lower 'shape' to widen the basis."
            )
        A = A / row_sums[:, None]

    info = {
        "lambda": lam,
        "method": method,
        "degree": int(degree),
        "row_sum_error": row_sum_error,
        "normalized": bool(normalize),
        "n_master": int(x_master.shape[0]),
        "n_slave": int(x_slave.shape[0]),
    }
    return A, info


# the coupling matrix, equation (35)
def _virtual_layer_coupling(nodes, interface_conn0, normals, slave_nodes0,
                            dofs_per_node, n_nodes):
    """
    ``Kc^v``: the ordinary coupling integral on the virtual layer.

    Identical to the conforming assembly, except that the pressure
    columns are numbered over the slave nodes (the virtual acoustic
    layer sitting on the structural grid) instead of over the mesh
    nodes.
    """
    from .coupling import _GAUSS_QUAD4, _quad4_shape

    position = {int(n): i for i, n in enumerate(slave_nodes0)}
    rows, cols, vals = [], [], []
    for face, n in zip(np.asarray(interface_conn0, dtype=int), normals):
        x_e = nodes[face]
        Kc_e = np.zeros((4, 3, 4))
        for (xi, eta), w in _GAUSS_QUAD4:
            N, dNdxi, dNdeta = _quad4_shape(xi, eta)
            detJ = np.linalg.norm(np.cross(dNdxi @ x_e, dNdeta @ x_e))
            Kc_e += w * detJ * np.einsum("a,i,b->aib", N, n, N)
        for a, node_s in enumerate(face):
            for i in range(3):
                for b, node_a in enumerate(face):
                    rows.append(dofs_per_node * int(node_s) + i)
                    cols.append(position[int(node_a)])
                    vals.append(Kc_e[a, i, b])

    Kc_v = coo_matrix(
        (vals, (rows, cols)),
        shape=(dofs_per_node * n_nodes, len(slave_nodes0)),
    ).tocsr()
    Kc_v.sum_duplicates()
    return Kc_v


def build_nonconforming_coupling(
    nodes,
    interface_conn0,
    fluid_conn0,
    *,
    method=DEFAULT_METHOD,
    shape=DEFAULT_SHAPE,
    degree=DEFAULT_DEGREE,
    normalize=True,
    wet_factor=1.5,
    align_min=0.5,
    sign=None,
    num_nodes=None,
    dofs_per_node=6,
):
    """
    Coupling matrix of an interface whose meshes do not share nodes.

    The three steps of section 3.3 of the paper, in order: a virtual
    acoustic layer on the structural grid, the ordinary coupling
    integral on it, and the interpolation that expresses its pressures
    with the fluid ones —

    .. math:: \\mathbf K_c = \\mathbf K_c^v \\mathbf A,
              \\qquad \\mathbf p^v = \\mathbf A \\mathbf p .

    Parameters
    ----------
    nodes : ndarray (N, 3)
        Mesh node coordinates. Both meshes are numbered in this array.
    interface_conn0 : ndarray (n_faces, 4)
        Structural faces, 0-based.
    fluid_conn0 : ndarray
        Fluid volume connectivity, 0-based.
    method, shape, degree, normalize
        Passed to :func:`interpolation_matrix`.
    wet_factor, align_min : float, optional
        Passed to :func:`wet_boundary`.
    sign : {+1, -1}, optional
        Impose the orientation of the structural faces instead of taking
        it from the fluid boundary they face.
    num_nodes : int, optional
        Total number of mesh nodes; defaults to ``len(nodes)``.
    dofs_per_node : int, optional

    Returns
    -------
    Kc : scipy.sparse.csr_matrix, shape (dofs_per_node * N, N)
        Same shape, same sign convention and same use as the conforming
        matrix of :func:`pycafe.build_matrices.coupling.build_coupling_matrix`,
        so nothing downstream has to know how it was built.
    report : dict
        ``interpolation`` (the info of :func:`interpolation_matrix`),
        ``wet_radius``, ``normals``, and ``consistency``:

        ``load_virtual``
            Nodal load of a uniform unit pressure applied to the virtual
            layer, i.e. what the conforming integral alone gives.
        ``load_interpolated``
            The same load after the interpolation, ``Kc`` summed over
            its columns.
        ``relative_error``
            Norm of the difference over the norm of the first: the part
            of a uniform pressure that the interface loses on the way.
            It vanishes when the interpolation reproduces a constant.
        ``area_structural``, ``area_transferred``
            :math:`\\int_S \\mathbf n\\, dS` before and after, kept for
            reading. On a closed surface both are zero by symmetry,
            which is why they are not what the check is made of.
    """
    nodes = np.asarray(nodes, dtype=float)
    conn = np.asarray(interface_conn0, dtype=int)
    if conn.ndim != 2 or conn.shape[1] != 4:
        raise ValueError(
            "Only 4-node (quadrilateral) interface faces are supported; "
            f"got connectivity of shape {conn.shape}."
        )
    n_nodes = int(num_nodes if num_nodes is not None else nodes.shape[0])

    wet = wet_boundary(nodes, fluid_conn0, conn, wet_factor=wet_factor,
                       align_min=align_min)

    if sign is None:
        normals = normals_from_wet_boundary(nodes, conn, wet)
    else:
        from .coupling import interface_normals

        normals = interface_normals(nodes, conn, None, sign=sign)

    slave_nodes0 = np.unique(conn)
    master_nodes0 = wet["nodes0"]

    Kc_v = _virtual_layer_coupling(nodes, conn, normals, slave_nodes0,
                                   dofs_per_node, n_nodes)

    A, info = interpolation_matrix(
        nodes[slave_nodes0], nodes[master_nodes0],
        method=method, shape=shape, degree=degree, normalize=normalize,
    )

    # Scatter A into the mesh-wide pressure numbering, then equation (35).
    rows = np.repeat(np.arange(len(slave_nodes0)), len(master_nodes0))
    cols = np.tile(master_nodes0, len(slave_nodes0))
    A_global = csr_matrix(
        (A.ravel(), (rows, cols)),
        shape=(len(slave_nodes0), n_nodes),
    )
    Kc = (Kc_v @ A_global).tocsr()
    Kc.eliminate_zeros()

    from .coupling import interface_area_vector

    area_s = np.zeros(3)
    for face, n in zip(conn, normals):
        x_e = nodes[face]
        from .coupling import _GAUSS_QUAD4, _quad4_shape

        for (xi, eta), w in _GAUSS_QUAD4:
            _, dNdxi, dNdeta = _quad4_shape(xi, eta)
            area_s += w * np.linalg.norm(
                np.cross(dNdxi @ x_e, dNdeta @ x_e)) * n

    # A uniform pressure is the reference load: what the conforming
    # integral puts on each node, against what survives the
    # interpolation. Unlike the vector area, this does not cancel on a
    # closed surface, so it is the quantity the check is made of.
    load_virtual = np.asarray(Kc_v.sum(axis=1)).ravel()
    load_interpolated = np.asarray(Kc.sum(axis=1)).ravel()
    denom = np.linalg.norm(load_virtual)

    report = {
        "interpolation": info,
        "wet_radius": wet["radius"],
        "normals": normals,
        "master_nodes0": master_nodes0,
        "slave_nodes0": slave_nodes0,
        "consistency": {
            "load_virtual": load_virtual,
            "load_interpolated": load_interpolated,
            "relative_error": float(
                np.linalg.norm(load_interpolated - load_virtual) / denom
            ) if denom > 0.0 else float("nan"),
            "area_structural": area_s,
            "area_transferred": interface_area_vector(Kc, dofs_per_node),
        },
    }

    tol = 1e-6
    if report["consistency"]["relative_error"] > tol:
        warnings.warn(
            "the non-conforming interpolation does not transfer a uniform "
            "pressure exactly: the nodal load differs by "
            f"{report['consistency']['relative_error']:.2%}. The two grids "
            "are probably too different for the shape parameter in use "
            f"(lambda = {info['lambda']:.3e} 1/m).",
            RuntimeWarning,
        )
    return Kc, report
