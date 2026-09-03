r"""
Acoustic-structure coupling matrix of the vibroacoustic problem.

On a conforming mesh the interface is a set of faces shared by the
shell elements of the structure and by the volume elements of the
fluid, with the same node numbering on both sides. Everything the
coupling needs is one surface integral over those faces,

.. math:: \mathbf K_c = \int_S \mathbf N_s^T\, \mathbf n\, \mathbf N_a
          \; dS

with :math:`\mathbf N_s` the structural shape functions (6 DOF/node,
only the three translations are loaded by a pressure),
:math:`\mathbf N_a` the acoustic ones (one pressure per node) and
:math:`\mathbf n` the unit normal, taken here **pointing out of the
fluid**.

Sign convention
---------------
With :math:`\mathbf n` out of the fluid and the
:math:`e^{+j\omega t}` convention of the solver:

- the fluid pushes the structure along :math:`+\mathbf n`, so the nodal
  force on the structure is

  .. math:: \mathbf F_s = \mathbf K_c\, \mathbf p

- the wall moves with velocity :math:`j\omega \mathbf w`, so the normal
  velocity seen by the fluid is
  :math:`v_n = j\omega\, (\mathbf n \cdot \mathbf w)`, which through the
  pyCAFE velocity load
  :math:`-j\rho_0\omega \int_S \mathbf N_a v_n\, dS` gives

  .. math:: \mathbf F_a = \rho_0 \omega^2 \mathbf K_c^T \mathbf w

Written as the single unsymmetric system of the Eulerian
:math:`(\mathbf w, \mathbf p)` formulation, both coupling terms move to
the left-hand side:

.. math::

   \begin{bmatrix}
   \mathbf K_s + j\omega \mathbf C_s - \omega^2 \mathbf M_s &
       -\mathbf K_c \\
   -\rho_0 \omega^2 \mathbf K_c^T &
       \mathbf K_a + j\omega \mathbf C_a - \omega^2 \mathbf M_a
   \end{bmatrix}
   \begin{Bmatrix} \mathbf w \\ \mathbf p \end{Bmatrix}
   =
   \begin{Bmatrix} \mathbf F_s \\ \mathbf F_a \end{Bmatrix}

so the acoustic coupling matrix of the classical notation is
:math:`\mathbf M_c = -\rho_0 \mathbf K_c^T`: the same interface seen
from the two sides, which is why one matrix is the transpose of the
other. The asymmetry is not a loss of reciprocity -- it comes from
pairing a displacement unknown with a pressure unknown, so that one
coupling lands in the stiffness block and the other in the mass block.

Examples
--------
The coupling is normally built by the system preparation, which orients
the normals from the fluid elements behind the interface faces::

    system = prepare_vibroacoustic_system(
        nodes=nodes, groups=groups, rho0=1.204, c0=343.0,
        t=0.002, rho_s=7800.0, E=210e9, nu=0.3,
    )
    Kc = system["coupling"]["Kc"]

Applying a uniform unit pressure must return the vector area of the
interface, which is the cheapest check of orientation and Jacobian::

    >>> interface_area_vector(Kc)          # doctest: +SKIP
    array([0.  , 0.  , 0.48])

See Also
--------
pycafe_vibro.solver_vibroacoustic : Coupled system and solvers.
pycafe_vibro.prepare_vibroacoustic_system : Builds this matrix.
"""

import warnings

import numpy as np
from scipy.sparse import coo_matrix

from pycafe.build_matrices.faces import _VOLUME_FACES, _fluid_face_owners

# 2x2 Gauss rule on the reference quadrilateral.
_G = 1.0 / np.sqrt(3.0)
_GAUSS_QUAD4 = [((-_G, -_G), 1.0), ((_G, -_G), 1.0),
                ((_G, _G), 1.0), ((-_G, _G), 1.0)]


def _quad4_shape(xi, eta):
    """Bilinear shape functions and natural derivatives on a quad face."""
    N = 0.25 * np.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
                         (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)])
    dNdxi = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)])
    dNdeta = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])
    return N, dNdxi, dNdeta


def interface_is_conforming(interface_conn0, fluid_conn0):
    """
    Is every interface face a face of exactly one fluid element?

    This is the question that decides how the coupling can be built: a
    True means the shared-node integral of
    :func:`build_coupling_matrix` applies, a False means the two meshes
    describe the same surface twice and the interpolation route of
    :mod:`pycafe_vibro.coupling_nonconforming` is needed.

    Parameters
    ----------
    interface_conn0 : ndarray (n_faces, 4)
        Structural faces, 0-based.
    fluid_conn0 : ndarray
        Fluid volume connectivity, 0-based.

    Returns
    -------
    conforming : bool
        True when no face is missing from the fluid mesh. Faces shared
        by *two* fluid elements do not make it False: those are interior
        faces of a structure embedded in the fluid, which still share
        their nodes and only need an imposed ``sign``.
    orphans : list of int
        Indices of the faces that are not faces of any fluid element.
    interior : list of int
        Indices of the faces shared by two fluid elements.

    Examples
    --------
    >>> import numpy as np
    >>> fluid = np.array([[0, 1, 2, 3, 4, 5, 6, 7]])
    >>> interface_is_conforming(np.array([[0, 1, 2, 3]]), fluid)[0]
    True
    >>> interface_is_conforming(np.array([[8, 9, 10, 11]]), fluid)[0]
    False
    """
    owners = _fluid_face_owners(fluid_conn0)
    conn = np.asarray(interface_conn0, dtype=int)
    orphans, interior = [], []
    for k, face in enumerate(conn):
        key = tuple(sorted(face.tolist()))
        if key not in owners:
            orphans.append(k)
        elif len(owners[key]) > 1:
            interior.append(k)
    return (not orphans), orphans, interior


def interface_normals(nodes, interface_conn0, fluid_conn0=None, sign=None):
    """
    Unit normal of every interface face, pointing out of the fluid.

    The node ordering of a face fixes its normal only up to a sign, and
    that sign is what decides whether the fluid pushes or pulls the
    structure. It is resolved geometrically: the fluid element behind
    the face is located, and the normal is flipped so that it points
    away from that element's centroid.

    Parameters
    ----------
    nodes : ndarray (N, 3)
        Mesh node coordinates.
    interface_conn0 : ndarray (n_faces, 4)
        Interface faces, 0-based.
    fluid_conn0 : ndarray, optional
        Connectivity of the fluid volume elements, 0-based. Without it
        the orientation cannot be resolved and ``sign`` must be given.
    sign : {+1, -1}, optional
        Force the orientation instead of deducing it, the raw normal of
        the face node ordering being multiplied by this value.

    Returns
    -------
    normals : ndarray (n_faces, 3)
        Unit normals, out of the fluid.

    Raises
    ------
    ValueError
        If neither ``fluid_conn0`` nor ``sign`` is given, if a face is
        not a face of any fluid element (non-conforming mesh), or if it
        is shared by two fluid elements (interior face, no outward
        direction to deduce -- pass ``sign`` to impose one).
    """
    nodes = np.asarray(nodes, dtype=float)
    conn = np.asarray(interface_conn0, dtype=int)

    if sign is None and fluid_conn0 is None:
        raise ValueError(
            "Give either 'fluid_conn0', to deduce the orientation of the "
            "interface normals from the fluid elements, or 'sign' to "
            "impose it."
        )

    owners = None if fluid_conn0 is None else _fluid_face_owners(fluid_conn0)
    fluid_conn = (None if fluid_conn0 is None
                  else np.asarray(fluid_conn0, dtype=int))

    normals = np.zeros((conn.shape[0], 3))
    for k, face in enumerate(conn):
        x_e = nodes[face]
        # Normal of the bilinear face at its centre.
        _, dNdxi, dNdeta = _quad4_shape(0.0, 0.0)
        raw = np.cross(dNdxi @ x_e, dNdeta @ x_e)
        norm = np.linalg.norm(raw)
        if norm <= 0.0:
            raise ValueError(f"Interface face {k} is degenerate.")
        n = raw / norm

        if sign is not None:
            n = sign * n
        else:
            key = tuple(sorted(face.tolist()))
            if key not in owners:
                raise ValueError(
                    f"Interface face {k} (nodes {face.tolist()}) is not a "
                    "face of any fluid element: the mesh is not conforming "
                    "at the interface, so the coupling cannot be built by "
                    "shared nodes."
                )
            if len(owners[key]) > 1:
                raise ValueError(
                    f"Interface face {k} (nodes {face.tolist()}) is shared "
                    f"by {len(owners[key])} fluid elements: it lies inside "
                    "the fluid, where there is no outward normal to deduce. "
                    "Either the interface group also selects internal "
                    "faces, or the structure is embedded in the fluid; in "
                    "the latter case pass 'sign' to impose the orientation "
                    "explicitly."
                )
            elem = fluid_conn[owners[key][0]]
            # Point the normal away from the fluid element behind it.
            outward = x_e.mean(axis=0) - nodes[elem].mean(axis=0)
            if np.dot(outward, n) < 0.0:
                n = -n

        normals[k] = n

    return normals


def build_coupling_matrix(
    nodes,
    interface_conn0,
    fluid_conn0=None,
    *,
    sign=None,
    num_nodes=None,
    dofs_per_node=6,
    nonconforming="auto",
    report=None,
):
    """
    Assemble the acoustic-structure coupling matrix ``Kc``.

    Implements equation (1) of the module header,
    ``Kc = int_S Ns^T n Na dS``, over the shared faces of a conforming
    mesh. Rows are structural DOFs numbered over all mesh nodes
    (``dofs_per_node * N``, as
    :func:`pycafe_vibro.domains.build_KM_structural_domain`
    numbers them), columns are acoustic pressure DOFs, one per node, so
    that both sides can be reduced afterwards with the index sets of
    their own domain.

    Only the three translations of a node are loaded: an inviscid fluid
    applies a normal traction, no couple, so the rotational rows are
    structurally zero.

    Parameters
    ----------
    nodes : ndarray (N, 3)
        Mesh node coordinates.
    interface_conn0 : ndarray (n_faces, 4)
        Interface faces, 0-based. On the usual case (a shell whose
        elements are faces of the fluid) this is the structural
        connectivity itself.
    fluid_conn0 : ndarray, optional
        Fluid volume connectivity, 0-based, used to orient the normals;
        see :func:`interface_normals`.
    sign : {+1, -1}, optional
        Impose the orientation instead of deducing it.
    num_nodes : int, optional
        Total number of mesh nodes; defaults to ``len(nodes)``.
    dofs_per_node : int, optional
        Structural DOFs per node (6 for CQUAD4F).
    nonconforming : {"auto", False} or dict, optional
        What to do when a structural face is **not** a face of any fluid
        element, i.e. when the two meshes do not share their interface
        nodes.

        - ``"auto"`` (default): fall back to the interpolation method of
          :func:`pycafe_vibro.coupling_nonconforming.build_nonconforming_coupling`
          and warn. A conforming interface never reaches that code: it
          is detected first and assembled by the integral below, so
          nothing changes for it.
        - ``False``: refuse, as before, with the error of
          :func:`interface_normals`.
        - a dict: the options of the fallback (``method``, ``shape``,
          ``degree``, ``wet_factor``, ...), which also selects it.
    report : dict, optional
        Filled in place with what the fallback did — see the ``report``
        of
        :func:`~pycafe_vibro.coupling_nonconforming.build_nonconforming_coupling`,
        plus ``conforming``. Left untouched on a conforming interface
        except for that flag.

    Returns
    -------
    Kc : scipy.sparse.csr_matrix, shape (dofs_per_node * N, N)
        Pressure-to-force operator, see equations (2) and (3).

    Notes
    -----
    The total force transferred by a uniform unit pressure is the
    vector area of the interface: ``Kc.sum(axis=1)`` gathers, on the
    translation DOFs, exactly ``int_S n dS``. That identity is the
    cheapest check that both the orientation and the surface Jacobian
    are right, and it is the one the non-conforming fallback verifies
    before returning.

    See Also
    --------
    interface_normals : Orientation of the interface normals.
    pycafe_vibro.coupling_nonconforming : Interpolation route.
    pycafe_vibro.solver_vibroacoustic : Coupled system and solvers.
    """
    nodes = np.asarray(nodes, dtype=float)
    conn = np.asarray(interface_conn0, dtype=int)
    if conn.ndim != 2 or conn.shape[1] != 4:
        raise ValueError(
            "Only 4-node (quadrilateral) interface faces are supported; "
            f"got connectivity of shape {conn.shape}."
        )

    n_nodes = int(num_nodes if num_nodes is not None else nodes.shape[0])

    # Only a mesh that is genuinely not sharing its interface nodes is
    # sent to the interpolation; everything else goes on as before.
    if fluid_conn0 is not None and nonconforming is not False:
        conforming, orphans, _interior = interface_is_conforming(
            conn, fluid_conn0
        )
        if report is not None:
            report["conforming"] = conforming
        if not conforming:
            from .coupling_nonconforming import build_nonconforming_coupling

            options = dict(nonconforming) if isinstance(nonconforming, dict) \
                else {}
            warnings.warn(
                f"{len(orphans)} of {conn.shape[0]} interface faces are not "
                "faces of any fluid element: the meshes do not share their "
                "interface nodes. The coupling is built by the interpolation "
                "method of Mi & Zheng (2018), CMA 338:264-297, instead of by "
                "shared nodes. Pass nonconforming=False to refuse instead.",
                RuntimeWarning,
            )
            Kc, nc_report = build_nonconforming_coupling(
                nodes, conn, fluid_conn0,
                sign=sign, num_nodes=n_nodes, dofs_per_node=dofs_per_node,
                **options,
            )
            if report is not None:
                report.update(nc_report)
                report["conforming"] = False
            return Kc
    normals = interface_normals(nodes, conn, fluid_conn0, sign=sign)

    rows, cols, vals = [], [], []
    for face, n in zip(conn, normals):
        x_e = nodes[face]
        Kc_e = np.zeros((4, 3, 4))          # [node_s, direction, node_a]

        for (xi, eta), w in _GAUSS_QUAD4:
            N, dNdxi, dNdeta = _quad4_shape(xi, eta)
            # Surface Jacobian: the area of the parallelogram spanned by
            # the two tangents.
            detJ = np.linalg.norm(np.cross(dNdxi @ x_e, dNdeta @ x_e))
            # Ns^T n Na, with Ns = Na = N on a conforming face.
            Kc_e += w * detJ * np.einsum("a,i,b->aib", N, n, N)

        for a, node_s in enumerate(face):
            for i in range(3):              # translations only
                for b, node_a in enumerate(face):
                    rows.append(dofs_per_node * node_s + i)
                    cols.append(node_a)
                    vals.append(Kc_e[a, i, b])

    Kc = coo_matrix(
        (vals, (rows, cols)),
        shape=(dofs_per_node * n_nodes, n_nodes),
    ).tocsr()
    Kc.sum_duplicates()
    return Kc


def acoustic_coupling_matrix(Kc, rho0):
    """
    Acoustic-side coupling matrix ``Mc = -rho0 * Kc^T``.

    The classical notation writes the acoustic coupling load as
    ``-omega^2 Mc w``; with the convention of this module that equals
    ``+rho0 omega^2 Kc^T w``, equation (3). The matrix is returned for
    interoperability with that notation -- the solvers here use ``Kc``
    directly.

    Parameters
    ----------
    Kc : sparse matrix
        Output of :func:`build_coupling_matrix`.
    rho0 : float
        Fluid density.

    Returns
    -------
    Mc : sparse matrix, shape (N, dofs_per_node * N)
    """
    return -float(rho0) * Kc.T.tocsr()


def interface_area_vector(Kc, dofs_per_node=6):
    """
    Vector area ``int_S n dS`` recovered from the coupling matrix.

    Summing the columns of ``Kc`` applies a uniform unit pressure; the
    result, gathered per direction, is the resultant force -- the vector
    area of the interface. On a flat interface its norm is the area and
    its direction the normal.

    Returns
    -------
    area : ndarray (3,)
    """
    load = np.asarray(Kc.sum(axis=1)).ravel()
    return np.array([load[i::dofs_per_node].sum() for i in range(3)])
