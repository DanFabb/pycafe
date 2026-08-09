"""
Spherical wave radiation on an artificial boundary, and the incident
field that may cross it.

The anechoic condition of
:meth:`pycafe.boundary_condition.acoustic_bc.AcousticBC.add_anechoic`
asks the fluid to behave as a plane wave leaving the domain,
``dp/dn = -j k p``. It is exact only at normal incidence on a flat
front, so a body that radiates a spherical wave sees part of it come
back unless the boundary is pushed far away.

On a boundary that is a sphere the exact first-order condition is
available instead:

.. math:: \\frac{\\partial p}{\\partial n} + \\left(jk + \\frac1r\\right) p = 0 ,

with ``r`` the distance from the centre of the sphere. The ``1/r`` term
is what the plane-wave condition is missing; it matters when ``kr`` is
small, i.e. exactly when the boundary is close to the body. In the weak
form the condition contributes

.. math:: \\int_S N_a \\left(jk + \\frac1r\\right) N_b \\, dS

to the dynamic stiffness -- a matrix of its own, assembled here, not an
impedance in disguise: an impedance would have to be
``rho c / (1 - j/(kr))``, which hides a geometric term inside a material
one and cannot express the incident field below.

**Incident field.** When a wave arrives from outside the truncated
domain, the same boundary carries it in. Writing the total field as
``p = p_i + p_s`` and asking the *scattered* part to leave through the
sphere gives an inhomogeneous condition,

.. math::
    \\frac{\\partial p}{\\partial n} + \\left(jk + \\frac1r\\right) p
    = \\frac{\\partial p_i}{\\partial n} + \\left(jk + \\frac1r\\right) p_i
    =: g ,

whose right-hand side is the load vector ``int_S N_a g dS`` assembled by
:meth:`SphericalRadiation.load`. With no scatterer in the domain the
discrete problem then has ``p = p_i`` as its exact solution, which is
the cheapest check that the two terms agree.

Normals come from the geometry, ``n = (x - centre) / r``, not from the
orientation of the face connectivity: on a sphere the outward direction
is known, and reading it from the mesh would make the condition depend
on how the mesher happened to wind its triangles.

See Also
--------
pycafe.boundary_condition.acoustic_bc : Where the condition is declared.
pycafe.build_matrices.pml : The other way to truncate a domain.
"""

import numpy as np
from scipy.sparse import coo_matrix

from pycafe.build_matrices.bc_surface import (
    _surface_jacobian,
    boundary_element_spec,
)


def fit_sphere(points):
    """
    Centre and radius of the sphere that best fits a set of points.

    Used to read the geometry of a radiation boundary off the mesh
    instead of asking for it twice. The fit is the algebraic one: with
    ``|x|^2 = 2 c.x + (R^2 - |c|^2)`` the unknowns enter linearly, so a
    least-squares solve is enough and no starting guess is needed.

    Parameters
    ----------
    points : ndarray of shape (n, 3)
        Coordinates of the boundary nodes.

    Returns
    -------
    centre : ndarray of shape (3,)
    radius : float
        Mean distance from the centre.
    spread : float
        ``max |r_i - radius| / radius``, i.e. how far the points are
        from lying on one sphere. A value of a few times the element
        size over the radius is normal for a faceted mesh; a value of
        order one means the boundary is not a sphere at all.

    Examples
    --------
    >>> t = np.linspace(0.0, np.pi, 7)
    >>> pts = np.column_stack([np.cos(t), np.sin(t), np.zeros_like(t)])
    >>> centre, radius, spread = fit_sphere(pts)
    >>> bool(np.allclose(centre, 0.0, atol=1e-12)), round(radius, 12)
    (True, 1.0)
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"fit_sphere expects (n, 3) coordinates, got {points.shape}."
        )
    if points.shape[0] < 4:
        raise ValueError(
            f"A sphere needs at least 4 points to be fitted, got "
            f"{points.shape[0]}."
        )

    A = np.column_stack([2.0 * points, np.ones(points.shape[0])])
    b = np.sum(points ** 2, axis=1)
    solution, *_ = np.linalg.lstsq(A, b, rcond=None)

    centre = solution[:3]
    distances = np.linalg.norm(points - centre, axis=1)
    radius = float(distances.mean())
    spread = float(np.abs(distances - radius).max() / radius) if radius else 0.0
    return centre, radius, spread


def _quadrature(nodes, faces_by_type):
    """
    Gauss points of a boundary, with the weights already multiplied by
    the surface Jacobian.

    Everything the radiation condition needs is evaluated at those
    points, and none of it depends on frequency, so the sweep only
    re-scales what is computed here once.

    Parameters
    ----------
    nodes : ndarray of shape (N, 2) or (N, 3)
    faces_by_type : dict
        ``{gmsh_element_name: connectivity}``, 1-based, as returned by
        :func:`pycafe.build_matrices.bc_surface.resolve_boundary_faces`.

    Returns
    -------
    conn0 : ndarray of shape (n_elements, n_nodes)
        Element connectivity, 0-based, one entry per element.
    shape_values : ndarray of shape (n_elements, n_gauss, n_nodes)
    weights : ndarray of shape (n_elements, n_gauss)
        ``w * detJ``.
    x_gauss : ndarray of shape (n_elements, n_gauss, 3)

    Notes
    -----
    Element types with different node counts cannot share one array, so
    a mixed boundary (triangles and quadrilaterals) comes back as a list
    of such tuples, one per type.
    """
    nodes = np.asarray(nodes, dtype=float)
    if nodes.shape[1] == 2:
        nodes = np.column_stack([nodes, np.zeros(nodes.shape[0])])

    blocks = []
    for name, conn in faces_by_type.items():
        spec = boundary_element_spec(name)
        if spec is None:
            raise ValueError(
                f"'{name}' is not a supported boundary element type."
            )
        conn0 = np.asarray(conn, dtype=int).reshape(-1, spec["n_nodes"]) - 1
        pts, gauss_w = spec["rule"]()
        evaluated = [spec["shape"](p) for p in pts]

        n_e, n_gp, n_n = conn0.shape[0], len(pts), spec["n_nodes"]
        shape_values = np.empty((n_e, n_gp, n_n))
        weights = np.empty((n_e, n_gp))
        x_gauss = np.empty((n_e, n_gp, 3))

        for e, idx in enumerate(conn0):
            x_e = nodes[idx, :3]
            for q, (N, dN) in enumerate(evaluated):
                shape_values[e, q] = N
                weights[e, q] = gauss_w[q] * _surface_jacobian(dN, x_e)
                x_gauss[e, q] = N @ x_e

        blocks.append((conn0, shape_values, weights, x_gauss))

    return blocks


class SphericalRadiation:
    """
    The spherical radiation condition on one boundary.

    Holds the geometry of the boundary once and answers two questions
    per frequency: what it adds to the dynamic stiffness
    (:meth:`matrix`) and what it puts on the right-hand side
    (:meth:`load`, zero unless an incident field was declared).

    Parameters
    ----------
    nodes : ndarray of shape (N, 2) or (N, 3)
        Nodal coordinates of the whole mesh.
    faces_by_type : dict
        Boundary faces, 1-based, from
        :func:`~pycafe.build_matrices.bc_surface.resolve_boundary_faces`.
    n_dof : int
        Size of the acoustic system.
    c0 : float
        Speed of sound, which turns ``omega`` into ``k``.
    centre : array-like of shape (3,), optional
        Centre of the sphere. Fitted from the boundary nodes if omitted.
    incident : IncidentPlaneWave-like, optional
        Object with ``p0`` and a unit ``direction``; see
        :class:`pycafe.boundary_condition.acoustic_bc.IncidentPlaneWave`.

    Attributes
    ----------
    centre : ndarray of shape (3,)
    radius : float
        Mean radius of the boundary nodes.
    spread : float
        Relative deviation of the boundary from that sphere.

    See Also
    --------
    pycafe.boundary_condition.acoustic_bc.SphericalRadiationOperator
    """

    def __init__(self, nodes, faces_by_type, n_dof, *, c0,
                 centre=None, incident=None):
        self.n_dof = int(n_dof)
        self.c0 = float(c0)
        self.incident = incident

        blocks = _quadrature(nodes, faces_by_type)
        if not blocks:
            raise ValueError(
                "A spherical radiation boundary has no faces: check the "
                "physical group named in the boundary condition."
            )
        self._blocks = blocks

        used = np.unique(np.concatenate([b[0].ravel() for b in blocks]))
        nodes = np.asarray(nodes, dtype=float)
        if nodes.shape[1] == 2:
            nodes = np.column_stack([nodes, np.zeros(nodes.shape[0])])
        fitted_centre, self.radius, self.spread = fit_sphere(nodes[used, :3])
        self.centre = (np.asarray(centre, dtype=float).reshape(3)
                       if centre is not None else fitted_centre)
        if centre is not None:
            distances = np.linalg.norm(nodes[used, :3] - self.centre, axis=1)
            self.radius = float(distances.mean())
            self.spread = float(
                np.abs(distances - self.radius).max() / self.radius
            )

        self._S, self._S_over_r = self._assemble_matrices()

    def _assemble_matrices(self):
        """The two frequency-independent halves, ``S`` and ``S / r``."""
        rows, cols, s_vals, r_vals = [], [], [], []
        for conn0, N, w, x_gp in self._blocks:
            r = np.linalg.norm(x_gp - self.centre, axis=2)      # (n_e, n_gp)
            # S_e = sum_q w_q N N^T, and the same weighted by 1/r_q.
            S_e = np.einsum("eq,eqa,eqb->eab", w, N, N)
            R_e = np.einsum("eq,eqa,eqb->eab", w / r, N, N)
            n_n = conn0.shape[1]
            rows.append(np.repeat(conn0, n_n, axis=1).ravel())
            cols.append(np.tile(conn0, (1, n_n)).ravel())
            s_vals.append(S_e.reshape(conn0.shape[0], -1).ravel())
            r_vals.append(R_e.reshape(conn0.shape[0], -1).ravel())

        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        shape = (self.n_dof, self.n_dof)
        S = coo_matrix((np.concatenate(s_vals), (rows, cols)), shape).tocsr()
        S_over_r = coo_matrix(
            (np.concatenate(r_vals), (rows, cols)), shape
        ).tocsr()
        return S, S_over_r

    @property
    def area(self):
        """Area of the boundary, ``sum of w * detJ``."""
        return float(sum(b[2].sum() for b in self._blocks))

    def matrix(self, omega):
        """
        ``int_S N (jk + 1/r) N dS`` at the given angular frequency.

        Returns
        -------
        scipy.sparse.csr_matrix of complex, shape (n_dof, n_dof)
        """
        k = float(omega) / self.c0
        return (1j * k) * self._S.astype(complex) + self._S_over_r

    def load(self, omega):
        """
        ``int_S N g dS`` with ``g`` the incident-field residual.

        Zero when no incident field was declared, in which case the
        boundary only lets waves out.

        Returns
        -------
        ndarray of complex, shape (n_dof,)
        """
        f = np.zeros(self.n_dof, dtype=complex)
        if self.incident is None:
            return f

        k = float(omega) / self.c0
        direction = np.asarray(self.incident.direction, dtype=float)
        direction = direction / np.linalg.norm(direction)
        k_vec = k * direction
        p0 = complex(self.incident.p0)

        for conn0, N, w, x_gp in self._blocks:
            offset = x_gp - self.centre
            r = np.linalg.norm(offset, axis=2)               # (n_e, n_gp)
            normal = offset / r[:, :, None]
            # p_i = p0 exp(-j k.x) with the exp(+j omega t) convention;
            # dp_i/dn = -j (k.n) p_i.
            p_inc = p0 * np.exp(-1j * (x_gp @ k_vec))
            g = (-1j * np.einsum("eqi,i->eq", normal, k_vec)
                 + 1j * k + 1.0 / r) * p_inc
            contribution = np.einsum("eq,eq,eqa->ea", w, g, N)
            np.add.at(f, conn0.ravel(), contribution.ravel())

        return f

    def incident_field(self, nodes, omega):
        """
        The incident wave sampled at the mesh nodes.

        Handy for post-processing: the scattered field is the computed
        solution minus this.

        Parameters
        ----------
        nodes : ndarray of shape (N, 2) or (N, 3)
        omega : float

        Returns
        -------
        ndarray of complex, shape (N,)
        """
        if self.incident is None:
            raise ValueError("This boundary carries no incident field.")
        nodes = np.asarray(nodes, dtype=float)
        if nodes.shape[1] == 2:
            nodes = np.column_stack([nodes, np.zeros(nodes.shape[0])])
        direction = np.asarray(self.incident.direction, dtype=float)
        direction = direction / np.linalg.norm(direction)
        k_vec = (float(omega) / self.c0) * direction
        return complex(self.incident.p0) * np.exp(-1j * (nodes[:, :3] @ k_vec))
