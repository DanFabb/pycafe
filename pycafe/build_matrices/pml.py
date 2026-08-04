"""
Perfectly Matched Layer for the acoustic elements, in 2D and 3D.

A PML truncates an unbounded domain by surrounding it with a layer in
which the coordinates are continued into the complex plane,

.. math:: \\tilde x_j = \\int_0^{x_j} \\gamma_j(s)\\, ds ,
   \\qquad \\gamma_j = 1 - i\\,\\frac{\\sigma_j}{k} ,
   \\qquad k = \\frac{\\omega}{c_0} .

With the ``exp(+i omega t)`` convention used throughout pyCAFE an
outgoing wave is ``exp(-i k x)``, so inside the layer it becomes
``exp(-i k x) exp(-int sigma)``: it decays, and does so without
reflecting at the interface, because the stretching is analytic there.

Writing the Helmholtz equation in the stretched coordinates and going
back to the weak form gives, on a PML element,

.. math::
    \\int_{V_e} \\sum_j \\Lambda_j \\,
    \\partial_j p \\, \\partial_j q \\; dV
    \\; - \\; k^2 \\Big(\\prod_j \\gamma_j\\Big) \\int_{V_e} p\\,q \\; dV ,
    \\qquad
    \\Lambda_j = \\frac{\\prod_m \\gamma_m}{\\gamma_j^2} ,

which in 2D is the familiar ``diag(gamma_y/gamma_x, gamma_x/gamma_y)``
and in 3D ``diag(gamma_y gamma_z/gamma_x, ...)``. Both come out of the
same expression, with ``gamma_z = 1`` in the plane case.

Note that the stretching enters the stiffness **once**, not squared: the
two gradients are those of the trial and test function and only the
operator between them is stretched. Nothing is conjugated — the whole
point is that the result is complex.

Frequency dependence
--------------------
``Lambda_j`` is a rational function of ``omega``, so the PML stiffness
cannot be split into a finite set of frequency-independent matrices the
way ``K - omega^2 M`` is. What *can* be done, and is done here, is to
integrate once and rescale: ``sigma`` is evaluated at the element
centroid, so it is constant over the element and the four scalars
``Lambda_j`` and ``-k^2 prod(gamma)`` multiply integrals that never
change. :meth:`PMLOperator.matrix` then rebuilds the sparse matrix at
each frequency without touching the quadrature.

Taking ``sigma`` at the centroid makes the profile piecewise constant
across the layer. It is a choice of this implementation, not a property
of the method.
"""

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from .element_cquad4 import cquad4_shape
from .element_cquad8 import cquad8_shape, gauss_rule_quad_3x3, jacobian_2d
from .element_hex8 import gauss_rule_hex_2x2x2, hex8_shape, jacobian_3d
from .element_tetra4 import tetra4_geometry


# ---------------------------------------------------------------------------
#  Absorption profile
# ---------------------------------------------------------------------------

@dataclass
class CartesianPML:
    """
    A box-shaped PML wrapped around a rectangular physical domain.

    The physical (non-absorbing) region is the box
    ``[inner_min, inner_max]``; outside it, along each axis, the layer
    extends by ``thickness`` and the absorption grows as a power of the
    depth,

    .. math:: \\sigma_j(x) = \\sigma_0^{(j)}
              \\left(\\frac{d_j(x)}{L_j}\\right)^{n} ,

    where ``d_j`` is how far past the interface the point lies along
    axis ``j`` (zero inside the physical box) and ``L_j`` the thickness
    on that axis. An axis with zero thickness is not stretched, which is
    how a layer on one side only — the end of a duct — is described.

    Parameters
    ----------
    inner_min, inner_max : array_like of shape (dim,)
        Bounds of the physical region, ``dim`` being 2 or 3.
    thickness : float or array_like of shape (dim,)
        Layer thickness per axis [m]. Zero disables the axis.
    sigma0 : float or array_like of shape (dim,)
        Absorption at the outer face of the layer, in **1/m** — it is
        added to the wavenumber, so it carries its units.
    order : int, optional
        Exponent ``n`` of the profile. Default 2.

    Notes
    -----
    ``sigma0`` and ``order`` set the reflection the layer leaves behind.
    For a normally incident plane wave on a layer of thickness ``L`` the
    round trip through it attenuates the amplitude by
    ``exp(-2 int_0^L sigma ds) = exp(-2 sigma0 L / (n+1))``, which is the
    quantity to size, against the discretization error of the same
    layer: too much absorption over too few elements reflects off the
    jump in ``sigma`` between neighbouring elements instead of absorbing.

    See Also
    --------
    build_pml_operator : Assemble the layer.
    """

    inner_min: np.ndarray
    inner_max: np.ndarray
    thickness: np.ndarray
    sigma0: np.ndarray
    order: int = 2

    def __post_init__(self):
        self.inner_min = np.atleast_1d(np.asarray(self.inner_min, float))
        self.inner_max = np.atleast_1d(np.asarray(self.inner_max, float))
        dim = self.inner_min.size
        if dim not in (2, 3):
            raise ValueError(f"A PML needs 2 or 3 dimensions, got {dim}.")
        if self.inner_max.size != dim:
            raise ValueError(
                f"inner_min has {dim} components, inner_max "
                f"{self.inner_max.size}."
            )
        if np.any(self.inner_max <= self.inner_min):
            raise ValueError(
                f"inner_max must exceed inner_min on every axis, got "
                f"{self.inner_min} .. {self.inner_max}."
            )

        self.thickness = np.broadcast_to(
            np.asarray(self.thickness, float), (dim,)
        ).astype(float).copy()
        self.sigma0 = np.broadcast_to(
            np.asarray(self.sigma0, float), (dim,)
        ).astype(float).copy()

        if np.any(self.thickness < 0.0):
            raise ValueError(f"Negative PML thickness: {self.thickness}.")
        if np.any(self.sigma0 < 0.0):
            raise ValueError(f"Negative sigma0: {self.sigma0}.")
        if self.order < 0:
            raise ValueError(f"Negative profile order: {self.order}.")

    @property
    def dim(self):
        return self.inner_min.size

    def depth(self, points):
        """
        Depth into the layer, per axis, normalized by the thickness.

        Parameters
        ----------
        points : ndarray of shape (npts, dim) or (dim,)

        Returns
        -------
        ndarray of shape (npts, dim)
            Values in ``[0, 1]``; zero inside the physical box. Points
            beyond the outer face are clipped to 1 rather than
            extrapolated, so a slightly oversized layer does not blow
            the profile up.
        """
        pts = np.atleast_2d(np.asarray(points, float))[:, :self.dim]
        past = np.maximum(
            np.maximum(self.inner_min - pts, pts - self.inner_max), 0.0
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            normalized = np.where(self.thickness > 0.0,
                                  past / self.thickness, 0.0)
        return np.clip(normalized, 0.0, 1.0)

    def sigma(self, points):
        """
        Absorption ``sigma_j`` [1/m] at the given points.

        Returns
        -------
        ndarray of shape (npts, dim)
        """
        return self.sigma0 * self.depth(points) ** self.order


# ---------------------------------------------------------------------------
#  Element quadrature: N, physical gradients, and the integration weight
# ---------------------------------------------------------------------------

def _quadrature_quad(shape_fn, quad_rule):
    def points(x_e):
        x2d = np.asarray(x_e, float)[:, :2]
        xi_pts, eta_pts, w_pts = quad_rule()
        for xi, eta, w in zip(xi_pts, eta_pts, w_pts):
            N, dN_dxi, dN_deta = shape_fn(xi, eta)
            _, detJ, invJ = jacobian_2d(dN_dxi, dN_deta, x2d)
            grad = invJ @ np.vstack([dN_dxi, dN_deta])
            yield N, grad, detJ * w
    return points


def _quadrature_hex8(x_e):
    x3d = np.asarray(x_e, float)[:, :3]
    xi_pts, eta_pts, zeta_pts, w_pts = gauss_rule_hex_2x2x2()
    for xi, eta, zeta, w in zip(xi_pts, eta_pts, zeta_pts, w_pts):
        N, dN_dxi, dN_deta, dN_dzeta = hex8_shape(xi, eta, zeta)
        _, detJ, invJ = jacobian_3d(dN_dxi, dN_deta, dN_dzeta, x3d)
        grad = invJ @ np.vstack([dN_dxi, dN_deta, dN_dzeta])
        yield N, grad, detJ * w


def _quadrature_tetra4(x_e):
    """
    One point is exact for the gradients, which are constant, but the
    mass integrand is quadratic — hence the four-point rule of degree 2
    on the reference tetrahedron.
    """
    volume, grad = tetra4_geometry(np.asarray(x_e, float)[:, :3])
    a, b = 0.585410196624969, 0.138196601125011      # (5 +- sqrt(5))/20
    bary = np.array([
        [a, b, b, b], [b, a, b, b], [b, b, a, b], [b, b, b, a],
    ])
    for N in bary:
        yield N, grad, volume / 4.0


_QUADRATURE = {
    "Quadrilateral 4": (4, 2, _quadrature_quad(cquad4_shape,
                                               gauss_rule_quad_3x3)),
    "Quadrangle 4": (4, 2, _quadrature_quad(cquad4_shape,
                                            gauss_rule_quad_3x3)),
    "Quadrilateral 8": (8, 2, _quadrature_quad(cquad8_shape,
                                               gauss_rule_quad_3x3)),
    "Quadrangle 8": (8, 2, _quadrature_quad(cquad8_shape,
                                            gauss_rule_quad_3x3)),
    "Hexahedron 8": (8, 3, _quadrature_hex8),
    "Tetrahedron 4": (4, 3, _quadrature_tetra4),
}


# ---------------------------------------------------------------------------
#  The assembled layer
# ---------------------------------------------------------------------------

@dataclass
class PMLOperator:
    """
    The contribution of the PML elements, ready to be evaluated at any
    frequency.

    This **replaces** the ordinary ``K - omega^2 M`` of those elements;
    it is not a correction added on top. Assemble the rest of the mesh
    without them.

    Attributes
    ----------
    n_dof : int
        Size of the global system.
    dim : int
    c0 : float
    sigma_e : ndarray of shape (n_elem, dim)
        Absorption of each PML element, at its centroid.
    """

    n_dof: int
    dim: int
    c0: float
    sigma_e: np.ndarray
    _rows: np.ndarray = field(repr=False)
    _cols: np.ndarray = field(repr=False)
    _entry_elem: np.ndarray = field(repr=False)
    _grad_vals: np.ndarray = field(repr=False)
    _mass_vals: np.ndarray = field(repr=False)

    @property
    def n_elements(self):
        return self.sigma_e.shape[0]

    def stretching(self, omega):
        """
        The per-element stretching factors at one frequency.

        Returns
        -------
        gamma : ndarray (n_elem, dim), complex
            ``gamma_j = 1 - i sigma_j / k``.
        lambdas : ndarray (n_elem, dim), complex
            ``Lambda_j = prod(gamma) / gamma_j**2``, the coefficient of
            ``d/dx_j`` in the stiffness.
        mass_coeff : ndarray (n_elem,), complex
            ``-k^2 prod(gamma)``, the coefficient of ``int N N``.
        """
        if omega <= 0.0:
            raise ValueError(
                f"A PML is defined at a positive frequency, got "
                f"omega = {omega}. The stretching gamma = 1 - i sigma/k "
                "is singular at omega = 0."
            )
        k = omega / self.c0
        gamma = 1.0 - 1j * self.sigma_e / k
        prod = np.prod(gamma, axis=1)
        return gamma, prod[:, None] / gamma ** 2, -(k ** 2) * prod

    def matrix(self, omega):
        """
        Assemble the PML contribution at one angular frequency.

        Parameters
        ----------
        omega : float
            Angular frequency [rad/s], strictly positive.

        Returns
        -------
        scipy.sparse.csr_matrix, complex, shape (n_dof, n_dof)
        """
        _, lambdas, mass_coeff = self.stretching(omega)

        values = (self._grad_vals * lambdas[self._entry_elem]).sum(axis=1)
        values = values + self._mass_vals * mass_coeff[self._entry_elem]

        return sp.coo_matrix(
            (values, (self._rows, self._cols)),
            shape=(self.n_dof, self.n_dof),
        ).tocsr()


def build_pml_operator(nodes, elements, c0, pml, n_dof=None):
    """
    Integrate the PML elements once, for use at any frequency.

    Parameters
    ----------
    nodes : ndarray of shape (N, 2) or (N, 3)
        Nodal coordinates of the whole mesh.
    elements : dict
        Connectivity of the **PML elements only**, ``{gmsh name: (n_elem,
        n_nodes) array}``, 1-based. Typically the physical group of the
        layer, from ``load_mesh_with_groups``.
    c0 : float
        Speed of sound.
    pml : CartesianPML
        Geometry and absorption profile of the layer.
    n_dof : int, optional
        Size of the global system. Defaults to the number of nodes.

    Returns
    -------
    PMLOperator

    Raises
    ------
    ValueError
        If an element type has no quadrature here, if the mesh mixes
        topological dimensions, or if the elements do not match the
        dimension of ``pml``.
    """
    nodes = np.asarray(nodes, dtype=float)
    n_dof = nodes.shape[0] if n_dof is None else int(n_dof)

    unknown = [name for name in elements if name not in _QUADRATURE]
    if unknown:
        raise ValueError(
            f"No PML quadrature for element types {unknown}. "
            f"Available: {sorted(_QUADRATURE)}."
        )
    if not elements:
        raise ValueError("No PML elements were given.")

    dims = {_QUADRATURE[name][1] for name in elements}
    if len(dims) > 1:
        raise ValueError(
            f"The PML groups mix {sorted(dims)}D elements; give the "
            "volume elements of the layer only."
        )
    dim = dims.pop()
    if dim != pml.dim:
        raise ValueError(
            f"The layer holds {dim}D elements but the CartesianPML is "
            f"{pml.dim}D."
        )

    rows, cols, entry_elem = [], [], []
    grad_vals, mass_vals, sigma_e = [], [], []
    elem_offset = 0

    for name, conn in elements.items():
        n_nodes, _, quadrature = _QUADRATURE[name]
        conn = np.asarray(conn, dtype=int)
        if conn.shape[1] != n_nodes:
            raise ValueError(
                f"'{name}' should have {n_nodes} nodes per element, got "
                f"{conn.shape[1]}."
            )

        for local, elem in enumerate(conn):
            idx = elem - 1
            x_e = nodes[idx]

            K_dir = np.zeros((dim, n_nodes, n_nodes))
            M_e = np.zeros((n_nodes, n_nodes))
            for N, grad, weight in quadrature(x_e):
                for j in range(dim):
                    K_dir[j] += np.outer(grad[j], grad[j]) * weight
                M_e += np.outer(N, N) * weight

            # sigma at the centroid: constant over the element, which is
            # what lets the frequency factors be applied afterwards.
            centroid = x_e[:, :dim].mean(axis=0)
            sigma_e.append(pml.sigma(centroid)[0])

            rr, cc = np.meshgrid(idx, idx, indexing="ij")
            rows.append(rr.ravel())
            cols.append(cc.ravel())
            entry_elem.append(np.full(n_nodes * n_nodes,
                                      elem_offset + local))
            grad_vals.append(
                np.stack([K_dir[j].ravel() for j in range(dim)], axis=1)
            )
            mass_vals.append(M_e.ravel())

        elem_offset += conn.shape[0]

    return PMLOperator(
        n_dof=n_dof,
        dim=dim,
        c0=float(c0),
        sigma_e=np.array(sigma_e, dtype=float),
        _rows=np.concatenate(rows),
        _cols=np.concatenate(cols),
        _entry_elem=np.concatenate(entry_elem),
        _grad_vals=np.concatenate(grad_vals, axis=0),
        _mass_vals=np.concatenate(mass_vals),
    )
