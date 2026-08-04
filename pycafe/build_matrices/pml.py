"""
Perfectly Matched Layer for the acoustic elements, in 2D and 3D.

A PML truncates an unbounded domain by surrounding it with a layer in
which one coordinate is continued into the complex plane. With the
``exp(+i omega t)`` convention used throughout pyCAFE an outgoing wave
is ``exp(-i k x)``, so with

.. math:: \\gamma = 1 - i\\,\\frac{\\sigma}{k}, \\qquad k = \\omega / c_0 ,

the wave inside the layer becomes ``exp(-ikx) exp(-int sigma)``: it
decays, and does so without reflecting at the interface, because the
stretching is analytic there.

Written in the weak form, every layer in this module has the same shape,

.. math::
    \\int_{V_e} (\\nabla q)^T \\Lambda\\, \\nabla p \\; dV
    \\;+\\; \\mu \\int_{V_e} p\\,q \\; dV ,

with ``Lambda`` diagonal **in a frame attached to the element** and
``mu`` a scalar. What changes between a flat layer and a curved one is
only the frame and the two coefficients:

============  ===================================  =========================
layer         ``Lambda`` (diagonal entries)         ``mu``
============  ===================================  =========================
Cartesian     ``prod_m gamma_m / gamma_j^2``        ``-k^2 prod_m gamma_m``
spherical     ``rt^2/(gamma_r r^2)``, ``gamma_r``   ``-k^2 rt^2 gamma_r/r^2``
              (radial, then both tangents)
cylindrical   ``rt/(gamma_r r)``,                   ``-k^2 rt gamma_r / r``
              ``gamma_r r/rt``, ``rt gamma_r/r``
============  ===================================  =========================

``rt`` is the stretched radius ``r - (i/k) int_R^r sigma``; the curved
layers need it because the Jacobian of the map carries the radius, while
the flat one does not. Setting ``sigma = 0`` gives ``Lambda = I`` and
``mu = -k^2`` in every case, i.e. the ordinary ``K - omega^2 M``.

Note that the stretching enters the stiffness **once**: the two
gradients are those of the trial and test function, and only the
operator between them is stretched. Nothing is conjugated — the result
is meant to be complex, and the assembled matrix is complex symmetric,
not Hermitian.

Frequency dependence
--------------------
The coefficients are rational in ``omega``, so a PML cannot be split
into a finite set of frequency-independent matrices the way
``K - omega^2 M`` is. What can be done, and is done here, is to
integrate once and rescale: the geometry is evaluated at the element
centroid, so the frame and the profile are constant over the element and
the coefficients multiply integrals that never change.
:meth:`PMLOperator.matrix` rebuilds the sparse matrix at each frequency
without touching the quadrature.

Taking the profile at the centroid makes it piecewise constant across
the layer. It is a choice of this implementation, not a property of the
method.

Getting one from a mesh
-----------------------
:func:`pml_from_groups` reads the layer off a Gmsh mesh: a physical
group named ``pml`` next to the ``fluid`` one is the whole switch, and
the shape of the layer — box, cylindrical shell, spherical shell — its
interface and its thickness are measured from the two groups. The only
number that cannot be measured is how much the layer should absorb; that
is ``target_reflection``.
"""

import warnings
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from .element_cquad4 import cquad4_shape, gauss_rule_quad_2x2
from .element_cquad8 import cquad8_shape, gauss_rule_quad_3x3, jacobian_2d
from .element_hex8 import gauss_rule_hex_2x2x2, hex8_shape, jacobian_3d
from .element_tetra4 import tetra4_geometry


# ---------------------------------------------------------------------------
#  Absorption profile
# ---------------------------------------------------------------------------

@dataclass
class PowerProfile:
    """
    Polynomial absorption ``sigma(d) = sigma0 (d/L)^n`` across a layer.

    Parameters
    ----------
    sigma0 : float
        Absorption at the outer face, in **1/m**: it is added to the
        wavenumber, so it carries its units.
    thickness : float
        Layer thickness ``L`` [m].
    order : int, optional
        Exponent ``n``. Default 2.
    """

    sigma0: float
    thickness: float
    order: int = 2

    def __post_init__(self):
        self.sigma0 = float(self.sigma0)
        self.thickness = float(self.thickness)
        if self.sigma0 < 0.0:
            raise ValueError(f"Negative sigma0: {self.sigma0}.")
        if self.thickness < 0.0:
            raise ValueError(f"Negative thickness: {self.thickness}.")
        if self.order < 0:
            raise ValueError(f"Negative profile order: {self.order}.")

    def sigma(self, depth):
        """Absorption at a depth into the layer [m], clipped to the face."""
        return self.sigma0 * self._normalized(depth) ** self.order

    def integral(self, depth):
        """
        ``int_0^d sigma ds``, the quantity the curved layers stretch the
        radius with.
        """
        u = self._normalized(depth)
        return self.sigma0 * self.thickness * u ** (self.order + 1) \
            / (self.order + 1)

    def _normalized(self, depth):
        d = np.maximum(np.asarray(depth, float), 0.0)
        if self.thickness <= 0.0:
            return np.zeros_like(d)
        return np.clip(d / self.thickness, 0.0, 1.0)

    @classmethod
    def for_reflection(cls, thickness, target_reflection, order=2):
        """
        The profile whose round trip attenuates by ``target_reflection``.

        A plane wave at normal incidence crosses the layer, reflects off
        whatever closes it and comes back, so its amplitude is divided by
        ``exp(2 int_0^L sigma) = exp(2 sigma0 L / (n+1))``. Inverting
        that gives ``sigma0``.

        This is what the layer would reflect if it were solved exactly.
        The mesh sets a floor under it that no ``sigma0`` gets past: the
        profile is piecewise constant across the elements, and past some
        absorption the wave starts reflecting off those steps instead of
        being absorbed by them. Asking for less than the layer can
        deliver therefore makes the reflection **worse**, not better.
        :func:`layer_resolution` reports how many elements the profile
        is spread over.
        """
        if not 0.0 < target_reflection < 1.0:
            raise ValueError(
                f"target_reflection must lie in (0, 1), got "
                f"{target_reflection}."
            )
        if thickness <= 0.0:
            raise ValueError(f"Non-positive thickness: {thickness}.")
        sigma0 = -(order + 1) * np.log(target_reflection) / (2.0 * thickness)
        return cls(sigma0=sigma0, thickness=thickness, order=order)


# ---------------------------------------------------------------------------
#  Layer geometries
# ---------------------------------------------------------------------------

# Below this many element layers, the piecewise-constant profile is a
# staircase rather than a ramp. The number is a chosen warning
# threshold, not a property of the method.
MIN_LAYER_STEPS = 4


def _unit(v, axis=-1):
    norm = np.linalg.norm(v, axis=axis, keepdims=True)
    return np.divide(v, norm, out=np.zeros_like(v), where=norm > 0.0)


@dataclass
class CartesianPML:
    """
    A box-shaped layer wrapped around a rectangular physical region.

    The physical region is ``[inner_min, inner_max]``; outside it, along
    each axis, the layer extends by ``thickness`` and is stretched with
    its own profile. An axis with zero thickness is not stretched, which
    is how a layer on one side only — the end of a duct — is described.

    Parameters
    ----------
    inner_min, inner_max : array_like of shape (dim,)
        Bounds of the physical region; ``dim`` is 2 or 3.
    profiles : sequence of PowerProfile
        One per axis. An axis with ``thickness = 0`` is inert.
    """

    inner_min: np.ndarray
    inner_max: np.ndarray
    profiles: tuple

    def __post_init__(self):
        self.inner_min = np.atleast_1d(np.asarray(self.inner_min, float))
        self.inner_max = np.atleast_1d(np.asarray(self.inner_max, float))
        if self.inner_min.size not in (2, 3):
            raise ValueError(
                f"A PML needs 2 or 3 dimensions, got {self.inner_min.size}."
            )
        if self.inner_max.size != self.inner_min.size:
            raise ValueError(
                f"inner_min has {self.inner_min.size} components, "
                f"inner_max {self.inner_max.size}."
            )
        if np.any(self.inner_max <= self.inner_min):
            raise ValueError(
                f"inner_max must exceed inner_min on every axis, got "
                f"{self.inner_min} .. {self.inner_max}."
            )
        self.profiles = tuple(self.profiles)
        if len(self.profiles) != self.dim:
            raise ValueError(
                f"{self.dim} axes but {len(self.profiles)} profiles."
            )

    @property
    def dim(self):
        return self.inner_min.size

    def frames(self, centroids):
        """The Cartesian axes, the same for every element."""
        n = np.asarray(centroids).shape[0]
        return np.broadcast_to(np.eye(self.dim), (n, self.dim, self.dim))

    def sigma(self, points):
        """
        Absorption ``sigma_j`` [1/m] at the given points.

        Returns
        -------
        ndarray of shape (npts, dim)
        """
        pts = np.atleast_2d(np.asarray(points, float))[:, :self.dim]
        depth = np.maximum(
            np.maximum(self.inner_min - pts, pts - self.inner_max), 0.0
        )
        return np.column_stack([
            p.sigma(depth[:, j]) for j, p in enumerate(self.profiles)
        ])

    def depth(self, points):
        """How far into the layer a point lies [m], along the deepest axis."""
        pts = np.atleast_2d(np.asarray(points, float))[:, :self.dim]
        return np.maximum(
            np.maximum(self.inner_min - pts, pts - self.inner_max), 0.0
        ).max(axis=1)

    @property
    def thickness(self):
        return max(p.thickness for p in self.profiles)

    def prepare(self, centroids):
        return {"sigma": self.sigma(centroids)}

    def coefficients(self, data, k):
        gamma = 1.0 - 1j * data["sigma"] / k
        prod = np.prod(gamma, axis=1)
        return prod[:, None] / gamma ** 2, -(k ** 2) * prod


@dataclass
class RadialPML:
    """
    A shell-shaped layer, spherical in 3D and circular in 2D.

    The physical region is the ball of radius ``inner_radius`` around
    ``center``; the layer is the shell outside it. Only the radial
    direction is stretched, and the frame of each element is its own
    radial direction plus any orthonormal completion — the tangential
    coefficient is the same in every tangential direction, so the choice
    does not matter.

    Parameters
    ----------
    center : array_like of shape (dim,)
    inner_radius : float
    profile : PowerProfile
    """

    center: np.ndarray
    inner_radius: float
    profile: PowerProfile

    def __post_init__(self):
        self.center = np.atleast_1d(np.asarray(self.center, float))
        if self.center.size not in (2, 3):
            raise ValueError(
                f"A PML needs 2 or 3 dimensions, got {self.center.size}."
            )
        self.inner_radius = float(self.inner_radius)
        if self.inner_radius <= 0.0:
            raise ValueError(
                f"Non-positive inner radius: {self.inner_radius}."
            )

    @property
    def dim(self):
        return self.center.size

    def _radial(self, centroids):
        pts = np.asarray(centroids, float)[:, :self.dim]
        offset = pts - self.center
        r = np.linalg.norm(offset, axis=1)
        return r, _unit(offset)

    def frames(self, centroids):
        r, r_hat = self._radial(centroids)
        _check_radius(r, "centre")
        return np.stack([r_hat] + _tangents(r_hat), axis=1)

    def depth(self, points):
        """How far past the inner radius a point lies [m]."""
        r, _ = self._radial(np.atleast_2d(np.asarray(points, float)))
        return np.maximum(r - self.inner_radius, 0.0)

    @property
    def thickness(self):
        return self.profile.thickness

    def prepare(self, centroids):
        r, _ = self._radial(centroids)
        depth = np.maximum(r - self.inner_radius, 0.0)
        return {"r": r,
                "sigma": self.profile.sigma(depth),
                "integral": self.profile.integral(depth)}

    def coefficients(self, data, k):
        r = data["r"]
        gamma = 1.0 - 1j * data["sigma"] / k
        r_tilde = r - 1j * data["integral"] / k

        if self.dim == 3:
            jacobian = r_tilde ** 2 * gamma / r ** 2
            radial = r_tilde ** 2 / (gamma * r ** 2)
            tangential = gamma
            lambdas = np.column_stack([radial, tangential, tangential])
        else:
            jacobian = r_tilde * gamma / r
            radial = r_tilde / (gamma * r)
            tangential = gamma * r / r_tilde
            lambdas = np.column_stack([radial, tangential])

        return lambdas, -(k ** 2) * jacobian


@dataclass
class CylindricalPML:
    """
    A tube-shaped layer: a circular shell in the plane normal to
    ``axis``, unstretched along the axis itself.

    This is the 3D "annular" layer — the physical region is a cylinder
    and the layer wraps its curved side, leaving the two flat ends to
    whatever boundary condition they carry.

    Parameters
    ----------
    center : array_like of shape (3,)
        Any point on the axis.
    axis : array_like of shape (3,)
        Direction of the axis; normalized internally.
    inner_radius : float
    profile : PowerProfile
    """

    center: np.ndarray
    axis: np.ndarray
    inner_radius: float
    profile: PowerProfile

    def __post_init__(self):
        self.center = np.asarray(self.center, float).ravel()
        self.axis = np.asarray(self.axis, float).ravel()
        if self.center.size != 3 or self.axis.size != 3:
            raise ValueError("A cylindrical PML is a 3D layer.")
        norm = np.linalg.norm(self.axis)
        if norm == 0.0:
            raise ValueError("The cylinder axis cannot be the zero vector.")
        self.axis = self.axis / norm
        self.inner_radius = float(self.inner_radius)
        if self.inner_radius <= 0.0:
            raise ValueError(
                f"Non-positive inner radius: {self.inner_radius}."
            )

    @property
    def dim(self):
        return 3

    def _radial(self, centroids):
        offset = np.asarray(centroids, float)[:, :3] - self.center
        along = offset @ self.axis
        perp = offset - along[:, None] * self.axis
        return np.linalg.norm(perp, axis=1), _unit(perp)

    def frames(self, centroids):
        rho, rho_hat = self._radial(centroids)
        _check_radius(rho, "axis")
        theta_hat = np.cross(self.axis, rho_hat)
        axis = np.broadcast_to(self.axis, rho_hat.shape)
        return np.stack([rho_hat, theta_hat, axis], axis=1)

    def depth(self, points):
        """How far past the inner radius a point lies [m]."""
        rho, _ = self._radial(np.atleast_2d(np.asarray(points, float)))
        return np.maximum(rho - self.inner_radius, 0.0)

    @property
    def thickness(self):
        return self.profile.thickness

    def prepare(self, centroids):
        rho, _ = self._radial(centroids)
        depth = np.maximum(rho - self.inner_radius, 0.0)
        return {"r": rho,
                "sigma": self.profile.sigma(depth),
                "integral": self.profile.integral(depth)}

    def coefficients(self, data, k):
        rho = data["r"]
        gamma = 1.0 - 1j * data["sigma"] / k
        rho_tilde = rho - 1j * data["integral"] / k

        jacobian = rho_tilde * gamma / rho
        lambdas = np.column_stack([
            rho_tilde / (gamma * rho),      # radial
            gamma * rho / rho_tilde,        # circumferential
            jacobian,                       # along the axis
        ])
        return lambdas, -(k ** 2) * jacobian


def _check_radius(r, what):
    """A layer cannot contain its own centre or axis: there is no frame."""
    if np.any(r <= 0.0):
        raise ValueError(
            f"{int(np.sum(r <= 0.0))} PML element(s) sit on the {what} of "
            "the layer, where the radial direction is undefined. A shell "
            "layer must not contain it."
        )


def _tangents(r_hat):
    """Two unit vectors completing ``r_hat`` into a frame (3D), one (2D)."""
    if r_hat.shape[1] == 2:
        return [np.column_stack([-r_hat[:, 1], r_hat[:, 0]])]

    # Cross with whichever axis is least aligned, so the result never
    # degenerates.
    helper = np.zeros_like(r_hat)
    helper[np.arange(len(r_hat)), np.argmin(np.abs(r_hat), axis=1)] = 1.0
    t1 = _unit(np.cross(r_hat, helper))
    return [t1, np.cross(r_hat, t1)]


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
    One point would be exact for the gradients, which are constant, but
    the mass integrand is quadratic — hence the four-point rule of
    degree 2 on the reference tetrahedron.
    """
    volume, grad = tetra4_geometry(np.asarray(x_e, float)[:, :3])
    a, b = 0.585410196624969, 0.138196601125011      # (5 +- sqrt(5))/20
    for N in np.array([[a, b, b, b], [b, a, b, b],
                       [b, b, a, b], [b, b, b, a]]):
        yield N, grad, volume / 4.0


_QUADRATURE = {
    # The rules match the element kernels: a distorted quad integrates
    # differently under 2x2 and 3x3, and the sigma = 0 limit has to come
    # out equal to K - omega^2 M entry by entry.
    "Quadrilateral 4": (4, 2, _quadrature_quad(cquad4_shape,
                                               gauss_rule_quad_2x2)),
    "Quadrangle 4": (4, 2, _quadrature_quad(cquad4_shape,
                                            gauss_rule_quad_2x2)),
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
    The contribution of the PML elements, ready at any frequency.

    This **replaces** the ordinary ``K - omega^2 M`` of those elements;
    it is not a correction added on top. Assemble the rest of the mesh
    without them.

    Attributes
    ----------
    n_dof : int
    dim : int
    c0 : float
    layer : CartesianPML, RadialPML or CylindricalPML
    """

    n_dof: int
    dim: int
    c0: float
    layer: object
    _data: dict = field(repr=False)
    _depth_extent: np.ndarray = field(repr=False)
    _rows: np.ndarray = field(repr=False)
    _cols: np.ndarray = field(repr=False)
    _entry_elem: np.ndarray = field(repr=False)
    _grad_vals: np.ndarray = field(repr=False)
    _mass_vals: np.ndarray = field(repr=False)

    @property
    def n_elements(self):
        return int(self._entry_elem.max()) + 1

    @property
    def element_depth_extent(self):
        """
        How far each element spans along the depth of the layer [m].

        The size that matters for the profile: an element long in the
        depth direction takes a big step in ``sigma``, however small it
        is across.
        """
        return self._depth_extent

    @property
    def sigma_e(self):
        """
        Absorption of each element, at its centroid [1/m].

        Shape ``(n_elem, dim)`` for a Cartesian layer, one column per
        axis; ``(n_elem,)`` for a curved one, where only the radial
        direction is stretched.
        """
        return self._data["sigma"]

    def coefficients(self, omega):
        """
        The per-element coefficients at one frequency.

        Returns
        -------
        lambdas : ndarray (n_elem, dim), complex
            Diagonal of ``Lambda`` in the element frame.
        mu : ndarray (n_elem,), complex
            Coefficient of ``int N N``.
        """
        if omega <= 0.0:
            raise ValueError(
                f"A PML is defined at a positive frequency, got "
                f"omega = {omega}. The stretching gamma = 1 - i sigma/k "
                "is singular at omega = 0."
            )
        return self.layer.coefficients(self._data, omega / self.c0)

    def matrix(self, omega):
        """
        Assemble the PML contribution at one angular frequency.

        Returns
        -------
        scipy.sparse.csr_matrix, complex, shape (n_dof, n_dof)
        """
        lambdas, mu = self.coefficients(omega)

        values = (self._grad_vals * lambdas[self._entry_elem]).sum(axis=1)
        values = values + self._mass_vals * mu[self._entry_elem]

        return sp.coo_matrix(
            (values, (self._rows, self._cols)),
            shape=(self.n_dof, self.n_dof),
        ).tocsr()

    def reduce(self, idx_free):
        """A view of the operator on the free degrees of freedom."""
        idx_free = np.asarray(idx_free, dtype=int)
        position = np.full(self.n_dof, -1, dtype=int)
        position[idx_free] = np.arange(idx_free.size)

        keep = (position[self._rows] >= 0) & (position[self._cols] >= 0)
        return PMLOperator(
            n_dof=idx_free.size,
            dim=self.dim,
            c0=self.c0,
            layer=self.layer,
            _data=self._data,
            _depth_extent=self._depth_extent,
            _rows=position[self._rows[keep]],
            _cols=position[self._cols[keep]],
            _entry_elem=self._entry_elem[keep],
            _grad_vals=self._grad_vals[keep],
            _mass_vals=self._mass_vals[keep],
        )


def build_pml_operator(nodes, elements, c0, layer, n_dof=None):
    """
    Integrate the PML elements once, for use at any frequency.

    Parameters
    ----------
    nodes : ndarray of shape (N, 2) or (N, 3)
        Nodal coordinates of the whole mesh.
    elements : dict
        Connectivity of the **PML elements only**, ``{gmsh name:
        (n_elem, n_nodes) array}``, 1-based. Typically the ``pml``
        physical group, from ``load_mesh_with_groups``.
    c0 : float
        Speed of sound.
    layer : CartesianPML, RadialPML or CylindricalPML
    n_dof : int, optional
        Size of the global system. Defaults to the number of nodes.

    Returns
    -------
    PMLOperator
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
    if dim != layer.dim:
        raise ValueError(
            f"The layer holds {dim}D elements but "
            f"{type(layer).__name__} is {layer.dim}D."
        )

    # ------------------------------------------------ element geometry
    centroids, conns = [], []
    for name, conn in elements.items():
        n_nodes = _QUADRATURE[name][0]
        conn = np.asarray(conn, dtype=int)
        if conn.shape[1] != n_nodes:
            raise ValueError(
                f"'{name}' should have {n_nodes} nodes per element, got "
                f"{conn.shape[1]}."
            )
        conns.append((name, conn))
        centroids.append(nodes[conn - 1][:, :, :dim].mean(axis=1))
    centroids = np.concatenate(centroids, axis=0)

    frames = layer.frames(centroids)
    data = layer.prepare(centroids)

    depth_extent = []
    for name, conn in conns:
        node_depth = layer.depth(nodes[conn - 1].reshape(-1, nodes.shape[1]))
        node_depth = node_depth.reshape(conn.shape)
        depth_extent.append(node_depth.max(axis=1) - node_depth.min(axis=1))
    depth_extent = np.concatenate(depth_extent)

    # ------------------------------------------------ quadrature
    rows, cols, entry_elem = [], [], []
    grad_vals, mass_vals = [], []
    offset = 0

    for name, conn in conns:
        n_nodes, _, quadrature = _QUADRATURE[name]
        for local, elem in enumerate(conn):
            idx = elem - 1
            frame = frames[offset + local]              # (dim, dim)

            K_dir = np.zeros((dim, n_nodes, n_nodes))
            M_e = np.zeros((n_nodes, n_nodes))
            for N, grad, weight in quadrature(nodes[idx]):
                directional = frame @ grad              # (dim, n_nodes)
                for j in range(dim):
                    K_dir[j] += np.outer(directional[j],
                                         directional[j]) * weight
                M_e += np.outer(N, N) * weight

            rr, cc = np.meshgrid(idx, idx, indexing="ij")
            rows.append(rr.ravel())
            cols.append(cc.ravel())
            entry_elem.append(np.full(n_nodes * n_nodes, offset + local))
            grad_vals.append(
                np.stack([K_dir[j].ravel() for j in range(dim)], axis=1)
            )
            mass_vals.append(M_e.ravel())
        offset += conn.shape[0]

    return PMLOperator(
        n_dof=n_dof,
        dim=dim,
        c0=float(c0),
        layer=layer,
        _data=data,
        _depth_extent=depth_extent,
        _rows=np.concatenate(rows),
        _cols=np.concatenate(cols),
        _entry_elem=np.concatenate(entry_elem),
        _grad_vals=np.concatenate(grad_vals, axis=0),
        _mass_vals=np.concatenate(mass_vals),
    )


# ---------------------------------------------------------------------------
#  Reading the layer off a mesh
# ---------------------------------------------------------------------------

def _group_nodes(group):
    """0-based node indices touched by a physical group."""
    tags = set()
    for conn in group["elements"].values():
        tags.update(np.asarray(conn, dtype=int).ravel().tolist())
    return np.array(sorted(tags), dtype=int) - 1


def _volume_elements(group, dim):
    """The ``dim``-dimensional element arrays of a group."""
    return {
        name: np.asarray(conn, dtype=int)
        for name, conn in group["elements"].items()
        if name in _QUADRATURE and _QUADRATURE[name][1] == dim
    }


@dataclass
class LayerShape:
    """What the geometry of a layer was measured to be."""

    kind: str                 # "box", "cylindrical" or "spherical"
    residual: float           # how well the interface fits that shape
    detail: str

    def __str__(self):
        return f"{self.kind} layer ({self.detail}, fit {self.residual:.2e})"


def _fit_sphere(interface, dim):
    """Centre and radius of the best sphere through the interface nodes."""
    # Linear least squares on |x|^2 - 2 c.x + |c|^2 = R^2.
    A = np.column_stack([2.0 * interface, np.ones(len(interface))])
    b = (interface ** 2).sum(axis=1)
    solution, *_ = np.linalg.lstsq(A, b, rcond=None)
    center = solution[:dim]
    radius = np.sqrt(max(solution[dim] + (center ** 2).sum(), 0.0))
    r = np.linalg.norm(interface - center, axis=1)
    residual = float(np.std(r) / max(np.mean(r), 1e-30))
    return center, radius, residual


def _fit_cylinder(interface, axis):
    """Centre and radius of the best circle in the plane normal to axis."""
    axis = _unit(np.asarray(axis, float).reshape(1, 3))[0]
    along = interface @ axis
    perp = interface - along[:, None] * axis
    center_perp, radius, residual = _fit_sphere(perp, 3)
    # Put the centre back on the axis through the middle of the span.
    center = center_perp - (center_perp @ axis) * axis \
        + 0.5 * (along.min() + along.max()) * axis
    rho = np.linalg.norm(
        (interface - center) - ((interface - center) @ axis)[:, None] * axis,
        axis=1,
    )
    residual = float(np.std(rho) / max(np.mean(rho), 1e-30))
    return center, float(np.mean(rho)), residual


def _fit_box(fluid_pts, interface, dim):
    """
    How well the interface sits on the faces of the physical box.

    The box is the bounding box of the **fluid**, not of the interface:
    a layer on one side only has a flat interface, whose own bounding
    box is degenerate along the direction that matters.
    """
    lo, hi = fluid_pts.min(axis=0), fluid_pts.max(axis=0)
    span = np.maximum(hi - lo, 1e-30)
    on_face = np.minimum(
        (interface - lo) / span, (hi - interface) / span
    ).min(axis=1)
    return lo, hi, float(np.max(np.abs(on_face)))


def identify_layer_shape(nodes, fluid_nodes, pml_nodes, axis_hint=None):
    """
    Work out the shape of a layer from the mesh alone.

    The interface is the set of nodes the fluid and the layer share. A
    box interface has every node on a face of its own bounding box; a
    spherical one has every node at the same distance from a centre; a
    cylindrical one at the same distance from an axis. Each candidate
    gets a dimensionless residual and the best one wins.

    Parameters
    ----------
    nodes : ndarray (N, 2) or (N, 3)
    fluid_nodes, pml_nodes : ndarray of int
        0-based node indices of the two groups.
    axis_hint : array_like of shape (3,), optional
        Axis to try for the cylindrical fit. Without it the three
        Cartesian axes are tried and the best kept.

    Returns
    -------
    shape : LayerShape
    parameters : dict
        Everything the corresponding layer class needs, except the
        profile.
    """
    nodes = np.asarray(nodes, float)
    dim = 2 if nodes.shape[1] == 2 else 3
    interface_idx = np.intersect1d(fluid_nodes, pml_nodes)
    if interface_idx.size < dim + 1:
        raise ValueError(
            f"The fluid and the PML share {interface_idx.size} nodes, too "
            "few to tell the shape of the layer from. Are the two groups "
            "conforming?"
        )
    interface = nodes[interface_idx][:, :dim]
    layer_pts = nodes[pml_nodes][:, :dim]
    fluid_pts = nodes[fluid_nodes][:, :dim]

    candidates = []

    lo, hi, box_residual = _fit_box(fluid_pts, interface, dim)
    candidates.append((
        box_residual, "box",
        {"inner_min": lo, "inner_max": hi},
        f"{np.round(lo, 4).tolist()} .. {np.round(hi, 4).tolist()}",
    ))

    center, radius, sphere_residual = _fit_sphere(interface, dim)
    if radius > 0.0:
        candidates.append((
            sphere_residual, "spherical",
            {"center": center, "inner_radius": radius},
            f"centre {np.round(center, 4).tolist()}, R = {radius:.4g}",
        ))

    if dim == 3:
        axes = ([np.asarray(axis_hint, float)] if axis_hint is not None
                else [np.eye(3)[j] for j in range(3)])
        for a in axes:
            c, rho, residual = _fit_cylinder(interface, a)
            candidates.append((
                residual, "cylindrical",
                {"center": c, "axis": _unit(a.reshape(1, 3))[0],
                 "inner_radius": rho},
                f"axis {np.round(_unit(a.reshape(1, 3))[0], 3).tolist()}, "
                f"R = {rho:.4g}",
            ))

    residual, kind, parameters, detail = min(candidates, key=lambda c: c[0])

    # Thickness: how far the layer reaches past the interface.
    if kind == "box":
        past = np.maximum(
            np.maximum(parameters["inner_min"] - layer_pts,
                       layer_pts - parameters["inner_max"]),
            0.0,
        ).max(axis=0)
        parameters["thickness"] = past
    elif kind == "spherical":
        r = np.linalg.norm(layer_pts - parameters["center"], axis=1)
        parameters["thickness"] = float(r.max() - parameters["inner_radius"])
    else:
        offset = layer_pts - parameters["center"]
        along = offset @ parameters["axis"]
        rho = np.linalg.norm(offset - along[:, None] * parameters["axis"],
                             axis=1)
        parameters["thickness"] = float(rho.max()
                                        - parameters["inner_radius"])

    return LayerShape(kind=kind, residual=residual, detail=detail), parameters


def layer_resolution(operator):
    """
    How many elements the absorption profile is spread over.

    The profile is constant on each element, so it climbs to ``sigma0``
    in as many steps as there are elements across the thickness. Few
    steps mean large jumps, and a wave reflects off a large jump in
    ``sigma`` however well the layer is matched analytically.

    The count is the thickness divided by the typical size of an element
    **measured along the depth direction** — not the number of distinct
    values of ``sigma``, which on an unstructured mesh is simply the
    number of elements, since no two centroids share a depth.

    Parameters
    ----------
    operator : PMLOperator

    Returns
    -------
    int
        Estimated number of element layers across the thickness, at
        least 1.
    """
    extent = operator.element_depth_extent
    typical = float(np.median(extent[extent > 0.0])) if np.any(extent > 0.0) \
        else 0.0
    thickness = float(getattr(operator.layer, "thickness", 0.0))
    if typical <= 0.0 or thickness <= 0.0:
        return 1
    return max(1, int(round(thickness / typical)))


def pml_from_groups(nodes, groups, c0, *, target_reflection=1e-4, order=2,
                    shape=None, axis_hint=None, n_dof=None, verbose=True):
    """
    Build the PML of a mesh that carries a ``pml`` physical group.

    Everything but the amount of absorption is measured from the mesh:
    which elements are the layer, what shape it has, where it meets the
    fluid and how thick it is. If there is no ``pml`` group, this
    returns ``None`` and the model is simply not absorbing.

    Parameters
    ----------
    nodes : ndarray (N, 2) or (N, 3)
    groups : dict
        Physical groups with connectivity, from
        ``load_mesh_with_groups``.
    c0 : float
        Speed of sound.
    target_reflection : float, optional
        Amplitude a plane wave keeps after crossing the layer, hitting
        whatever closes it and coming back. This is the one number the
        mesh cannot supply; it sets ``sigma0`` through
        :meth:`PowerProfile.for_reflection`. Default ``1e-4``.
    order : int, optional
        Exponent of the absorption profile. Default 2.
    shape : {"box", "spherical", "cylindrical"}, optional
        Force the shape instead of measuring it.
    axis_hint : array_like of shape (3,), optional
        Axis of a cylindrical layer, when it is not a Cartesian one.
    n_dof : int, optional
        Size of the global system. Defaults to the number of nodes.
    verbose : bool, optional
        Print what was measured. Default True.

    Returns
    -------
    PMLOperator or None

    Raises
    ------
    ValueError
        If the ``pml`` group holds no volume elements, if it shares
        elements with the fluid, or if a forced ``shape`` is unknown.
    """
    from pycafe.create_geom.conventions import names_for

    pml_names = names_for("pml")
    fluid_names = names_for("fluid")

    pml_group = next((g for g in groups if g.lower() in pml_names), None)
    if pml_group is None:
        return None

    fluid_group = next((g for g in groups if g.lower() in fluid_names), None)
    if fluid_group is None:
        raise ValueError(
            f"The mesh has a '{pml_group}' group but no fluid group "
            f"(one of {sorted(fluid_names)}); there is nothing for the "
            "layer to absorb from."
        )

    nodes = np.asarray(nodes, float)
    dim = 2 if nodes.shape[1] == 2 else 3

    pml_elements = _volume_elements(groups[pml_group], dim)
    if not pml_elements:
        raise ValueError(
            f"Physical group '{pml_group}' holds no {dim}D acoustic "
            f"elements; it contains "
            f"{sorted(groups[pml_group]['elements'])}."
        )
    fluid_elements = _volume_elements(groups[fluid_group], dim)

    for name, conn in pml_elements.items():
        shared = fluid_elements.get(name)
        if shared is None:
            continue
        rows = set(map(tuple, np.sort(conn, axis=1).tolist()))
        if rows & set(map(tuple, np.sort(shared, axis=1).tolist())):
            raise ValueError(
                f"Groups '{pml_group}' and '{fluid_group}' share "
                f"'{name}' elements. The layer replaces the ordinary "
                "matrices of its elements, so an element in both would "
                "be counted twice."
            )

    measured, parameters = identify_layer_shape(
        nodes,
        _group_nodes(groups[fluid_group]),
        _group_nodes(groups[pml_group]),
        axis_hint=axis_hint,
    )
    kind = measured.kind if shape is None else str(shape).lower()

    if kind == "box":
        thickness = np.atleast_1d(parameters["thickness"])
        profiles = tuple(
            PowerProfile(sigma0=0.0, thickness=0.0, order=order)
            if L <= 0.0 else
            PowerProfile.for_reflection(L, target_reflection, order)
            for L in thickness
        )
        layer = CartesianPML(inner_min=parameters["inner_min"],
                             inner_max=parameters["inner_max"],
                             profiles=profiles)
    elif kind in ("spherical", "radial"):
        layer = RadialPML(
            center=parameters["center"],
            inner_radius=parameters["inner_radius"],
            profile=PowerProfile.for_reflection(
                parameters["thickness"], target_reflection, order
            ),
        )
    elif kind == "cylindrical":
        layer = CylindricalPML(
            center=parameters["center"],
            axis=parameters["axis"],
            inner_radius=parameters["inner_radius"],
            profile=PowerProfile.for_reflection(
                parameters["thickness"], target_reflection, order
            ),
        )
    else:
        raise ValueError(
            f"Unknown layer shape '{shape}'. Use 'box', 'spherical' or "
            "'cylindrical', or leave it out to measure it."
        )

    operator = build_pml_operator(nodes, pml_elements, c0, layer,
                                  n_dof=n_dof)
    steps = layer_resolution(operator)

    if steps < MIN_LAYER_STEPS:
        warnings.warn(
            f"The PML is {steps} element(s) thick, so the absorption "
            f"jumps to its peak in {steps} step(s). A wave reflects off "
            "those steps, and asking for a smaller target_reflection "
            "makes that worse rather than better. Refine the layer, or "
            f"raise target_reflection above {target_reflection:g}.",
            RuntimeWarning,
            stacklevel=2,
        )

    if verbose:
        forced = "" if shape is None else " (forced)"
        print(f"PML from group '{pml_group}': {measured}{forced}, "
              f"thickness {np.round(parameters['thickness'], 4)}, "
              f"{steps} elements across, "
              f"target reflection {target_reflection:g}")

    return operator
