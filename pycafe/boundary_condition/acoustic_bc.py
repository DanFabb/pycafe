# acoustic_bc.py
#
# Declarative description of the acoustic boundary conditions, and the
# operators that turn them into the terms of the dynamic system
#
#     (K + j omega C - omega^2 M) p = Q + V_n + P .
#
# Three of the four boundary condition types live here:
#
#   * prescribed normal velocity (Neumann)  -> V_n, a right-hand side
#   * impedance / admittance     (Robin)    -> C, a boundary matrix
#   * rigid wall                            -> nothing (homogeneous Neumann)
#
# The fourth, prescribed pressure (Dirichlet), is an essential condition
# handled by direct assignment and DOF elimination in
# :mod:`pycafe.build_matrices.bc_ops`; it is only *described* here.
#
# Sign convention
# ---------------
# Time dependence exp(+j omega t), and the normal velocity is measured
# along the **outward** normal of the fluid domain:
#
#     dp/dn = -j rho0 omega v_n         =>    V_n,a = -j rho0 omega int N_a vbar_n dOmega
#
# A boundary moving into the fluid therefore has vbar_n < 0.
from dataclasses import dataclass, field
from typing import Callable, List, Sequence, Union

import numpy as np

from pycafe.build_matrices.bc_surface import (
    boundary_integrals,
    resolve_boundary_faces,
)

Selection = Union[Sequence[str], Sequence[int], str]
Value = Union[complex, float, Callable]

# Unit tags accepted for an impedance value.
_NORMALIZED_UNITS = {"normalized", "norm", "n", "zeta", "ratio", "relative"}
_ABSOLUTE_UNITS = {"abs", "absolute", "si", "rayl", "rayls", "specific",
                   "pa*s/m", "pas/m"}


# ============================================================
#  IMPEDANCE / ADMITTANCE
# ============================================================
def make_admittance(Z, rho, c0):
    """
    Build the specific admittance ``A(omega) = 1 / Z(omega)``.

    Parameters
    ----------
    Z : complex, callable, tuple or None
        Impedance specification. Accepted forms:

        - ``Z = 3.0`` or ``Z = 1 - 2j`` : **normalized** impedance
          ``zeta``, i.e. the absolute impedance is ``zeta * rho * c0``.
          This is the legacy pyCAFE convention.
        - ``Z = (1200.0, "abs")`` : absolute specific impedance in
          Pa*s/m. Any tag from ``{"abs", "absolute", "si", "rayl",
          "specific"}`` selects this reading; ``{"normalized", "zeta",
          ...}`` forces the normalized one.
        - ``Z = f`` with ``f(omega)`` returning an impedance : a
          frequency-dependent liner. Interpreted as normalized unless
          wrapped as ``(f, "abs")``.
        - ``Z = 0`` or ``Z = None`` : rigid wall, no contribution.
    rho : float
        Fluid density.
    c0 : float
        Speed of sound, used to de-normalize ``zeta``.

    Returns
    -------
    admittance : callable or None
        ``admittance(omega) -> complex`` in m/(Pa*s), or ``None`` for a
        rigid wall (which contributes nothing to ``C``).

    Raises
    ------
    ValueError
        If the unit tag is unknown, or if the impedance evaluates to
        zero (a zero impedance means infinite admittance).

    Notes
    -----
    The normalized default keeps existing scripts working: a bare
    number has always meant ``Z / (rho * c0)`` in pyCAFE.
    """
    unit = "normalized"

    if isinstance(Z, tuple) and len(Z) == 2 and isinstance(Z[1], str):
        Z, unit = Z[0], Z[1].strip().lower()
        if unit in _ABSOLUTE_UNITS:
            unit = "absolute"
        elif unit in _NORMALIZED_UNITS:
            unit = "normalized"
        else:
            raise ValueError(
                f"Unknown impedance unit '{unit}'. Use one of "
                f"{sorted(_ABSOLUTE_UNITS | _NORMALIZED_UNITS)}."
            )

    if Z is None:
        return None
    if not callable(Z) and Z == 0:
        # Legacy meaning of Z = 0: rigid wall, no impedance boundary.
        return None

    scale = 1.0 if unit == "absolute" else rho * c0

    def admittance(omega):
        value = Z(omega) if callable(Z) else Z
        Z_spec = complex(value) * scale
        if Z_spec == 0:
            raise ValueError(
                "Impedance evaluated to zero (infinite admittance) at "
                f"omega = {omega}."
            )
        return 1.0 / Z_spec

    return admittance


# ============================================================
#  BOUNDARY CONDITION ENTRIES
# ============================================================
@dataclass
class VelocityBC:
    """
    Prescribed normal velocity on a portion of the boundary.

    Attributes
    ----------
    selection : list of str or list of int
        Physical-group names, or 1-based node tags.
    v_n : complex or callable
        Normal velocity along the **outward** normal [m/s]. A callable
        ``v_n(omega)`` describes a frequency-dependent excitation.
    """
    selection: Selection
    v_n: Value = 0.0


@dataclass
class ImpedanceBC:
    """
    Impedance (Robin) condition on a portion of the boundary.

    Attributes
    ----------
    selection : list of str or list of int
        Physical-group names, or 1-based node tags.
    Z : complex, callable or tuple
        Impedance specification; see :func:`make_admittance`.
    """
    selection: Selection
    Z: Value = 0.0


@dataclass
class PressureBC:
    """Prescribed pressure (Dirichlet) on a portion of the boundary."""
    selection: Selection
    value: complex = 0.0


@dataclass
class MonopoleSource:
    """
    Volumetric acoustic source ``Q_a = j rho0 omega int N_a q dV``.

    Exactly one of the three placements must be given:

    Attributes
    ----------
    position : array-like, optional
        Coordinates of a point monopole ``q(x) = q delta(x - x_pos)``;
        ``q`` is then the source volume velocity [m^3/s]. The source is
        distributed over the containing element through its shape
        functions.
    node : int, optional
        0-based node index of a point monopole placed exactly at a
        node: ``Q_i = j rho0 omega q``.
    distributed : bool
        If True, ``q`` is a volumetric source density [1/s] constant
        over the whole acoustic domain: ``Q_a = j rho0 omega q int N_a dV``.
    q : complex or callable
        Source strength; a callable ``q(omega)`` gives a
        frequency-dependent source.
    """
    q: Value = 0.0
    position: Sequence[float] = None
    node: int = None
    distributed: bool = False


@dataclass
class PointPressureBC:
    """Prescribed pressure at a single node (0-based index)."""
    node: int
    value: complex = 0.0


@dataclass
class AcousticBC:
    """
    Complete acoustic boundary condition set of a model.

    Unlike the legacy 9-element tuple, which carried a single impedance
    value and a single normal velocity for the whole model, every entry
    here holds its own value, so different walls can have different
    liners or different velocities.

    Attributes
    ----------
    pressure_zero : list of str
        Boundaries where ``p = 0`` (eliminated from the system).
    pressure_constant : list of PressureBC
        Boundaries with a prescribed non-zero pressure.
    impedance : list of ImpedanceBC
        Boundaries with an impedance condition.
    velocity : list of VelocityBC
        Boundaries with a prescribed normal velocity.
    point_pressure : list of PointPressureBC
        Nodes with a prescribed pressure.
    monopoles : list of MonopoleSource
        Volumetric acoustic sources (point or distributed).

    See Also
    --------
    make_admittance : Impedance specification rules.
    build_impedance_operator : Assemble ``C(omega)``.
    build_velocity_operator : Assemble ``V_n(omega)``.
    build_source_operator : Assemble ``Q(omega)``.
    """
    pressure_zero: List[str] = field(default_factory=list)
    pressure_constant: List[PressureBC] = field(default_factory=list)
    impedance: List[ImpedanceBC] = field(default_factory=list)
    velocity: List[VelocityBC] = field(default_factory=list)
    point_pressure: List[PointPressureBC] = field(default_factory=list)
    monopoles: List[MonopoleSource] = field(default_factory=list)

    # ---------------- construction helpers ----------------
    def add_rigid_wall(self, selection):
        """No-op, kept for readability: a rigid wall adds no term."""
        return self

    def add_velocity(self, selection, v_n):
        """Add a prescribed normal velocity (outward positive)."""
        self.velocity.append(VelocityBC(_as_list(selection), v_n))
        return self

    def add_impedance(self, selection, Z):
        """Add an impedance condition; see :func:`make_admittance`."""
        self.impedance.append(ImpedanceBC(_as_list(selection), Z))
        return self

    def add_anechoic(self, selection):
        """
        Add a non-reflecting (anechoic) boundary on an artificial border.

        This is the plane-wave approximation of the Sommerfeld radiation
        condition, used to truncate an **exterior** problem: the infinite
        fluid domain ``V`` is split by an artificial surface ``Omega_e``
        into a meshed region ``V1`` (body -> ``Omega_e``) and the
        unmeshed remainder ``V2`` (``Omega_e`` -> infinity). On
        ``Omega_e`` the fluid is asked to behave as a progressive plane
        wave leaving the domain,

        .. math:: p = \\rho_0 c_0 \\, v_n ,

        i.e. an impedance equal to the characteristic impedance of the
        fluid, ``Z = rho0 * c0`` (normalized ``zeta = 1``). It is exactly
        the same as ``add_impedance(selection, 1.0)``, named for what it
        does.

        Parameters
        ----------
        selection : str or list of str or list of int
            The artificial boundary ``Omega_e``: physical-group names, or
            1-based node tags.

        Notes
        -----
        The condition is exact only for a plane wave hitting
        ``Omega_e`` at normal incidence. Spherical, oblique or
        evanescent components have ``p / v_n != rho0 * c0`` and are
        partly reflected back into ``V1``. Push ``Omega_e`` far enough
        from the radiating body that the field is locally plane there
        -- as a rule of thumb, at least one wavelength at the lowest
        frequency of interest, and beyond the reactive near field -- at
        the price of a larger mesh.

        With the ``exp(+j omega t)`` convention used by the solver, the
        contribution ``+j omega C = j k S`` is dissipative, so an
        outgoing wave is absorbed rather than amplified.

        See Also
        --------
        add_impedance : General (possibly frequency-dependent) liner.
        """
        return self.add_impedance(selection, 1.0)

    # Same condition, spelled the way the radiation problem names it.
    add_radiation = add_anechoic
    add_non_reflecting = add_anechoic

    def add_pressure(self, selection, value=0.0):
        """Add a prescribed pressure (zero-pressure if ``value == 0``)."""
        if value == 0:
            self.pressure_zero.extend(_as_list(selection))
        else:
            self.pressure_constant.append(PressureBC(_as_list(selection), value))
        return self

    def add_point_pressure(self, node, value):
        """Add a prescribed pressure at a single (0-based) node."""
        self.point_pressure.append(
            PointPressureBC(check_node_index(node, "point pressure"), value)
        )
        return self

    def add_monopole(self, q, *, position=None, node=None):
        """
        Add a point monopole of volume velocity ``q`` [m^3/s].

        Place it either at an arbitrary ``position`` (distributed over
        the containing element via its shape functions) or exactly at a
        0-based ``node``.
        """
        if (position is None) == (node is None):
            raise ValueError("Give exactly one of 'position' or 'node'.")
        if node is not None:
            node = check_node_index(node, "monopole")
        self.monopoles.append(
            MonopoleSource(q=q, position=position, node=node)
        )
        return self

    def add_distributed_source(self, q):
        """Add a source density ``q`` [1/s] constant over the domain."""
        self.monopoles.append(MonopoleSource(q=q, distributed=True))
        return self

    # ---------------- legacy interoperability ----------------
    @classmethod
    def from_legacy(cls, bc):
        """
        Build an :class:`AcousticBC` from the legacy 9-element tuple.

        The tuple layout is the one returned by
        :func:`pycafe.core.assign_boundary_conditions.assign_boundary_conditions`.
        An :class:`AcousticBC` instance is passed through unchanged.
        """
        if isinstance(bc, cls):
            return bc

        (
            pressure_zero,
            pressure_constant,
            pressure_constant_value,
            impedance_selection,
            Z_impedance,
            velocity_selection,
            value_velocity_normal,
            source_node,
            source_pressure_value,
        ) = bc

        obj = cls(pressure_zero=list(pressure_zero or []))

        if pressure_constant:
            obj.pressure_constant.append(
                PressureBC(_as_list(pressure_constant), pressure_constant_value)
            )
        if impedance_selection is not None and len(impedance_selection):
            obj.impedance.append(
                ImpedanceBC(_as_list(impedance_selection), Z_impedance)
            )
        if velocity_selection is not None and len(velocity_selection):
            obj.velocity.append(
                VelocityBC(_as_list(velocity_selection), value_velocity_normal)
            )
        if source_node is not None:
            obj.point_pressure.append(
                PointPressureBC(int(source_node), source_pressure_value or 0.0)
            )

        return obj

    def to_legacy(self):
        """
        Collapse back to the legacy 9-element tuple.

        Lossy by construction: the legacy format holds one impedance and
        one velocity value for the whole model, so only the first entry
        of each list survives, and callables cannot be represented.

        Raises
        ------
        ValueError
            If the boundary conditions cannot be expressed in the legacy
            format (several distinct values, or frequency-dependent ones).
        """
        def _single(entries, attr, label):
            if not entries:
                return [], 0.0
            values = {getattr(e, attr) for e in entries
                      if not callable(getattr(e, attr))}
            if any(callable(getattr(e, attr)) for e in entries):
                raise ValueError(
                    f"Frequency-dependent {label} cannot be expressed as a "
                    "legacy BC tuple."
                )
            if len(values) > 1:
                raise ValueError(
                    f"The legacy BC tuple holds a single {label} value, but "
                    f"{len(values)} distinct ones are defined."
                )
            selection = []
            for e in entries:
                selection.extend(e.selection)
            return selection, values.pop()

        p_sel, p_val = _single(self.pressure_constant, "value", "pressure")
        z_sel, z_val = _single(self.impedance, "Z", "impedance")
        v_sel, v_val = _single(self.velocity, "v_n", "normal velocity")

        if self.monopoles:
            raise ValueError(
                "Monopole sources cannot be expressed as a legacy BC tuple."
            )
        if len(self.point_pressure) > 1:
            raise ValueError(
                "The legacy BC tuple holds a single point source, but "
                f"{len(self.point_pressure)} are defined."
            )
        src_node = self.point_pressure[0].node if self.point_pressure else None
        src_val = self.point_pressure[0].value if self.point_pressure else 0.0

        return (
            list(self.pressure_zero),
            p_sel, p_val,
            z_sel, z_val,
            v_sel, v_val,
            src_node, src_val,
        )


def _as_list(selection):
    """Normalize a selection to a list (a bare name becomes a 1-list)."""
    if isinstance(selection, str):
        return [selection]
    return list(selection)


def check_node_index(node, what, n_dof=None):
    """
    Validate a 0-based node index and return it as an ``int``.

    Node selectors go straight into NumPy indexing, where a negative
    value is legal and silently addresses a node counted from the end:
    ``node=-1`` would put the source on the last node of the mesh
    instead of failing. Out-of-range values are rejected too when the
    mesh size is known.
    """
    idx = int(node)
    if idx < 0:
        raise ValueError(
            f"The {what} node index must be 0-based and non-negative; "
            f"got {node}."
        )
    if n_dof is not None and idx >= n_dof:
        raise ValueError(
            f"The {what} node index {idx} is outside the mesh, which has "
            f"{n_dof} nodes (valid range 0..{n_dof - 1})."
        )
    return idx


# ============================================================
#  OPERATORS
# ============================================================
class ImpedanceOperator:
    """
    Frequency-dependent acoustic damping matrix ``C(omega)``.

    Assembles ``C(omega) = rho0 * sum_b A_b(omega) * S_b``, where
    ``S_b`` is the boundary mass matrix of the b-th impedance boundary
    and ``A_b`` its specific admittance. Keeping the geometry (``S_b``)
    and the material (``A_b``) apart means a frequency sweep only
    re-scales already assembled matrices.

    Parameters
    ----------
    n_dof : int
        Size of the acoustic system.
    terms : list of (sparse matrix, callable)
        Pairs ``(S_b, admittance_b)``.
    rho : float
        Fluid density.

    See Also
    --------
    build_impedance_operator : Build one from an :class:`AcousticBC`.
    """

    def __init__(self, n_dof, terms, rho):
        self.n_dof = int(n_dof)
        self.terms = list(terms)
        self.rho = float(rho)
        self._cache = None

    @property
    def shape(self):
        return (self.n_dof, self.n_dof)

    @property
    def is_empty(self):
        """True when no impedance boundary is active."""
        return len(self.terms) == 0

    @property
    def is_frequency_dependent(self):
        """True when at least one admittance depends on frequency."""
        return not self._is_constant()

    def _is_constant(self):
        """
        Probe the admittances at two frequencies to detect constancy.

        A constant ``C`` can be assembled once and reused across the
        whole sweep; a liner defined by a callable cannot.
        """
        if not self.terms:
            return True
        for _, adm in self.terms:
            try:
                if not np.isclose(adm(1.0), adm(1234.0)):
                    return False
            except Exception:
                # An admittance that fails to evaluate at a probe
                # frequency is assumed frequency dependent.
                return False
        return True

    def at(self, omega):
        """
        Assemble ``C`` at the given angular frequency.

        Returns
        -------
        C : scipy.sparse.csr_matrix of complex, shape (n_dof, n_dof)
        """
        from scipy.sparse import csr_matrix

        if self._cache is not None:
            return self._cache

        C = csr_matrix(self.shape, dtype=complex)
        for S, admittance in self.terms:
            C = C + (self.rho * admittance(omega)) * S.astype(complex)

        if self._is_constant():
            self._cache = C
        return C

    def reduce(self, idx_free):
        """Restrict the operator to the retained (free) degrees of freedom."""
        idx_free = np.asarray(idx_free, dtype=int)
        terms = [(S[np.ix_(idx_free, idx_free)], adm)
                 for S, adm in self.terms]
        return ImpedanceOperator(idx_free.size, terms, self.rho)


class NormalVelocityOperator:
    """
    Frequency-dependent normal-velocity load vector ``V_n(omega)``.

    Assembles ``V_n(omega) = -j rho0 omega * sum_b vbar_b(omega) * g_b``
    with ``g_b`` the boundary load vector of the b-th velocity boundary.
    The velocity is positive along the **outward** normal of the fluid.

    Parameters
    ----------
    n_dof : int
        Size of the acoustic system.
    terms : list of (ndarray, callable)
        Pairs ``(g_b, v_b)``; ``v_b(omega)`` returns the normal velocity.
    rho : float
        Fluid density.

    See Also
    --------
    build_velocity_operator : Build one from an :class:`AcousticBC`.
    """

    def __init__(self, n_dof, terms, rho):
        self.n_dof = int(n_dof)
        self.terms = list(terms)
        self.rho = float(rho)

    @property
    def shape(self):
        return (self.n_dof,)

    @property
    def is_empty(self):
        """True when no velocity boundary is active."""
        return len(self.terms) == 0

    def at(self, omega):
        """
        Assemble the load vector at the given angular frequency.

        Returns
        -------
        f : ndarray of complex, shape (n_dof,)
        """
        f = np.zeros(self.n_dof, dtype=complex)
        for g, v_n in self.terms:
            f += -1j * self.rho * omega * complex(v_n(omega)) * g
        return f

    def reduce(self, idx_free):
        """Restrict the operator to the retained (free) degrees of freedom."""
        idx_free = np.asarray(idx_free, dtype=int)
        terms = [(g[idx_free], v) for g, v in self.terms]
        return NormalVelocityOperator(idx_free.size, terms, self.rho)


class AcousticSourceOperator:
    """
    Frequency-dependent volumetric source vector ``Q(omega)``.

    Assembles ``Q(omega) = +j rho0 omega * sum_s q_s(omega) * g_s``,
    where ``g_s`` holds the shape function weights of the s-th source:
    ``N_a(x_s)`` for a point monopole, ``int N_a dV`` for a distributed
    density. Note the sign: the manual's convention gives the source
    term ``+j rho0 omega`` (opposite to the outward-velocity load).

    Parameters
    ----------
    n_dof : int
        Size of the acoustic system.
    terms : list of (ndarray, callable)
        Pairs ``(g_s, q_s)``; ``q_s(omega)`` returns the strength.
    rho : float
        Fluid density.

    See Also
    --------
    build_source_operator : Build one from an :class:`AcousticBC`.
    NormalVelocityOperator : The boundary counterpart ``V_n(omega)``.
    """

    def __init__(self, n_dof, terms, rho):
        self.n_dof = int(n_dof)
        self.terms = list(terms)
        self.rho = float(rho)

    @property
    def shape(self):
        return (self.n_dof,)

    @property
    def is_empty(self):
        """True when no source is active."""
        return len(self.terms) == 0

    def at(self, omega):
        """
        Assemble the source vector at the given angular frequency.

        Returns
        -------
        f : ndarray of complex, shape (n_dof,)
        """
        f = np.zeros(self.n_dof, dtype=complex)
        for g, q in self.terms:
            f += 1j * self.rho * omega * complex(q(omega)) * g
        return f

    def reduce(self, idx_free):
        """Restrict the operator to the retained (free) degrees of freedom."""
        idx_free = np.asarray(idx_free, dtype=int)
        terms = [(g[idx_free], q) for g, q in self.terms]
        return AcousticSourceOperator(idx_free.size, terms, self.rho)


def _as_callable(value):
    """Wrap a constant into a frequency function, leave callables alone."""
    if callable(value):
        return value
    constant = complex(value)
    return lambda omega: constant


# ============================================================
#  BUILDERS
# ============================================================
def build_impedance_operator(
    bc,
    *,
    nodes,
    rho,
    c0,
    boundaries=None,
    groups=None,
    elements=None,
    boundary_dim=None,
):
    """
    Assemble the impedance operator ``C(omega)`` from a BC description.

    Parameters
    ----------
    bc : AcousticBC or legacy tuple
        Boundary condition description.
    nodes : ndarray of shape (N, 2) or (N, 3)
        Nodal coordinates.
    rho, c0 : float
        Fluid density and speed of sound.
    boundaries : dict, optional
        ``{name: [1-based node tags]}`` from the mesh loader.
    groups : dict, optional
        Physical groups with connectivity (``load_mesh_with_groups``).
    elements : dict, optional
        Mesh connectivity, used to find the boundary faces/edges.
    boundary_dim : int, optional
        1 for edges, 2 for faces. Inferred from ``elements`` if omitted.

    Returns
    -------
    ImpedanceOperator

    See Also
    --------
    ImpedanceOperator, make_admittance
    """
    bc = AcousticBC.from_legacy(bc)
    n_dof = np.asarray(nodes).shape[0]

    terms = []
    for entry in bc.impedance:
        admittance = make_admittance(entry.Z, rho, c0)
        if admittance is None:
            continue                      # rigid wall: nothing to add
        faces = resolve_boundary_faces(
            entry.selection,
            nodes=nodes,
            boundaries=boundaries,
            groups=groups,
            elements=elements,
            boundary_dim=boundary_dim,
        )
        if not faces:
            continue
        _, S = boundary_integrals(nodes, faces, n_dof)
        terms.append((S, admittance))

    return ImpedanceOperator(n_dof, terms, rho)


def build_velocity_operator(
    bc,
    *,
    nodes,
    rho,
    boundaries=None,
    groups=None,
    elements=None,
    boundary_dim=None,
):
    """
    Assemble the normal-velocity operator ``V_n(omega)``.

    Parameters
    ----------
    bc : AcousticBC or legacy tuple
        Boundary condition description.
    nodes : ndarray of shape (N, 2) or (N, 3)
        Nodal coordinates.
    rho : float
        Fluid density.
    boundaries, groups, elements, boundary_dim
        Same meaning as in :func:`build_impedance_operator`.

    Returns
    -------
    NormalVelocityOperator

    See Also
    --------
    NormalVelocityOperator
    """
    bc = AcousticBC.from_legacy(bc)
    n_dof = np.asarray(nodes).shape[0]

    terms = []
    for entry in bc.velocity:
        if not callable(entry.v_n) and entry.v_n == 0:
            continue                      # rigid wall
        faces = resolve_boundary_faces(
            entry.selection,
            nodes=nodes,
            boundaries=boundaries,
            groups=groups,
            elements=elements,
            boundary_dim=boundary_dim,
        )
        if not faces:
            continue
        g, _ = boundary_integrals(nodes, faces, n_dof)
        terms.append((g, _as_callable(entry.v_n)))

    return NormalVelocityOperator(n_dof, terms, rho)


def build_source_operator(
    bc,
    *,
    nodes,
    rho,
    elements=None,
):
    """
    Assemble the volumetric source operator ``Q(omega)``.

    Parameters
    ----------
    bc : AcousticBC or legacy tuple
        Boundary condition description; only its ``monopoles`` list is
        used here.
    nodes : ndarray of shape (N, 2) or (N, 3)
        Nodal coordinates.
    rho : float
        Fluid density.
    elements : dict, optional
        Mesh connectivity, needed to locate a monopole placed at an
        arbitrary position and to integrate a distributed density.
        Not needed for sources placed exactly at a node.

    Returns
    -------
    AcousticSourceOperator

    Raises
    ------
    ValueError
        If a source needs the mesh (position or distributed) but
        ``elements`` was not provided, if the point lies outside the
        mesh, or if a nodal monopole names a node outside it.

    See Also
    --------
    AcousticSourceOperator
    pycafe.build_matrices.source_volume.point_source_shape
    pycafe.build_matrices.source_volume.volume_load_vector
    """
    from pycafe.build_matrices.source_volume import (
        point_source_shape,
        volume_load_vector,
    )

    bc = AcousticBC.from_legacy(bc)
    nodes = np.asarray(nodes)
    n_dof = nodes.shape[0]

    terms = []
    for src in bc.monopoles:
        if not callable(src.q) and src.q == 0:
            continue

        g = np.zeros(n_dof, dtype=float)

        if src.distributed:
            if elements is None:
                raise ValueError(
                    "A distributed source needs the mesh 'elements'."
                )
            g = volume_load_vector(nodes, elements, n_dof)

        elif src.node is not None:
            node = check_node_index(src.node, "monopole", n_dof)
            g[node] = 1.0                 # N_a(x_i) = delta_ai

        else:
            if elements is None:
                raise ValueError(
                    "A monopole at an arbitrary position needs the "
                    "mesh 'elements' to locate its element."
                )
            idx, N = point_source_shape(nodes, elements, src.position)
            g[idx] = N

        terms.append((g, _as_callable(src.q)))

    return AcousticSourceOperator(n_dof, terms, rho)
