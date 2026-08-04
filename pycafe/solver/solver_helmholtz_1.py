import numpy as np
from pycafe.build_matrices.element_cquad8 import (
    element_matrices_cquad8,
    gauss_rule_quad_2x2,
    gauss_rule_quad_3x3,
    cquad8_shape,
    jacobian_2d, B_damping, calcola_line,linear_shape_1d,jacobian_1d, 
)
from scipy.sparse.linalg import spsolve
try:
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
except Exception:
    sp = None
    spla = None


def _as_1d_int(a, name):
    a = np.asarray(a, dtype=int).ravel()
    if a.size == 0:
        return a
    if np.any(a < 0):
        raise ValueError(f"{name} contains negative indices.")
    return a


def _as_1d_complex(a, name, n_expected=None):
    # accetta scalare o array
    if np.isscalar(a):
        if n_expected is None:
            raise ValueError(f"{name} is scalar but n_expected is None.")
        return np.full(n_expected, complex(a), dtype=complex)

    a = np.asarray(a, dtype=complex).ravel()
    if n_expected is not None and a.size != n_expected:
        raise ValueError(f"{name} has size {a.size}, expected {n_expected}.")
    return a


def _solve_linear(A, b):
    if sp is not None and sp.issparse(A):
        return spla.spsolve(A, b)
    else:
        return np.linalg.solve(A, b)

def build_normal_velocity_rhs(
    nodes,
    boundary_velocity_nodes,
    idx_free,
    rho,
    omega,
    v_n,
    boundaries=None,
    elements=None,
    groups=None,
    boundary_dim=None,
):
    """
    Assemble the right-hand side vector due to prescribed normal velocity.

    The Neumann condition ``v_n = vbar_n`` on ``Omega_v`` enters the
    dynamic system as the load vector

    .. math::
        V_{n,a} = -j \\rho_0 \\omega \\int_{\\Omega_v} N_a \\, \\bar v_n \\, d\\Omega ,

    consistent with a ``exp(+j omega t)`` time dependence and with
    ``vbar_n`` measured along the **outward** normal of the fluid
    domain: a wall moving into the fluid has ``vbar_n < 0``.

    The integral is evaluated on the actual boundary elements of the
    mesh -- edges for a 2D fluid, faces for a 3D one -- so the same call
    works for CQUAD4/CQUAD8 models and for HEXA8 cavities.

    Parameters
    ----------
    nodes : ndarray of shape (N, 2) or (N, 3)
        Coordinates of the mesh nodes.
    boundary_velocity_nodes : list of str or array-like of int
        Boundary selection: physical-group names (resolved through
        ``boundaries``/``groups``) or 1-based node tags.
    idx_free : ndarray of int
        Indices of the free degrees of freedom in the reduced system
        (0-based indexing).
    rho : float
        Fluid density.
    omega : float
        Angular frequency (rad/s).
    v_n : float, complex or callable
        Prescribed outward normal velocity. A callable ``v_n(omega)``
        describes a frequency-dependent excitation.
    boundaries : dict, optional
        ``{name: [1-based node tags]}``. Required when the selection is
        given by name.
    elements : dict, optional
        Mesh connectivity, used to locate the boundary elements.
    groups : dict, optional
        Physical groups with connectivity (``load_mesh_with_groups``);
        the authoritative source for 3D boundary faces.
    boundary_dim : int, optional
        1 for edges, 2 for faces. Inferred from ``elements`` if omitted.

    Returns
    -------
    velocity_nodes_red : ndarray of int
        Indices of the reduced degrees of freedom carrying a velocity
        contribution.
    velocity_values : ndarray of complex
        Corresponding right-hand side values.

    Raises
    ------
    ValueError
        If boundary names are given but cannot be resolved.

    Notes
    -----
    Only the free degrees of freedom are returned: contributions on
    nodes eliminated by a Dirichlet condition are reaction terms and do
    not act on the reduced system.

    For a frequency sweep, build a
    :class:`~pycafe.boundary_condition.acoustic_bc.NormalVelocityOperator`
    once instead of calling this function per frequency: the boundary
    integral does not depend on ``omega``.

    See Also
    --------
    solve_helmholtz_frequency_sweep : Frequency-domain acoustic solver.
    build_impedance_matrix : Assembly of impedance boundary contributions.
    prepare_acoustic_system : High-level acoustic FEM system preparation.
    """
    from pycafe.boundary_condition.acoustic_bc import (
        AcousticBC,
        build_velocity_operator,
    )

    if boundary_velocity_nodes is None or len(boundary_velocity_nodes) == 0:
        return np.array([], dtype=int), np.array([], dtype=complex)

    bc = AcousticBC().add_velocity(boundary_velocity_nodes, v_n)
    operator = build_velocity_operator(
        bc,
        nodes=nodes,
        rho=rho,
        boundaries=boundaries,
        groups=groups,
        elements=elements,
        boundary_dim=boundary_dim,
    )

    if operator.is_empty:
        return np.array([], dtype=int), np.array([], dtype=complex)

    idx_free = np.asarray(idx_free, dtype=int)
    f_red = operator.reduce(idx_free).at(omega)

    active = np.nonzero(f_red)[0]
    return active.astype(int), f_red[active]



def solve_helmholtz_single_frequency(
    K_red,
    M_red,
    C_red,
    omega,
    pressure_nodes_red,
    pressure_values,
    velocity_nodes_red=None,
    velocity_values=None,
    boundaries=None,
    f_red=None,
):
    """
    Solve the reduced acoustic Helmholtz problem at a single frequency.

    This function solves the complex-valued linear system

    .. math::
        (K + j \\omega C - \\omega^2 M) \\, p = f

    where prescribed pressure boundary conditions are enforced directly
    and optional normal velocity excitations contribute to the right-hand
    side vector.

    Parameters
    ----------
    K_red : scipy.sparse matrix
        Reduced acoustic stiffness matrix.
    M_red : scipy.sparse matrix
        Reduced acoustic mass matrix.
    C_red : scipy.sparse matrix
        Reduced acoustic impedance (damping) matrix.
    omega : float
        Angular frequency (rad/s).
    pressure_nodes_red : array-like of int
        Indices of reduced degrees of freedom where pressure is prescribed.
    pressure_values : array-like of complex
        Prescribed pressure values corresponding to ``pressure_nodes_red``.
    velocity_nodes_red : array-like of int, optional
        Indices of reduced degrees of freedom where a normal velocity
        excitation is applied.
    velocity_values : array-like of complex, optional
        Complex right-hand-side values associated with the velocity
        excitation.
    boundaries : dict, optional
        Boundary definition dictionary. Included for interface consistency;
        not used directly in this function.
    f_red : array-like of complex, optional
        Complete right-hand side of the reduced system, of length
        ``K_red.shape[0]``. Added to the velocity contribution; this is
        the path used by the frequency sweep, which assembles the load
        vector once and only re-scales it per frequency.

    Returns
    -------
    p_red : ndarray of complex, shape (N_red,)
        Complex acoustic pressure solution in the reduced system.

    Raises
    ------
    RuntimeError
        If no unknown degrees of freedom remain after applying
        prescribed pressure boundary conditions.

    Notes
    -----
    Prescribed pressure boundary conditions are enforced using a
    partitioned system approach, separating known and unknown
    degrees of freedom.

    Velocity excitations are incorporated into the right-hand side
    and are assumed to already include all geometric and physical
    integration effects.

    See Also
    --------
    solve_helmholtz_frequency_sweep : Solve the Helmholtz problem
        over a frequency range.
    build_normal_velocity_rhs : Assembly of velocity-induced RHS terms.
    prepare_acoustic_system : High-level FEM system preparation.
    """

    N = K_red.shape[0]
    A = K_red + 1j * omega * C_red - (omega**2) * M_red

    pressure_nodes_red = np.asarray(pressure_nodes_red, dtype=int)
    pressure_values = np.asarray(pressure_values, dtype=complex)

    # DOF split
    all_dofs = np.arange(N)
    mask_known = np.zeros(N, dtype=bool)
    mask_known[pressure_nodes_red] = True

    known = all_dofs[mask_known]
    unknown = all_dofs[~mask_known]

    if unknown.size == 0:
        raise RuntimeError("No unknown DOFs left (system fully constrained).")

    # RHS on the full reduced system, then restricted to the unknown
    # DOFs. Rows of prescribed pressure are dropped: there the load is
    # balanced by the acoustic reaction P, which never enters the
    # reduced system.
    rhs_full = np.zeros(N, dtype=complex)

    if f_red is not None:
        rhs_full += np.asarray(f_red, dtype=complex).ravel()

    if velocity_nodes_red is not None and velocity_values is not None:
        np.add.at(
            rhs_full,
            np.asarray(velocity_nodes_red, dtype=int),
            np.asarray(velocity_values, dtype=complex),
        )

    rhs = rhs_full[unknown]

    # Matrix blocks
    A_ii = A[np.ix_(unknown, unknown)]
    A_in = A[np.ix_(unknown, known)]

    rhs_eff = rhs - A_in @ pressure_values
    
    # Solve
    p_unknown = spsolve(A_ii, rhs_eff)

    # Recompose
    p_red = np.zeros(N, dtype=complex)
    p_red[known] = pressure_values
    p_red[unknown] = p_unknown

    return p_red

def solve_helmholtz_frequency_sweep(
    K_red,
    M_red,
    C_red,
    frequencies,
    pressure_nodes_red,
    pressure_values,
    *,
    nodes=None,
    boundary_velocity_nodes=None,
    idx_free=None,
    rho=1.2,
    v_n=None,
    boundaries=None,
    elements=None,
    groups=None,
    boundary_dim=None,
    velocity_operator=None,
    impedance_operator=None,
    source_operator=None,
):
    """
    Solve the acoustic Helmholtz problem over a frequency range.

    Handles the three non-essential boundary conditions of the dynamic
    response:

    - prescribed normal velocity (Neumann), through the load vector
      ``V_n(omega) = -j rho0 omega vbar_n int N dOmega``;
    - impedance (Robin), through the boundary matrix ``C(omega)``,
      which may be frequency dependent;
    - prescribed pressure (Dirichlet), enforced by partitioning.

    Parameters
    ----------
    K_red, M_red, C_red : sparse matrices
        Reduced acoustic stiffness, mass and damping matrices.
        ``C_red`` is ignored when ``impedance_operator`` is given.
    frequencies : array-like of float
        Frequencies of the sweep [Hz].
    pressure_nodes_red : array-like of int
        Reduced DOFs with prescribed pressure.
    pressure_values : array-like of complex
        Prescribed pressure values.
    nodes : ndarray, optional
        Nodal coordinates; needed to assemble a velocity boundary.
    boundary_velocity_nodes : list of str or list of int, optional
        Velocity boundary selection (names or 1-based node tags).
    idx_free : ndarray of int, optional
        Global indices of the retained DOFs, used to map the load
        vector onto the reduced system.
    rho : float, optional
        Fluid density. Default 1.2.
    v_n : float, complex or callable, optional
        Prescribed outward normal velocity; may depend on ``omega``.
    boundaries, elements, groups, boundary_dim : optional
        Mesh information used to resolve the boundary elements; see
        :func:`build_normal_velocity_rhs`.
    velocity_operator : NormalVelocityOperator, optional
        Pre-assembled load operator **already reduced** to the free
        DOFs. Overrides ``boundary_velocity_nodes``/``v_n``.
    impedance_operator : ImpedanceOperator, optional
        Pre-assembled damping operator **already reduced** to the free
        DOFs. Required for a frequency-dependent impedance; otherwise
        the constant ``C_red`` is used.
    source_operator : AcousticSourceOperator, optional
        Pre-assembled volumetric source (monopole) operator **already
        reduced** to the free DOFs; its ``Q(omega)`` is added to the
        right-hand side.

    Returns
    -------
    P_red : ndarray of complex, shape (N_red, N_freq)
        Reduced pressure solution at each frequency.

    Notes
    -----
    The boundary integrals do not depend on frequency, so they are
    assembled once before the loop and only re-scaled inside it.

    See Also
    --------
    solve_helmholtz_single_frequency : Single-frequency solve.
    pycafe.boundary_condition.acoustic_bc : Boundary condition operators.
    """
    from pycafe.boundary_condition.acoustic_bc import (
        AcousticBC,
        build_velocity_operator,
    )

    frequencies = np.atleast_1d(frequencies)
    N_red = K_red.shape[0]
    N_freq = len(frequencies)

    # --- assemble the velocity boundary once (geometry is frequency
    #     independent; only the -j rho omega vbar factor changes) ---
    if (velocity_operator is None
            and boundary_velocity_nodes is not None
            and v_n is not None
            and len(boundary_velocity_nodes) > 0):
        bc = AcousticBC().add_velocity(boundary_velocity_nodes, v_n)
        operator = build_velocity_operator(
            bc,
            nodes=nodes,
            rho=rho,
            boundaries=boundaries,
            groups=groups,
            elements=elements,
            boundary_dim=boundary_dim,
        )
        if not operator.is_empty and idx_free is not None:
            velocity_operator = operator.reduce(idx_free)
        elif not operator.is_empty:
            velocity_operator = operator

    P_red = np.zeros((N_red, N_freq), dtype=complex)

    for i, f in enumerate(frequencies):
        omega = 2 * np.pi * f

        f_red = (velocity_operator.at(omega)
                 if velocity_operator is not None else None)

        if source_operator is not None and not source_operator.is_empty:
            f_src = source_operator.at(omega)
            f_red = f_src if f_red is None else f_red + f_src

        C_omega = (impedance_operator.at(omega)
                   if impedance_operator is not None else C_red)

        P_red[:, i] = solve_helmholtz_single_frequency(
            K_red,
            M_red,
            C_omega,
            omega,
            pressure_nodes_red,
            pressure_values,
            f_red=f_red,
        )

    return P_red
