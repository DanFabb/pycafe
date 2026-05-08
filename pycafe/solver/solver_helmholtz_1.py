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
):
    """
    Assemble the right-hand side vector due to prescribed normal velocity.

    This function computes the contribution to the reduced acoustic
    right-hand side (RHS) vector associated with a prescribed normal
    velocity boundary condition. The contribution is obtained by
    integrating the term ``i * omega * rho * v_n`` along the specified
    boundary using a one-dimensional finite element formulation.

    Parameters
    ----------
    nodes : ndarray of shape (N, 2) or (N, 3)
        Coordinates of the mesh nodes. For 2D acoustic problems,
        only the (x, y) coordinates are used.
    boundary_velocity_nodes : array-like of int or list of str
        Boundary definition for the normal velocity condition.
        Can be either:
        - A list of node indices (1-based), or
        - A list of boundary names, which are resolved using ``boundaries``.
    idx_free : ndarray of int
        Indices of the free degrees of freedom in the reduced system
        (0-based indexing).
    rho : float
        Fluid density.
    omega : float
        Angular frequency (rad/s).
    v_n : float
        Prescribed normal velocity amplitude.
    boundaries : dict, optional
        Dictionary mapping boundary names to node lists (1-based indexing).
        Required if ``boundary_velocity_nodes`` contains boundary names.

    Returns
    -------
    velocity_nodes_red : ndarray of int
        Indices of the reduced degrees of freedom where the velocity
        contribution is applied.
    velocity_values : ndarray of complex
        Complex RHS values corresponding to the velocity contribution,
        integrated along the boundary.

    Raises
    ------
    ValueError
        If boundary names are provided but ``boundaries`` is not specified,
        or if a boundary name is not found.

    Notes
    -----
    The velocity contribution is assembled using linear shape functions
    along boundary edges and Gaussian quadrature. Boundary node indices
    are internally converted from 1-based to 0-based indexing.

    If no velocity boundary is specified, empty arrays are returned.

    See Also
    --------
    solve_helmholtz_frequency_sweep : Frequency-domain acoustic solver.
    build_impedance_matrix : Assembly of impedance boundary contributions.
    prepare_acoustic_system : High-level acoustic FEM system preparation.
    """
    if len(boundary_velocity_nodes) == 0:
        return np.array([], dtype=int), np.array([], dtype=complex)

    if isinstance(boundary_velocity_nodes[0], str):
        if boundaries is None:
            raise ValueError(
                "boundary_velocity_nodes contains boundary names, "
                "but 'boundaries' was not provided."
            )

        node_list = []
        for name in boundary_velocity_nodes:
            if name not in boundaries:
                raise ValueError(
                    f"Velocity BC boundary '{name}' not found. "
                    f"Available: {list(boundaries.keys())}"
                )
            node_list.extend(boundaries[name])

        boundary_velocity_nodes = np.array(node_list, dtype=int) - 1
    else:
        boundary_velocity_nodes = np.array(boundary_velocity_nodes, dtype=int) - 1

    # Build O(1) lookup from global node index → reduced index
    idx_free_set = set(idx_free.tolist())
    idx_free_map = {int(g): i for i, g in enumerate(idx_free)}

    velocity_nodes_red = []
    velocity_values = []

    if elements is not None and 'Line 3' in elements:
        # -------------------------------------------------------
        # Quadratic Line3 integration using actual mesh boundary
        # elements.  This correctly includes corner nodes shared
        # with adjacent boundaries (Gmsh assigns corners to only
        # one physical group, so they would otherwise be missed).
        # -------------------------------------------------------
        bnd_set_0based = set(boundary_velocity_nodes.tolist())

        # 3-point Gauss rule: exact for polynomials up to degree 5
        sqrt35 = np.sqrt(3.0 / 5.0)
        Xi3 = np.array([-sqrt35, 0.0,  sqrt35])
        W3  = np.array([ 5./9., 8./9., 5./9.])

        for elem in elements['Line 3']:
            n1_1b = int(elem[0]); n2_1b = int(elem[1]); n3_1b = int(elem[2])
            n1 = n1_1b - 1; n2 = n2_1b - 1; n3 = n3_1b - 1

            # Keep element if either corner belongs to the velocity boundary.
            # This picks up the two end elements whose other corner is in an
            # adjacent physical group (e.g. 'bottom'/'top').
            if n1 not in bnd_set_0based and n2 not in bnd_set_0based:
                continue

            x1 = nodes[n1, :2]
            x2 = nodes[n2, :2]
            x3 = nodes[n3, :2]

            for xi, w in zip(Xi3, W3):
                # Lagrange quadratic shape functions on ξ ∈ [-1, 1]
                # nodes at ξ = -1 (n1), +1 (n2), 0 (n3)
                Nv = np.array([
                    xi * (xi - 1.0) / 2.0,   # N1
                    xi * (xi + 1.0) / 2.0,   # N2
                    1.0 - xi * xi,             # N3
                ])
                # Jacobian: dx/dξ
                dN = np.array([
                    (2.0*xi - 1.0) / 2.0,
                    (2.0*xi + 1.0) / 2.0,
                    -2.0 * xi,
                ])
                dxy = dN[0]*x1 + dN[1]*x2 + dN[2]*x3
                detJ = np.linalg.norm(dxy)

                contrib = 1j * omega * rho * v_n * Nv * detJ * w

                for a, gnode in enumerate([n1, n2, n3]):
                    if gnode in idx_free_set:
                        velocity_nodes_red.append(idx_free_map[gnode])
                        velocity_values.append(contrib[a])

    else:
        # -------------------------------------------------------
        # Fallback: Line2 integration with spatially sorted nodes.
        # Gmsh may return boundary node lists in arbitrary order,
        # so we sort along the dominant boundary direction first.
        # -------------------------------------------------------
        bc_coords = nodes[boundary_velocity_nodes, :2]
        x_range = bc_coords[:, 0].max() - bc_coords[:, 0].min()
        y_range = bc_coords[:, 1].max() - bc_coords[:, 1].min()
        sort_axis = 1 if y_range >= x_range else 0
        sort_idx = np.argsort(bc_coords[:, sort_axis])
        boundary_velocity_nodes = boundary_velocity_nodes[sort_idx]

        GAUSS_POINT = 1.0 / np.sqrt(3.0)
        Xi = np.array([-GAUSS_POINT, GAUSS_POINT])

        for e in range(len(boundary_velocity_nodes) - 1):
            n1 = boundary_velocity_nodes[e]
            n2 = boundary_velocity_nodes[e + 1]

            x_start = nodes[n1, :2]
            x_end   = nodes[n2, :2]

            _, _, x1D_e = calcola_line(x_start, x_end)

            for xi in Xi:
                N, dN_dxi = linear_shape_1d(xi)
                _, detJ, _ = jacobian_1d(dN_dxi, x1D_e)

                contrib = 1j * omega * rho * v_n * N * detJ * 1.0

                for a, gnode in enumerate([n1, n2]):
                    if gnode in idx_free_set:
                        velocity_nodes_red.append(idx_free_map[gnode])
                        velocity_values.append(contrib[a])

    return (
        np.array(velocity_nodes_red, dtype=int),
        np.array(velocity_values, dtype=complex),
    )



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
):
    """
    Solve the reduced acoustic Helmholtz problem at a single frequency.

    This function solves the complex-valued linear system

    .. math::
        (K + i \\, \omega C - \omega^2 M) \\, p = f

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

    # RHS from velocity
    rhs = np.zeros(len(unknown), dtype=complex)

    if velocity_nodes_red is not None:
        for node, val in zip(velocity_nodes_red, velocity_values):
            if node in unknown:
                i = np.where(unknown == node)[0][0]
                rhs[i] += val

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
):
    """
    Frequency sweep Helmholtz solver with:
    - prescribed pressure (Dirichlet)
    - optional imposed normal velocity on boundaries

    Returns
    -
    P_red : (N_red, N_freq) complex ndarray
    """

    frequencies = np.atleast_1d(frequencies)
    N_red = K_red.shape[0]
    N_freq = len(frequencies)

    P_red = np.zeros((N_red, N_freq), dtype=complex)

    for i, f in enumerate(frequencies):
        omega = 2 * np.pi * f

        # VELOCITY RHS (automatic)
        if boundary_velocity_nodes is not None and v_n is not None:
            velocity_nodes_red, velocity_values = build_normal_velocity_rhs(
                nodes=nodes,
                boundary_velocity_nodes=boundary_velocity_nodes,
                idx_free=idx_free,
                rho=rho,
                omega=omega,
                v_n=v_n,
                boundaries=boundaries,
                elements=elements,
            )
        else:
            velocity_nodes_red = None
            velocity_values = None
        

        # SOLVE SINGLE FREQUENCY
        P_red[:, i] = solve_helmholtz_single_frequency(
            K_red,
            M_red,
            C_red,
            omega,
            pressure_nodes_red,
            pressure_values,
            velocity_nodes_red,
            velocity_values,
        )

    return P_red
