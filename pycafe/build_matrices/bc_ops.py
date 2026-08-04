# bc_ops.py
#
# Boundary-condition and reduction operations shared by all element
# types: Dirichlet reduction, pressure BC mapping, impedance matrix,
# expansion of reduced solutions. Moved here from the (previously
# duplicated) assembly_cquad4 / assembly_cquad8 modules.
import numpy as np
from scipy.sparse import lil_matrix


def reduce_KM_dirichlet(K_global, M_global, boundary_pressure_zero):


    # unique (come MATLAB)
    boundary_pressure_zero = np.unique(boundary_pressure_zero)

    K_reduced = K_global.copy()
    M_reduced = M_global.copy()

    # Rimozione righe
    K_reduced = np.delete(K_reduced, boundary_pressure_zero, axis=0)
    M_reduced = np.delete(M_reduced, boundary_pressure_zero, axis=0)

    # Rimozione colonne
    K_reduced = np.delete(K_reduced, boundary_pressure_zero, axis=1)
    M_reduced = np.delete(M_reduced, boundary_pressure_zero, axis=1)


    reduced_nodes = K_reduced.shape[0]

    return K_reduced, M_reduced, reduced_nodes

def reduce_KMC_dirichlet_mask(K_global, M_global, C, boundary_pressure_zero):
    """
    Reduce the acoustic system by enforcing zero-pressure (Dirichlet) conditions.

    This function reduces the global acoustic stiffness (K), mass (M),
    and impedance (C) matrices by imposing zero pressure on a set of
    boundary nodes. The reduction is performed using a boolean mask
    and index mapping, without modifying the original matrices.

    Parameters
    ----------
    K_global : ndarray or sparse matrix of shape (N, N)
        Global acoustic stiffness matrix (unmodified).
    M_global : ndarray or sparse matrix of shape (N, N)
        Global acoustic mass matrix (unmodified).
    C : ndarray or sparse matrix of shape (N, N)
        Global acoustic impedance/damping matrix (unmodified).
    boundary_pressure_zero : array-like of int
        Indices (0-based) of nodes where the acoustic pressure
        is constrained to zero.

    Returns
    -------
    K_red : ndarray or sparse matrix
        Reduced acoustic stiffness matrix.
    M_red : ndarray or sparse matrix
        Reduced acoustic mass matrix.
    C_red : ndarray or sparse matrix
        Reduced acoustic impedance/damping matrix.
    idx_free : ndarray of int
        Global indices of the retained degrees of freedom
        (mapping from reduced to full system).

    Notes
    -----
    The reduction is equivalent to removing the rows and columns
    associated with the constrained degrees of freedom. A boolean
    mask is used to ensure efficient and explicit mapping between
    the global and reduced systems.

    See Also
    --------
    get_pressure_zero_nodes : Identify nodes with zero-pressure conditions.
    prepare_acoustic_system : High-level acoustic system preparation.
    """

    # Numero totale di nodi
    n = K_global.shape[0]

    # --------------------------------------------------
    # 1) Rendo unici gli indici (come MATLAB unique)
    # --------------------------------------------------
    boundary_pressure_zero = np.unique(
        np.array(boundary_pressure_zero, dtype=int)
    )

    # --------------------------------------------------
    # 2) Creo la maschera: True = nodo mantenuto
    # --------------------------------------------------
    mask = np.ones(n, dtype=bool)
    mask[boundary_pressure_zero] = False

    # --------------------------------------------------
    # 3) Indici dei DOF liberi (sistema ridotto)
    # --------------------------------------------------
    idx_free = np.where(mask)[0]

    # --------------------------------------------------
    # 4) Estraggo sottomatrici (equivalente a delete righe+colonne)
    # --------------------------------------------------
    K_red = K_global[np.ix_(idx_free, idx_free)]
    M_red = M_global[np.ix_(idx_free, idx_free)]
    C_red = C[np.ix_(idx_free, idx_free)]


    return K_red, M_red,C_red, idx_free

def get_pressure_zero_nodes(boundaries, bc_pressure_zero):
    """
    Identify nodes with zero-pressure (Dirichlet) boundary conditions.

    This function extracts the global node indices where the acoustic
    pressure is constrained to zero, based on the boundary definitions
    provided by the mesh generator (e.g. Gmsh).

    Parameters
    ----------
    boundaries : dict
        Dictionary mapping boundary names to lists of node indices
        (1-based indexing, as provided by Gmsh).
    bc_pressure_zero : list of str
        Names of the boundaries where the acoustic pressure is set
        to zero (e.g. ``["bottom", "left"]``).

    Returns
    -------
    p0_nodes : ndarray of int
        Global node indices (0-based) where the pressure is constrained
        to zero.

    Notes
    -----
    The node indices provided by the mesh generator are assumed to be
    1-based and are converted internally to 0-based indexing for use
    in Python and NumPy.

    See Also
    --------
    reduce_KMC_dirichlet_mask : Reduce the system by enforcing
        zero-pressure boundary conditions.
    prepare_acoustic_system : High-level acoustic system preparation.
    """
    p0_nodes = []

    for name in bc_pressure_zero:
        if name in boundaries:
            p0_nodes.extend(boundaries[name])

    # da 1-based (gmsh) a 0-based (python)
    p0_nodes = np.unique(np.array(p0_nodes, dtype=int) - 1)

    return p0_nodes


def get_pressure_bc_from_boundaries(
    boundaries,
    bc_pressure_constant,
    idx_free,
    pressure_value,
):

    """
    Map constant-pressure boundary conditions to the reduced system.

    This function converts constant-pressure boundary conditions
    defined on mesh boundaries into node indices and values
    compatible with the reduced acoustic system obtained after
    applying Dirichlet boundary conditions.

    Parameters
    ----------
    boundaries : dict
        Dictionary mapping boundary names to lists of node indices
        (1-based indexing, as provided by the mesh generator).
    bc_pressure_constant : list of str
        Names of the boundaries where a constant acoustic pressure
        is imposed (e.g. ``["upper", "top"]``).
    idx_free : ndarray of int
        Global node indices retained in the reduced system,
        as returned by :func:`reduce_KMC_dirichlet_mask`.
    pressure_value : float or complex
        Imposed acoustic pressure value.

    Returns
    -------
    pressure_nodes_red : ndarray of int
        Node indices in the reduced system where the pressure
        is prescribed.
    pressure_values : ndarray of complex
        Pressure values corresponding to ``pressure_nodes_red``.

    Notes
    -----
    Boundary node indices are assumed to be 1-based and are internally
    converted to 0-based indexing. Only nodes retained in the reduced
    system are included in the output.

    See Also
    --------
    reduce_KMC_dirichlet_mask : Reduce the system by enforcing
        Dirichlet boundary conditions.
    get_pressure_zero_nodes : Identify zero-pressure boundary nodes.
    prepare_acoustic_system : High-level acoustic system preparation.
    """

    # 1) Nodi globali (0-based) dalla boundary
    nodes_global = []
    for name in bc_pressure_constant:
        if name in boundaries:
            nodes_global.extend(boundaries[name])

    nodes_global = np.unique(np.array(nodes_global, dtype=int) - 1)

    # 2) Mappo nel sistema ridotto
    pressure_nodes_red = []
    for n in nodes_global:
        if n in idx_free:
            pressure_nodes_red.append(
                np.where(idx_free == n)[0][0]
            )

    pressure_nodes_red = np.array(pressure_nodes_red, dtype=int)

    # 3) Valori di pressione
    pressure_values = np.full(
        len(pressure_nodes_red),
        pressure_value,
        dtype=complex,
    )

    return pressure_nodes_red, pressure_values

def find_closest_node(nodes, pos_xy):
    """
    nodes: (N,3) or (N,2)
    pos_xy: (2,) [x,y]
    returns: node_id (0-based)
    """
    pos_xy = np.asarray(pos_xy, dtype=float).reshape(1, 2)
    dist = np.linalg.norm(nodes[:, :2] - pos_xy, axis=1)
    return int(np.argmin(dist))



def get_pressure_bc_from_boundaries_1(
    boundaries,
    bc_pressure_constant,
    idx_free,
    pressure_value,
    source_node_global=None,
    source_pressure_value=None,
):


    # --------------------------------------------------
    # 1) Nodi globali (0-based) dalla boundary
    # --------------------------------------------------
    nodes_global = []

    if bc_pressure_constant is not None:
        for name in bc_pressure_constant:
            if name in boundaries:
                nodes_global.extend(boundaries[name])

    nodes_global = list(np.unique(np.array(nodes_global, dtype=int) - 1))

    # --------------------------------------------------
    # 2) Aggiungo POINT SOURCE (se presente)
    # --------------------------------------------------
    if source_node_global is not None and source_pressure_value is not None:
        if source_node_global in nodes_global:
            # se già presente (stessa boundary) → sovrascrivo
            idx = nodes_global.index(source_node_global)
        else:
            nodes_global.append(source_node_global)
            idx = len(nodes_global) - 1
    else:
        idx = None

    # --------------------------------------------------
    # 3) Mapping nel sistema RIDOTTO
    # --------------------------------------------------
    pressure_nodes_red = []
    pressure_values = []

    for i, n in enumerate(nodes_global):
        if n in idx_free:
            red_idx = np.where(idx_free == n)[0][0]
            pressure_nodes_red.append(red_idx)

            # boundary pressure
            if idx is None or i != idx:
                pressure_values.append(pressure_value)
            else:
                pressure_values.append(source_pressure_value)

    pressure_nodes_red = np.array(pressure_nodes_red, dtype=int)
    pressure_values = np.array(pressure_values, dtype=complex)

    return pressure_nodes_red, pressure_values



def build_impedance_matrix(
    nodes,
    boundary_nodes_impedance,
    rho,
    c0,
    Z,
    elements=None,
    *,
    boundaries=None,
    groups=None,
    boundary_dim=None,
    omega=0.0,
):
    """
    Assemble the global acoustic impedance (damping) matrix on a boundary.

    The impedance condition ``p = Zbar v_n`` on ``Omega_Z`` enters the
    dynamic system through the boundary matrix

    .. math::
        C_{ab} = \\int_{\\Omega_Z} \\rho_0 \\, \\bar A \\, N_a N_b \\, d\\Omega ,

    with ``Abar = 1 / Zbar`` the specific admittance. The integral is
    evaluated on the actual boundary elements of the mesh: edges for a
    2D fluid, faces for a 3D one.

    Parameters
    ----------
    nodes : ndarray of shape (N, 2) or (N, 3)
        Coordinates of the mesh nodes.
    boundary_nodes_impedance : list of str or array-like of int
        Boundary selection: physical-group names (resolved through
        ``boundaries``/``groups``) or 1-based node tags.
    rho : float
        Fluid density.
    c0 : float
        Speed of sound in the fluid.
    Z : complex, callable or tuple
        Impedance specification. A bare number is the **normalized**
        impedance ``zeta`` (absolute value ``zeta * rho * c0``);
        ``(value, "abs")`` gives Pa*s/m; a callable ``Z(omega)``
        describes a frequency-dependent liner. ``Z = 0`` means a rigid
        wall and yields an all-zero matrix. See
        :func:`pycafe.boundary_condition.acoustic_bc.make_admittance`.
    elements : dict, optional
        Mesh connectivity, used to locate the boundary elements.
    boundaries : dict, optional
        ``{name: [1-based node tags]}``. Required when the selection is
        given by name.
    groups : dict, optional
        Physical groups with connectivity (``load_mesh_with_groups``);
        the authoritative source for 3D boundary faces.
    boundary_dim : int, optional
        1 for edges, 2 for faces. Inferred from ``elements`` if omitted.
    omega : float, optional
        Angular frequency at which a frequency-dependent impedance is
        evaluated. Ignored for a constant impedance. Default 0.

    Returns
    -------
    C_global : scipy.sparse matrix of complex, shape (N, N)
        Global acoustic impedance (damping) matrix.

    Notes
    -----
    For a frequency sweep with a frequency-dependent impedance, build an
    :class:`~pycafe.boundary_condition.acoustic_bc.ImpedanceOperator`
    instead: it keeps the geometry and the material apart, so each
    frequency only re-scales an already assembled matrix.

    See Also
    --------
    reduce_KMC_dirichlet_mask : Reduce the system after applying
        Dirichlet boundary conditions.
    prepare_acoustic_system : High-level acoustic system preparation.
    """
    from pycafe.boundary_condition.acoustic_bc import (
        AcousticBC,
        build_impedance_operator,
    )

    num_nodes = np.asarray(nodes).shape[0]

    if boundary_nodes_impedance is None or len(boundary_nodes_impedance) == 0:
        return lil_matrix((num_nodes, num_nodes), dtype=complex)

    bc = AcousticBC().add_impedance(boundary_nodes_impedance, Z)
    operator = build_impedance_operator(
        bc,
        nodes=nodes,
        rho=rho,
        c0=c0,
        boundaries=boundaries,
        groups=groups,
        elements=elements,
        boundary_dim=boundary_dim,
    )

    if operator.is_empty:
        return lil_matrix((num_nodes, num_nodes), dtype=complex)

    return operator.at(omega)


def expand_to_full(p_red, idx_free, p0_nodes, num_nodes):
    """
    Expand a reduced acoustic solution to the full nodal domain.

    This function reconstructs the full pressure field from a reduced
    solution vector by reinserting the eliminated degrees of freedom.
    Nodes subject to Dirichlet boundary conditions (p = 0) are explicitly
    set to zero.

    Parameters
    ----------
    p_red : ndarray
        Reduced pressure solution. Can be either:
        - 1D array of shape (N_red,)
        - 2D array of shape (N_red, N_freq) for frequency-domain solutions.
    idx_free : ndarray of int
        Indices of the free degrees of freedom (0-based indexing).
        This array maps reduced DOFs to global node indices.
    p0_nodes : ndarray of int
        Indices of nodes where pressure is constrained to zero
        (Dirichlet boundary condition).
    num_nodes : int
        Total number of nodes in the full mesh.

    Returns
    -------
    p_full : ndarray
        Expanded pressure field in the full domain. Shape is:
        - (num_nodes,) for single-frequency solutions
        - (num_nodes, N_freq) for multi-frequency solutions.

    Notes
    -----
    This utility is typically used after solving a reduced acoustic
    finite element system to recover the full nodal pressure field
    for visualization or post-processing.

    See Also
    --------
    reduce_KMC_dirichlet_mask : Reduction of the global system with
        Dirichlet boundary conditions.
    expand_mode_to_full : Expansion of reduced modal shapes to the
        full domain.
    """

    p_red = np.asarray(p_red)

    if p_red.ndim == 1:
        p_full = np.zeros(num_nodes, dtype=complex)
        p_full[idx_free] = p_red
        p_full[p0_nodes] = 0.0
    else:
        n_freq = p_red.shape[1]
        p_full = np.zeros((num_nodes, n_freq), dtype=complex)
        p_full[idx_free, :] = p_red
        p_full[p0_nodes, :] = 0.0

    return p_full

def expand_mode_to_full(mode_red, idx_free, p0_nodes, Ntot):

    mode_full = np.zeros(Ntot)
    mode_full[idx_free] = mode_red
    mode_full[p0_nodes] = 0.0
    return mode_full
