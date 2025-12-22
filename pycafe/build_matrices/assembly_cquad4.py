# assemble_cquad8.py  (versione sparse-ready, senza PML / impedenza)
import numpy as np
from .element_cquad4 import (
    element_matrices_cquad4,
    gauss_rule_quad_2x2,
    cquad4_shape,
    jacobian_2d, B_damping, calcola_line,linear_shape_1d,jacobian_1d, 
)
import scipy
from scipy.sparse import lil_matrix


def get_quad4_connectivity(elements_dict):
    for name, conn in elements_dict.items():
        if ("Quadrilateral 4" in name or "Quadrangle 4" in name) and conn.shape[1] == 4:
            return conn.astype(int)
    raise RuntimeError("No CQUAD4 elements found.")


def build_KM_cquad4(
    nodes,
    elements_dict,
    c,
    debug=False,
    debug_elem_id=0,
    store_all_elements=False,
):
    """
    Assemble the global acoustic stiffness and mass matrices for CQUAD4 elements.

    This function performs finite element assembly of the global
    acoustic stiffness matrix (K) and mass matrix (M) using
    4-node quadrilateral (CQUAD4) elements. No boundary conditions
    are applied and no system reduction or solution is performed.

    Parameters
    ----------
    nodes : ndarray of shape (N, 2) or (N, 3)
        Coordinates of the mesh nodes.
    elements_dict : dict
        Dictionary containing element connectivity information,
        as returned by the mesh loader.
    c : float
        Speed of sound in the fluid.
    debug : bool, optional
        If True, detailed debug information for a single element
        is stored. Default is False.
    debug_elem_id : int, optional
        Index of the element for which detailed debug information
        is collected when ``debug=True``. Default is 0.
    store_all_elements : bool, optional
        If True, element-level matrices and data are stored for
        all elements. This is intended for debugging and validation.
        Default is False.

    Returns
    -------
    K_global : scipy.sparse.lil_matrix
        Global acoustic stiffness matrix assembled from all elements.
    M_global : scipy.sparse.lil_matrix
        Global acoustic mass matrix assembled from all elements.
    debug_data : dict or None
        Debug information for the selected element or, if
        ``store_all_elements=True``, for all elements.

    Notes
    -----
    This function performs matrix assembly only.
    Boundary conditions, system reduction, and solution procedures
    are handled by higher-level functions.

    The LIL sparse matrix format is used for efficient incremental
    assembly of the global matrices.

    See Also
    --------
    element_matrices_cquad4 : Compute element-level matrices.
    build_KM_cquad8 : Assembly routine for CQUAD8 elements.
    prepare_acoustic_system : High-level system preparation.
    """


    num_nodes = nodes.shape[0]
     
     # LIL = perfetto per assembly
    K_global = lil_matrix((num_nodes, num_nodes), dtype=float)
    M_global = lil_matrix((num_nodes, num_nodes), dtype=float)

    quad_conn = get_quad4_connectivity(elements_dict) - 1

    debug_data = None
    all_elements = []

    for ielem, elem in enumerate(quad_conn):
        coords = nodes[elem, :2]        # (8,2)
        K_e, M_e = element_matrices_cquad4(coords, c, gauss_rule_quad_2x2())

        # Assembly
        for a, A in enumerate(elem):
            for b, B in enumerate(elem):
                K_global[A, B] += K_e[a, b]
                M_global[A, B] += M_e[a, b]

        # Store all elements if requested
        if store_all_elements:
            all_elements.append({
                "elem_id": ielem,
                "nodes": elem.copy(),
                "coords": coords.copy(),
                "K_e": K_e.copy(),
                "M_e": M_e.copy(),
            })

        # Debug ONE element (Gauss points)
        if debug and ielem == debug_elem_id:
            xi_g, eta_g, _ = gauss_rule_quad_2x2()
            gp_data = []

            for xi, eta in zip(xi_g, eta_g):
                N, dNdxi, dNdeta = cquad4_shape(xi, eta)
                J, detJ, invJ = jacobian_2d(dNdxi, dNdeta, coords)

                gp_data.append({
                    "xi": xi,
                    "eta": eta,
                    "N": N,
                    "J": J,
                    "detJ": detJ,
                    "invJ": invJ,
                })

            debug_data = {
                "elem_id": ielem,
                "nodes": elem.copy(),
                "coords": coords.copy(),
                "K_e": K_e.copy(),
                "M_e": M_e.copy(),
                "gauss_points": gp_data,
            }

    if store_all_elements:
        debug_data = debug_data or {}
        debug_data["all_elements"] = all_elements

    return K_global, M_global, debug_data



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
):
    """
    Assemble the global acoustic impedance (damping) matrix on a boundary.

    This function constructs the global impedance matrix ``C`` associated
    with acoustic impedance boundary conditions. The matrix is assembled
    by integrating boundary contributions along the specified boundary
    edges using a one-dimensional finite element formulation.

    Parameters
    ----------
    nodes : ndarray of shape (N, 2) or (N, 3)
        Coordinates of the mesh nodes. For 2D acoustic problems,
        only the (x, y) coordinates are used.
    boundary_nodes_impedance : array-like of int
        Ordered list of node indices defining the impedance boundary.
        Indices are assumed to be 1-based (as provided by Gmsh).
    rho : float
        Fluid density.
    c0 : float
        Speed of sound in the fluid.
    Z : float or complex
        Acoustic impedance value. A value of zero corresponds to
        zero admittance (rigid boundary).

    Returns
    -------
    C_global : scipy.sparse.lil_matrix
        Global acoustic impedance (damping) matrix.

    Notes
    -----
    The impedance is introduced through its admittance, defined as
    the inverse of the specific acoustic impedance ``Z_spec = Z * rho * c0``.
    Boundary node indices are internally converted from 1-based to
    0-based indexing.

    The impedance matrix is assembled using linear shape functions
    along the boundary edges and Gaussian quadrature.

    See Also
    --------
    reduce_KMC_dirichlet_mask : Reduce the system after applying
        Dirichlet boundary conditions.
    prepare_acoustic_system : High-level acoustic system preparation.
    """

    num_nodes = nodes.shape[0]
    C_global = lil_matrix((num_nodes, num_nodes), dtype=complex)
    # Zeta = impedenza normalizzata (input utente)
    zeta = Z

    Z_spec = zeta * rho * c0        # Pa·s/m
    
    if Z == 0:
        admittance = 0
    else:
        admittance = 1.0 / Z_spec

    # numero di "edge" consecutivi
    num_edges = len(boundary_nodes_impedance) - 1

    gp = 1.0 / np.sqrt(3.0)
    Xi = np.array([-gp, gp])

    for e in range(num_edges):
        # nodi globali (0-based)
        node_indices = boundary_nodes_impedance[e:e+2] - 1

        x1 = nodes[node_indices[0], :2]
        x2 = nodes[node_indices[1], :2]

        T_1D, x0, x1D_e = calcola_line(x1, x2)

        C_e = np.zeros((2, 2), dtype=complex)

        for xi in Xi:
            w = 1.0
            N, dNdxi = linear_shape_1d(xi)
            J, detJ, invJ = jacobian_1d(dNdxi, x1D_e)

            B2 = B_damping(N)

            C_e += (rho * admittance) * (B2.T @ B2) * detJ * w

        # Assemblaggio globale
        for i in range(2):
            for j in range(2):
                C_global[node_indices[i], node_indices[j]] += C_e[i, j]

    return C_global


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

