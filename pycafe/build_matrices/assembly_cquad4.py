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

# ---------------------------------------------------------------------------
# Deprecated re-exports: shared BC/reduction functions moved to bc_ops.py.
# Import them from pycafe.build_matrices.bc_ops instead.
# ---------------------------------------------------------------------------
from .bc_ops import (  # noqa: F401
    reduce_KM_dirichlet,
    reduce_KMC_dirichlet_mask,
    get_pressure_zero_nodes,
    get_pressure_bc_from_boundaries,
    find_closest_node,
    get_pressure_bc_from_boundaries_1,
    build_impedance_matrix,
    expand_to_full,
    expand_mode_to_full,
)
