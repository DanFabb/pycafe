from pycafe.build_matrices.assembly_dispatcher import build_KM_acoustic
from pycafe.build_matrices.assembly_cquad8 import build_impedance_matrix
 
 
def create_matrices(
    *,
    nodes,
    elements,
    rho,
    c0,
    boundary_nodes_impedance=None,
    Z_impedance=0.0,
    debug=False,
):
    """
    Assemble the global acoustic system matrices.

    This function creates the global acoustic stiffness matrix (K),
    mass matrix (M), and damping/impedance matrix (C) for the given
    mesh and fluid properties. No boundary condition reduction is
    applied at this stage.

    Parameters
    ----------
    nodes : ndarray of shape (N, 2) or (N, 3)
        Coordinates of the mesh nodes.
    elements : dict
        Element connectivity information as returned by the mesh loader.
    rho : float
        Fluid density.
    c0 : float
        Speed of sound in the fluid.
    boundary_nodes_impedance : array-like or None, optional
        Node indices defining impedance boundaries.
        If None, no impedance matrix is assembled.
        Default is None.
    Z_impedance : float or complex, optional
        Acoustic impedance value.
        Default is 0.0.
    debug : bool, optional
        If True, additional debug information is stored during
        matrix assembly. Default is False.

    Returns
    -------
    K : sparse matrix
        Global acoustic stiffness matrix.
    M : sparse matrix
        Global acoustic mass matrix.
    C : sparse matrix
        Global acoustic impedance/damping matrix.
    info : dict
        Dictionary containing additional information, such as:
        - ``elem_type`` : str
            Element type used in the assembly.
        - ``debug`` : dict or None
            Debug data generated during assembly.

    Notes
    -----
    This function performs matrix assembly only.
    Boundary condition enforcement and system reduction are handled
    separately by higher-level functions.

    See Also
    --------
    prepare_acoustic_system : Assemble and reduce the acoustic system.
    build_KM_acoustic : Low-level matrix assembly routine.
    """
    # K, M
    K, M, dbg, elem_type = build_KM_acoustic(
        nodes,
        elements,
        c0,
        debug=debug,
    )

    # C
    if boundary_nodes_impedance is not None and Z_impedance != 0:
        C = build_impedance_matrix(
            nodes,
            boundary_nodes_impedance,
            rho,
            c0,
            Z_impedance,
        )
    else:
        import numpy as np
        from scipy.sparse import lil_matrix
        C = lil_matrix((nodes.shape[0], nodes.shape[0]))

    info = {
        "elem_type": elem_type,
        "debug": dbg,
    }

    return K, M, C, info