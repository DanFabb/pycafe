import numpy as np

from pycafe.build_matrices.assembly_dispatcher import build_KM_acoustic
from pycafe.build_matrices.assembly_cquad8 import (
    build_impedance_matrix,
    reduce_KMC_dirichlet_mask,
    get_pressure_zero_nodes,
    get_pressure_bc_from_boundaries_1,
)
from pycafe.build_matrices.assembly_cquad4 import (
    build_impedance_matrix,
    reduce_KMC_dirichlet_mask,
    get_pressure_zero_nodes,
    get_pressure_bc_from_boundaries_1,
)

def prepare_acoustic_system(
    *,
    nodes,
    elements,
    boundaries,
    rho,
    c0,
    bc,
    debug=False,
    debug_elem_id=0,
):
    """
    Prepare the reduced acoustic finite element system.

    This function assembles the global acoustic matrices, applies
    impedance and Dirichlet boundary conditions, and maps pressure
    constraints to the reduced system. The resulting data structure
    contains all matrices and mappings required to run an acoustic
    analysis.

    The preparation workflow includes:
    - Assembly of stiffness and mass matrices
    - Construction of the impedance matrix
    - Enforcement of zero-pressure (Dirichlet) boundary conditions
    - Mapping of constant pressure boundaries and point sources

    Parameters
    ----------
    nodes : ndarray of shape (N, 2) or (N, 3)
        Coordinates of the mesh nodes.
    elements : dict
        Element connectivity information as returned by the mesh loader.
    boundaries : dict
        Dictionary mapping boundary names to lists of node indices.
    rho : float
        Fluid density.
    c0 : float
        Speed of sound in the fluid.
    bc : tuple
        Boundary condition tuple returned by
        :func:`assign_boundary_conditions`.
    debug : bool, optional
        If True, additional debug information is stored during
        matrix assembly. Default is False.
    debug_elem_id : int, optional
        Element index used for detailed debug output when
        ``debug=True``. Default is 0.

    Returns
    -------
    system : dict
        Dictionary containing the complete acoustic system data with
        the following entries:

        - ``K`` : sparse matrix  
            Global acoustic stiffness matrix.
        - ``M`` : sparse matrix  
            Global acoustic mass matrix.
        - ``C`` : sparse matrix  
            Global acoustic impedance matrix.
        - ``K_red`` : sparse matrix  
            Reduced stiffness matrix.
        - ``M_red`` : sparse matrix  
            Reduced mass matrix.
        - ``C_red`` : sparse matrix  
            Reduced impedance matrix.
        - ``idx_free`` : ndarray  
            Indices of free degrees of freedom.
        - ``p0_nodes`` : ndarray  
            Node indices with zero-pressure boundary conditions.
        - ``pressure_nodes_red`` : ndarray  
            Indices of pressure-constrained nodes in the reduced system.
        - ``pressure_values`` : ndarray  
            Imposed pressure values in the reduced system.
        - ``bc_velocity`` : list of str  
            Boundary names with imposed normal velocity.
        - ``value_velocity_normal`` : float  
            Normal velocity value [m/s].
        - ``elem_type`` : str  
            Element type used for the assembly.
        - ``debug`` : dict or None  
            Debug information generated during assembly.

    Notes
    -----
    This function prepares the system but does not solve it.
    Use :func:`run_analysis` to perform modal or frequency-domain
    acoustic analyses.

    See Also
    --------
    create_matrices : Assemble global acoustic matrices.
    run_analysis : Run the acoustic simulation.
    assign_boundary_conditions : Interactive boundary condition setup.
    """

    (
        bc_pressure_zero,
        bc_pressure_constant,
        pressure_constant_value,
        boundary_nodes_impedance,
        Z_impedance,
        bc_velocity,
        value_velocity_normal,
        source_node,
        source_pressure_value,
    ) = bc

    # --------------------------------------------------
    # 1) Build K, M
    # --------------------------------------------------
    K, M, dbg, elem_type = build_KM_acoustic(
        nodes,
        elements,
        c0,
        debug=debug,
        debug_elem_id=debug_elem_id,
        store_all_elements=debug,
    )

    # --------------------------------------------------
    # 2) Build impedance matrix C
    # --------------------------------------------------
    C = build_impedance_matrix(
        nodes,
        boundary_nodes_impedance,
        rho,
        c0,
        Z_impedance,
    )

    # --------------------------------------------------
    # 3) Dirichlet pressure = 0
    # --------------------------------------------------
    p0_nodes = get_pressure_zero_nodes(
        boundaries,
        bc_pressure_zero,
    )

    K_red, M_red, C_red, idx_free = reduce_KMC_dirichlet_mask(
        K,
        M,
        C,
        p0_nodes,
    )

    # --------------------------------------------------
    # 4) Pressure BC (constant + point source)
    # --------------------------------------------------
    pressure_nodes_red, pressure_values = \
        get_pressure_bc_from_boundaries_1(
            boundaries=boundaries,
            bc_pressure_constant=bc_pressure_constant,
            idx_free=idx_free,
            pressure_value=pressure_constant_value,
            source_node_global=source_node,
            source_pressure_value=source_pressure_value,
        )

    return {
        "K": K,
        "M": M,
        "C": C,
        "K_red": K_red,
        "M_red": M_red,
        "C_red": C_red,
        "idx_free": idx_free,
        "p0_nodes": p0_nodes,
        "pressure_nodes_red": pressure_nodes_red,
        "pressure_values": pressure_values,
        "bc_velocity": bc_velocity,
        "value_velocity_normal": value_velocity_normal,
        "elem_type": elem_type,
        "debug": dbg,
    }
