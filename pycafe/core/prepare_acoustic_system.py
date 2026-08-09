import numpy as np

from pycafe.boundary_condition.acoustic_bc import (
    AcousticBC,
    check_node_index,
    build_impedance_operator,
    build_radiation_operator,
    build_source_operator,
    build_velocity_operator,
)
from pycafe.build_matrices.assembly_dispatcher import build_KM_acoustic
from pycafe.build_matrices.bc_ops import (
    reduce_KMC_dirichlet_mask,
    get_pressure_zero_nodes,
    get_pressure_bc_from_boundaries_1,
)
from pycafe.build_matrices.element_registry import ELEMENT_TYPES
from pycafe.build_matrices.pml import pml_from_groups
from pycafe.create_geom.conventions import names_for

def _merge_element_groups(*element_dicts):
    """Concatenate the connectivity of several physical groups."""
    merged = {}
    for elements in element_dicts:
        for name, conn in elements.items():
            conn = np.asarray(conn, dtype=int)
            merged[name] = (conn if name not in merged
                            else np.vstack([merged[name], conn]))
    return merged


def _check_no_orphan_dofs(K_red, M_red, pml_op, idx_free):
    """
    Refuse a system where some unknowns are touched by no element.

    Such a row is exactly zero at every frequency, so the solve fails
    with nothing more informative than "matrix is exactly singular". It
    happens when the assembled domain is a strict subset of the mesh —
    typically a physical group that leaves part of it out.
    """
    covered = np.zeros(K_red.shape[0], dtype=bool)
    for matrix in (K_red, M_red):
        csr = matrix.tocsr()
        covered |= np.diff(csr.indptr) > 0
    if pml_op is not None:
        covered[pml_op.covered_dofs] = True

    orphans = np.flatnonzero(~covered)
    if orphans.size:
        idx_free = np.asarray(idx_free, dtype=int)
        raise ValueError(
            f"{orphans.size} of {covered.size} acoustic unknowns are on no "
            "element, so the system is singular before any solve. Global "
            f"node indices (0-based) start at {idx_free[orphans[:5]].tolist()}"
            f"{' ...' if orphans.size > 5 else ''}. This usually means the "
            "assembled domain leaves part of the mesh out: check that every "
            "region of the fluid is in a physical group the analysis reads."
        )


def prepare_acoustic_system(
    *,
    nodes,
    elements,
    boundaries,
    rho,
    c0,
    bc,
    groups=None,
    pml=None,
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
    bc : AcousticBC or tuple
        Boundary condition description. Either an
        :class:`~pycafe.boundary_condition.acoustic_bc.AcousticBC`
        instance or the legacy tuple returned by
        :func:`assign_boundary_conditions`.
    groups : dict, optional
        Physical groups with element connectivity, as returned by
        ``load_mesh_with_groups``. Required to integrate impedance and
        velocity boundary conditions on the **faces** of a 3D mesh when
        the faces are not already present in ``elements``.
    debug : bool, optional
        If True, additional debug information is stored during
        matrix assembly. Default is False.
    pml : dict or False, optional
        Settings for the absorbing layer, passed on to
        :func:`~pycafe.build_matrices.pml.pml_from_groups` (for instance
        ``{"target_reflection": 1e-2}``). The layer itself comes from
        the mesh: a physical group named ``pml`` next to the fluid one
        turns it on, and nothing else is needed. Requires ``groups``.

        Pass ``False`` to treat the layer as ordinary fluid: its
        elements then join the acoustic domain instead of being replaced
        by the operator, which is the comparison to run before trusting
        an absorbing model.
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
            Global acoustic impedance matrix, evaluated at
            ``omega = 0`` when the impedance is frequency dependent.
        - ``C_op`` / ``C_red_op`` : ImpedanceOperator
            Global and reduced impedance operators; call ``.at(omega)``
            to get ``C`` at a given frequency.
        - ``velocity_op`` / ``velocity_red_op`` : NormalVelocityOperator
            Global and reduced normal-velocity load operators.
        - ``source_op`` / ``source_red_op`` : AcousticSourceOperator
            Global and reduced volumetric source (monopole) operators.
        - ``radiation_op`` / ``radiation_red_op`` : SphericalRadiationOperator
            Global and reduced spherical wave radiation operators; they
            contribute a matrix, and a load when an incident field was
            declared on the boundary.
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

    bc = AcousticBC.from_legacy(bc)

    # The global ``elements`` dict mixes the roles: on a mesh with a
    # separate absorbing layer, its tetrahedra sit in the same
    # "Tetrahedron 4" array as the fluid's. Only the physical groups
    # tell them apart, so when they are available the acoustic domain is
    # the fluid group and nothing else — otherwise the layer would be
    # assembled as ordinary fluid *and* again through the PML operator,
    # which replaces rather than corrects.
    fluid_group = pml_group = None
    if groups is not None:
        fluid_group = next(
            (g for g in groups if g.lower() in names_for("fluid")), None
        )
        pml_group = next(
            (g for g in groups if g.lower() in names_for("pml")), None
        )

    # ``pml=False`` means the layer is to be treated as ordinary fluid,
    # so its elements join the acoustic domain. Dropping them instead
    # would leave their nodes in the system with no element on them.
    merged = None
    if fluid_group is not None and pml_group is not None and pml is False:
        merged = _merge_element_groups(groups[fluid_group]["elements"],
                                       groups[pml_group]["elements"])

    if merged is not None:
        domain_elements = merged
    elif fluid_group is not None:
        domain_elements = groups[fluid_group]["elements"]
    else:
        domain_elements = elements

    # 1) Build K, M
    K, M, dbg, elem_type = build_KM_acoustic(
        nodes,
        elements if merged is None else merged,
        c0,
        groups=groups if (fluid_group is not None and merged is None)
        else None,
        debug=debug,
        debug_elem_id=debug_elem_id,
        store_all_elements=debug,
    )

    # The fluid boundary is one dimension below the fluid itself:
    # edges for a 2D domain, faces for a 3D one.
    boundary_dim = ELEMENT_TYPES[elem_type].dim - 1

    # 2) Impedance (Robin) -> boundary matrix C
    C_op = build_impedance_operator(
        bc,
        nodes=nodes,
        rho=rho,
        c0=c0,
        boundaries=boundaries,
        groups=groups,
        elements=elements,
        boundary_dim=boundary_dim,
    )
    C = C_op.at(0.0)

    # 2b) Normal velocity (Neumann) -> load vector V_n
    velocity_op = build_velocity_operator(
        bc,
        nodes=nodes,
        rho=rho,
        boundaries=boundaries,
        groups=groups,
        elements=elements,
        boundary_dim=boundary_dim,
    )

    # 2b') Spherical wave radiation -> a matrix and, with an incident
    # field, a load vector. Not an impedance: see
    # pycafe.build_matrices.bc_radiation.
    radiation_op = build_radiation_operator(
        bc,
        nodes=nodes,
        c0=c0,
        boundaries=boundaries,
        groups=groups,
        elements=elements,
        boundary_dim=boundary_dim,
    )

    # 2c) Volumetric sources (monopoles) -> load vector Q
    # Volumetric sources belong to the physical fluid: integrating a
    # distributed source over the absorbing layer as well would turn the
    # absorber into a radiator, and make the answer depend on how thick
    # it is.
    source_op = build_source_operator(
        bc,
        nodes=nodes,
        rho=rho,
        elements=domain_elements,
    )

    # 3) Dirichlet pressure = 0
    p0_nodes = get_pressure_zero_nodes(
        boundaries,
        bc.pressure_zero,
    )

    K_red, M_red, C_red, idx_free = reduce_KMC_dirichlet_mask(
        K,
        M,
        C,
        p0_nodes,
    )

    # 3b) Absorbing layer, if the mesh names one
    pml_op = None
    if groups is not None and pml is not False:
        pml_op = pml_from_groups(
            nodes, groups, c0, n_dof=nodes.shape[0],
            **(pml or {}),
        )

    _check_no_orphan_dofs(K_red, M_red, pml_op, idx_free)

    # 4) Pressure BC (constant + point source)
    # Each entry carries its own value, so they are mapped one by one
    # and concatenated; the point sources are appended last so that a
    # node shared with a boundary takes the source value.
    pressure_nodes_red = np.array([], dtype=int)
    pressure_values = np.array([], dtype=complex)

    # A node index outside the mesh would simply match nothing in the
    # reduced system, so the condition would vanish without a word.
    n_nodes = np.asarray(nodes).shape[0]
    entries = [(e.selection, e.value, None, None) for e in bc.pressure_constant]
    entries += [
        (None, 0.0, check_node_index(e.node, "point pressure", n_nodes), e.value)
        for e in bc.point_pressure
    ]

    for selection, value, src_node, src_value in entries:
        nodes_red, values = get_pressure_bc_from_boundaries_1(
            boundaries=boundaries,
            bc_pressure_constant=selection,
            idx_free=idx_free,
            pressure_value=value,
            source_node_global=src_node,
            source_pressure_value=src_value,
        )
        pressure_nodes_red = np.concatenate([pressure_nodes_red, nodes_red])
        pressure_values = np.concatenate([pressure_values, values])

    # Later entries win over earlier ones on a repeated DOF.
    if pressure_nodes_red.size:
        pressure_nodes_red, keep = np.unique(
            pressure_nodes_red[::-1], return_index=True
        )
        pressure_values = pressure_values[::-1][keep]

    # Legacy convenience fields: a single velocity boundary and value.
    # Models with several distinct velocities are described only by the
    # operators, which have no such limitation.
    bc_velocity, value_velocity_normal = [], 0.0
    if bc.velocity:
        for entry in bc.velocity:
            bc_velocity.extend(entry.selection)
        first = bc.velocity[0].v_n
        value_velocity_normal = first if not callable(first) else None

    return {
        "K": K,
        "M": M,
        "C": C,
        "K_red": K_red,
        "M_red": M_red,
        "C_red": C_red,
        "C_op": C_op,
        "C_red_op": C_op.reduce(idx_free),
        "velocity_op": velocity_op,
        "velocity_red_op": velocity_op.reduce(idx_free),
        "source_op": source_op,
        "source_red_op": source_op.reduce(idx_free),
        "radiation_op": radiation_op,
        "radiation_red_op": radiation_op.reduce(idx_free),
        "pml_op": pml_op,
        "pml_red_op": None if pml_op is None else pml_op.reduce(idx_free),
        "bc": bc,
        "boundary_dim": boundary_dim,
        "idx_free": idx_free,
        "p0_nodes": p0_nodes,
        "pressure_nodes_red": pressure_nodes_red,
        "pressure_values": pressure_values,
        "bc_velocity": bc_velocity,
        "value_velocity_normal": value_velocity_normal,
        "elem_type": elem_type,
        "debug": dbg,
    }
