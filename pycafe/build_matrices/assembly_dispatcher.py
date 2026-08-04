import numpy as np

from .element_registry import ELEMENT_TYPES, find_acoustic_elements


def detect_acoustic_element_type(elements):
    """
    Detect the acoustic element type present in the mesh.

    .. deprecated::
        Thin wrapper kept for backward compatibility. Use
        :func:`pycafe.build_matrices.element_registry.find_acoustic_elements`,
        which also returns the registry :class:`ElementSpec`.

    Parameters
    ----------
    elements : dict
        Dictionary mapping Gmsh element names to connectivity arrays
        of shape (n_elem, n_nodes).

    Returns
    -------
    elem_type : str
        Registry key of the detected acoustic element type
        (e.g. ``"CQUAD4"``, ``"CQUAD8"``, ``"CHEXA8"``).
    elem_key : str
        Key in the ``elements`` dictionary corresponding to the detected
        acoustic element group.

    Raises
    ------
    RuntimeError
        If no supported acoustic elements are found.

    See Also
    --------
    build_KM_acoustic : Automatic dispatcher for acoustic matrix assembly.
    """
    elem_type, elem_key, _ = find_acoustic_elements(elements)
    return elem_type, elem_key

def build_KM_acoustic(
    nodes,
    elements,
    c0,
    debug=False,
    debug_elem_id=None,
    store_all_elements=False,
    groups=None,
):
    """
    Assemble global acoustic FEM matrices with automatic element dispatching.

    When ``groups`` (physical groups with connectivity, from
    ``load_mesh_with_groups``) is provided, the acoustic domain is the
    physical group named ``fluid``/``acoustic``/``domain`` and only its
    elements are assembled — the authoritative choice on meshes that
    mix roles (e.g. plate vs wall quads). Without ``groups``, the
    element type is inferred from the global ``elements`` dict via the
    registry (legacy behaviour, fine for single-domain meshes).

    Parameters
    ----------
    nodes : ndarray of shape (N, 2) or (N, 3)
        Coordinates of the mesh nodes. For 2D acoustic problems,
        only the (x, y) coordinates are used.
    elements : dict
        Dictionary containing element connectivity arrays as provided
        by the mesh reader.
    c0 : float
        Speed of sound in the fluid.
    groups : dict, optional
        Physical groups with per-group element connectivity, as
        returned by ``load_mesh_with_groups``.
    debug : bool, optional
        If True, store detailed diagnostic information for one element.
        Default is False.
    debug_elem_id : int or None, optional
        Index of the element for which debug data should be collected.
        Ignored if ``debug=False``.
    store_all_elements : bool, optional
        If True, store element-level matrices for all elements.
        Intended for debugging or verification purposes.
        Default is False.

    Returns
    -------
    K_global : scipy.sparse.lil_matrix
        Global acoustic stiffness matrix.
    M_global : scipy.sparse.lil_matrix
        Global acoustic mass matrix.
    debug_data : dict or None
        Debug information for the selected element, or ``None`` if
        debugging is disabled.
    elem_type : str
        Detected acoustic element type (``"CQUAD4"`` or ``"CQUAD8"``).

    Raises
    ------
    RuntimeError
        If an unsupported or unknown element type is detected.

    Notes
    -----
    This function performs **assembly only**. Boundary conditions,
    system reduction, and solution steps are handled elsewhere
    in the workflow.

    See Also
    --------
    build_KM_cquad4 : Assembly routine for CQUAD4 elements.
    build_KM_cquad8 : Assembly routine for CQUAD8 elements.
    prepare_acoustic_system : High-level acoustic FEM setup.
    """

    if groups is not None:
        from .domains import identify_domains, build_KM_acoustic_domain

        domains = identify_domains(groups)
        print(
            f"Acoustic domain from physical group "
            f"'{domains['fluid']['group']}': {domains['fluid']['elem_type']}"
        )
        return build_KM_acoustic_domain(
            nodes, domains, c0,
            debug=debug,
            debug_elem_id=debug_elem_id or 0,
            store_all_elements=store_all_elements,
        )

    elem_type, elem_key, spec = find_acoustic_elements(elements)

    print(f"Detected acoustic element type: {elem_type}")

    if spec.kernel is None:
        raise NotImplementedError(
            f"Element type '{elem_type}' is registered but its kernel is "
            "not implemented yet (shape functions to be provided)."
        )

    from .assembly import assemble_KM

    conn0 = np.asarray(elements[elem_key], dtype=int) - 1

    K_global, M_global, debug_data = assemble_KM(
        nodes,
        conn0,
        spec.kernel,
        c0,
        debug=debug,
        debug_elem_id=debug_elem_id,
        store_all_elements=store_all_elements,
    )

    return K_global, M_global, debug_data, elem_type
