def detect_acoustic_element_type(elements):
    """
    Detect the acoustic quadrilateral element type present in the mesh.

    This function inspects the element connectivity arrays and determines
    whether the mesh contains 4-node (CQUAD4) or 8-node (CQUAD8)
    quadrilateral acoustic elements.

    Parameters
    ----------
    elements : dict
        Dictionary mapping element group names to connectivity arrays.
        Each connectivity array is expected to have shape (n_elem, n_nodes),
        where ``n_nodes`` is the number of nodes per element.

    Returns
    -------
    elem_type : str
        Acoustic element type. One of:
        - ``"CQUAD4"`` for 4-node quadrilateral elements
        - ``"CQUAD8"`` for 8-node quadrilateral elements
    elem_key : str
        Key in the ``elements`` dictionary corresponding to the detected
        acoustic element group.

    Raises
    ------
    RuntimeError
        If no supported acoustic quadrilateral elements are found.

    Notes
    -----
    Only quadrilateral acoustic elements are supported. Mixed meshes
    or unsupported element types are ignored.

    See Also
    --------
    build_KM_acoustic : Automatic dispatcher for acoustic matrix assembly.
    """

    for key, conn in elements.items():
        if conn.ndim != 2:
            continue

        nnode = conn.shape[1]

        if nnode == 4:
            return "CQUAD4", key

        if nnode == 8:
            return "CQUAD8", key

    raise RuntimeError(
        "No supported acoustic quadrilateral elements found.\n"
        "Expected CQUAD4 (4 nodes) or CQUAD8 (8 nodes)."
    )

def build_KM_acoustic(
    nodes,
    elements,
    c0,
    debug=False,
    debug_elem_id=None,
    store_all_elements=False,
):
    """
    Assemble global acoustic FEM matrices with automatic element dispatching.

    This function automatically detects the acoustic element type present
    in the mesh (CQUAD4 or CQUAD8) and dispatches the assembly to the
    appropriate element-specific routine.

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

    elem_type, elem_key = detect_acoustic_element_type(elements)

    print(f"Detected acoustic element type: {elem_type}")

    if elem_type == "CQUAD4":
        from .assembly_cquad4 import build_KM_cquad4

        return (*build_KM_cquad4(
            nodes,
            elements,
            c0,
            debug=debug,
            debug_elem_id=debug_elem_id,
            store_all_elements=store_all_elements,
        ), elem_type)

    elif elem_type == "CQUAD8":
        from .assembly_cquad8 import build_KM_cquad8

        return (*build_KM_cquad8(
            nodes,
            elements,
            c0,
            debug=debug,
            debug_elem_id=debug_elem_id,
            store_all_elements=store_all_elements,
        ), elem_type)

    else:
        raise RuntimeError("Unsupported element type.")
