from pycafe.boundary_condition.boundary_condition import choose_boundary_conditions


def assign_boundary_conditions(*, boundaries, nodes, interactive=True):
    """
    Assign acoustic boundary conditions to the mesh.

    By default, this function runs in interactive mode and prompts
    the user to assign boundary conditions via the terminal.
    When ``interactive=False``, no user interaction is performed
    and an empty boundary condition definition is returned. This
    mode is intended for automated workflows and testing.

    Parameters
    ----------
    boundaries : dict
        Dictionary mapping boundary names to node indices, as
        returned by the mesh loader.
    nodes : ndarray
        Array of node coordinates.
    interactive : bool, optional
        If True (default), boundary conditions are assigned
        interactively via user input. If False, no prompts
        are shown and default (empty) boundary conditions
        are returned.

    Returns
    -------
    bc : tuple
        Tuple containing all boundary condition data required
        for the acoustic analysis. The tuple includes pressure,
        impedance, velocity, and optional point source data.

    Notes
    -----
    Interactive input is disabled when ``interactive=False``.
    This is required for automated testing, batch simulations,
    and continuous integration.

    See Also
    --------
    choose_boundary_conditions : Low-level boundary condition
        assignment function.
    """
    if interactive:
        return choose_boundary_conditions(boundaries, nodes)

    # ------------------------------------------------------------
    # Non-interactive mode: all boundaries -> pressure = 0
    # ------------------------------------------------------------

    boundary_names = list(boundaries.keys())

    pressure_nodes = []
    for node_ids in boundaries.values():
        pressure_nodes.extend(node_ids)

    # remove duplicates
    pressure_nodes = list(dict.fromkeys(pressure_nodes))

    pressure_value = 0

    impedance_nodes = []
    impedance_value = 0

    velocity_nodes = []
    velocity_value = 0

    point_source_nodes = None
    point_source_values = None

    return (
        boundary_names,
        pressure_nodes,
        pressure_value,
        impedance_nodes,
        impedance_value,
        velocity_nodes,
        velocity_value,
        point_source_nodes,
        point_source_values,
    )