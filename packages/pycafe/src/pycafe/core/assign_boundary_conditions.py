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
        from pycafe.boundary_condition.boundary_condition import (
            choose_boundary_conditions,
        )

        return choose_boundary_conditions(boundaries, nodes)

    # Non-interactive mode: return an empty BC definition suitable for
    # tests and scripted workflows. The solver remains unconstrained
    # unless the caller specifies BCs explicitly.

    return (
        [],
        [],
        0.0,
        [],
        0.0,
        [],
        0.0,
        None,
        None,
    )
