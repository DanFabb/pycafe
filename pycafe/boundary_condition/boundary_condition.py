import numpy as np

def find_closest_node(nodes, pos_xy):
    pos_xy = np.asarray(pos_xy, dtype=float).reshape(1, 2)
    dist = np.linalg.norm(nodes[:, :2] - pos_xy, axis=1)
    return int(np.argmin(dist))

def choose_boundary_conditions(boundaries,nodes):
    """
    Interactively assign acoustic boundary conditions to the mesh.

    This function prompts the user to select a boundary condition
    for each mesh boundary. The selected conditions are returned
    in a structured tuple that can be used directly by the acoustic
    FEM solver.

    Available boundary condition types include:
    - Hard wall (default)
    - Zero pressure
    - Constant pressure
    - Acoustic impedance
    - Normal velocity
    - Optional point pressure source

    Parameters
    ----------
    boundaries : dict
        Dictionary mapping boundary names to lists of node indices
        (as returned by the mesh loader).
    nodes : ndarray
        Array of mesh node coordinates. Used to locate the closest
        node when defining a point pressure source.

    Returns
    -------
    bc : tuple
        Tuple containing the boundary condition data in the following
        order:

        - bc_pressure_zero : list of str
            Boundary names where pressure is set to zero.
        - bc_pressure_constant : list of str
            Boundary names with constant pressure.
        - pressure_constant_value : float
            Value of the imposed constant pressure [Pa].
        - boundary_nodes_impedance : list of str
            Boundary names with impedance boundary conditions.
        - Z_impedance : complex
            Acoustic impedance value.
        - bc_velocity : list of str
            Boundary names with imposed normal velocity.
        - value_velocity_normal : float
            Normal velocity value [m/s].
        - source_node : int or None
            Index of the node where a point pressure source is applied.
        - source_pressure_value : float
            Amplitude of the point pressure source [Pa].

    Notes
    -----
    This function is intended for interactive use and requires user
    input via the terminal. It should not be used in fully automated
    workflows without modification.

    See Also
    --------
    assign_boundary_conditions : High-level wrapper for interactive
        boundary condition assignment.
    """

    print("\nAvailable boundaries in the mesh:")
    print("---------------------------------")
    for i, name in enumerate(boundaries.keys()):
        print(f"{i+1}) {name} ({len(boundaries[name])} nodes)")

    print("\nAvailable boundary conditions:")
    print("  0 -> Hard wall (default)")
    print("  1 -> Pressure = 0")
    print("  2 -> Constant pressure")
    print("  3 -> Impedance")
    print("  4 -> Normal velocity")
    print("  5 -> Point pressure source (node)")

    bc_pressure_zero = []
    bc_pressure_constant = []
    boundary_nodes_impedance = []
    bc_velocity = []

    pressure_constant_value = 0
    Z_impedance = 0
    value_velocity_normal = 0

    source_node = None
    source_pressure_value = 0


    print("\nAssign BC to each boundary:\n")

    for name in boundaries.keys():
        choice = input(
            f"Boundary \"{name}\" → choose BC [0–4]: "
        ).strip()

        if choice == "1":
            bc_pressure_zero.append(name)


        elif choice == "2":
            bc_pressure_constant.append(name)
            if pressure_constant_value is 0:
                pressure_constant_value = float(
                    input("  Constant pressure value [Pa]: ")
                )

        elif choice == "3":
            boundary_nodes_impedance.append(name)
            if Z_impedance is 0:
                real = float(input("  Impedance real part: "))
                imag = float(input("  Impedance imaginary part: "))
                Z_impedance = real + 1j * imag

        elif choice == "4":
            bc_velocity.append(name)
            if value_velocity_normal is 0:
                value_velocity_normal = float(
                    input("  Normal velocity value [m/s]: ")
                )

        # 0 or anything else → hard wall (do nothing)

    # --- Optional point source
    add_source = input("\nAdd a point pressure source? [y/N]: ").lower()
    if add_source == "y":
        x = float(input("  Source x: "))
        y = float(input("  Source y: "))
        source_node = find_closest_node(nodes, [x, y])
        source_pressure_value = float(
            input("  Source pressure amplitude [Pa]: ")
        )

    return (
        bc_pressure_zero,
        bc_pressure_constant,
        pressure_constant_value,
        boundary_nodes_impedance,
        Z_impedance,
        bc_velocity,
        value_velocity_normal,
        source_node,
        source_pressure_value,
    )
