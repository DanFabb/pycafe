import numpy as np

from pycafe.boundary_condition.acoustic_bc import AcousticBC


def find_closest_node(nodes, pos_xy):
    pos_xy = np.asarray(pos_xy, dtype=float).reshape(1, 2)
    dist = np.linalg.norm(nodes[:, :2] - pos_xy, axis=1)
    return int(np.argmin(dist))


def _ask_complex(label):
    """Read a complex value as two real inputs."""
    real = float(input(f"  {label} real part: "))
    imag = float(input(f"  {label} imaginary part: "))
    return complex(real, imag)


def choose_boundary_conditions(boundaries, nodes):
    """
    Interactively assign acoustic boundary conditions to the mesh.

    This function prompts the user to select a boundary condition
    for each mesh boundary. Available types are:

    - Hard wall (default): homogeneous Neumann, adds nothing
    - Zero pressure: Dirichlet, the DOFs are eliminated
    - Constant pressure: Dirichlet with a prescribed value
    - Acoustic impedance: Robin, contributes the boundary matrix ``C``
    - Normal velocity: Neumann, contributes the load vector ``V_n``

    Each boundary is asked for its own value, so different walls can
    carry different liners or different velocities.

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
    bc : AcousticBC
        Boundary condition description, accepted directly by
        :func:`~pycafe.core.prepare_acoustic_system.prepare_acoustic_system`.

    Notes
    -----
    The normal velocity is measured along the **outward** normal of the
    fluid domain: a wall moving into the fluid has ``v_n < 0``.

    The impedance is entered as the **normalized** value
    ``zeta = Z / (rho * c0)`` unless the absolute unit is selected at
    the prompt.

    This function is intended for interactive use and requires user
    input via the terminal. It should not be used in fully automated
    workflows.

    See Also
    --------
    assign_boundary_conditions : High-level wrapper for interactive
        boundary condition assignment.
    AcousticBC : The returned boundary condition container.
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

    bc = AcousticBC()

    print("\nAssign BC to each boundary:\n")

    for name in boundaries.keys():
        choice = input(f"Boundary \"{name}\" -> choose BC [0-4]: ").strip()

        if choice == "1":
            bc.add_pressure(name, 0.0)

        elif choice == "2":
            value = float(input("  Constant pressure value [Pa]: "))
            bc.add_pressure(name, value)

        elif choice == "3":
            unit = input(
                "  Impedance unit [n = normalized (default), a = Pa*s/m]: "
            ).strip().lower()
            Z = _ask_complex("Impedance")
            bc.add_impedance(name, (Z, "abs") if unit.startswith("a") else Z)

        elif choice == "4":
            value = float(
                input("  Normal velocity [m/s, positive outward]: ")
            )
            bc.add_velocity(name, value)

        # 0 or anything else -> hard wall (nothing to add)

    # --- Optional point source
    if input("\nAdd a point pressure source? [y/N]: ").strip().lower() == "y":
        x = float(input("  Source x: "))
        y = float(input("  Source y: "))
        amplitude = float(input("  Source pressure amplitude [Pa]: "))
        bc.add_point_pressure(find_closest_node(nodes, [x, y]), amplitude)

    return bc
