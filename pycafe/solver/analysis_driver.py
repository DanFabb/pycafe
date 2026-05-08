import numpy as np


def run_acoustic_analysis(
    *,
    nodes,
    K_red,
    M_red,
    C_red,
    idx_free,
    p0_nodes,
    boundaries,
    pressure_nodes_red,
    pressure_values,
    bc_velocity,
    value_velocity_normal,
    rho,
    elements=None,
):
    """
    Launch an interactive acoustic analysis on a reduced FEM system.

    This function prompts the user to choose the type of acoustic analysis
    to perform and executes the corresponding solver and post-processing.
    Supported analysis types are:
    
    - Modal analysis (eigenvalue problem)
    - Direct frequency-domain response (Helmholtz frequency sweep)

    Parameters
    ----------
    nodes : ndarray of shape (N, 2) or (N, 3)
        Coordinates of the mesh nodes.
    K_red : scipy.sparse matrix
        Reduced acoustic stiffness matrix.
    M_red : scipy.sparse matrix
        Reduced acoustic mass matrix.
    C_red : scipy.sparse matrix
        Reduced acoustic impedance (damping) matrix.
    idx_free : ndarray of int
        Indices of the free degrees of freedom in the global system.
    p0_nodes : ndarray of int
        Indices of nodes where pressure is constrained to zero
        (Dirichlet boundary condition).
    boundaries : dict
        Dictionary defining mesh boundary groups.
    pressure_nodes_red : ndarray of int
        Indices of reduced degrees of freedom where pressure
        boundary conditions are applied.
    pressure_values : ndarray of complex
        Prescribed pressure values corresponding to
        ``pressure_nodes_red``.
    bc_velocity : list of str
        Boundary names where a normal velocity condition is applied.
    value_velocity_normal : float
        Normal velocity amplitude prescribed on velocity boundaries.
    rho : float
        Fluid density.

    Returns
    -------
    results : dict
        Dictionary containing the analysis results. The content depends
        on the selected analysis type:

        - Modal analysis:
          ``{"type": "modal", "freqs": freqs, "modes_red": modes_red}``
        - Direct frequency response:
          ``{"type": "direct", "frequencies": frequencies, "p_full": p_full}``

    Raises
    ------
    ValueError
        If an invalid analysis type is selected.

    Notes
    -----
    This function is intentionally interactive and intended for
    exploratory or educational workflows. For scripted or automated
    analyses, solver functions should be called directly.

    See Also
    --------
    prepare_acoustic_system : High-level FEM system preparation.
    solve_modal_acoustic_reduced : Modal acoustic solver.
    solve_helmholtz_frequency_sweep : Direct frequency-domain solver.
    run_modal_post_processing : Modal post-processing utilities.
    run_post_processing : Frequency-domain post-processing utilities.
    """
    print("\n=== ACOUSTIC ANALYSIS SETUP ===")
    analysis_type = input(
        "Choose analysis type [modal / direct]: "
    ).strip().lower()

    # MODAL ANALYSIS
    if analysis_type == "modal":
        from pycafe.solver.solver_modale import solve_modal_acoustic_reduced
        from pycafe.post_processing.post_processing_modal import (
            run_modal_post_processing,
        )

        num_modes = int(
            input("Number of acoustic modes to compute: ")
        )

        freqs, modes_red = solve_modal_acoustic_reduced(
            K_red,
            M_red,
            num_modes=num_modes,
        )

        run_modal_post_processing(
            nodes=nodes,
            modes_red=modes_red,
            freqs=freqs,
            idx_free=idx_free,
            p0_nodes=p0_nodes,
            elements=elements,
        )

        return {
            "type": "modal",
            "freqs": freqs,
            "modes_red": modes_red,
        }

    # DIRECT FREQUENCY RESPONSE
    
    elif analysis_type == "direct":
        from pycafe.solver.solver_helmholtz_1 import (
            solve_helmholtz_frequency_sweep,
        )
        from pycafe.build_matrices.assembly_cquad8 import expand_to_full
        from pycafe.post_processing.post_processing import run_post_processing

        print("\n--- Frequency sweep setup ---")
        fmin = float(input("Minimum frequency [Hz]: "))
        fmax = float(input("Maximum frequency [Hz]: "))
        df = float(input("Frequency step [Hz]: "))

        frequencies = np.arange(fmin, fmax + df, df)

        P_red = solve_helmholtz_frequency_sweep(
            K_red,
            M_red,
            C_red,
            frequencies,
            pressure_nodes_red,
            pressure_values,
            nodes=nodes,
            boundary_velocity_nodes=bc_velocity,
            idx_free=idx_free,
            rho=rho,
            v_n=value_velocity_normal,
            boundaries=boundaries,
        )

        p_full = expand_to_full(
            P_red,
            idx_free,
            p0_nodes,
            nodes.shape[0],
        )

        run_post_processing(
            nodes=nodes,
            p_full=p_full,
            frequencies=frequencies,
            elements=elements,
        )

        return {
            "type": "direct",
            "frequencies": frequencies,
            "p_full": p_full,
        }

    # INVALID OPTION
    else:
        raise ValueError(
            "Invalid analysis type. Choose 'modal' or 'direct'."
        )
