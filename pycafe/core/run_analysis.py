
import numpy as np

from pycafe.solver.analysis_driver import run_acoustic_analysis
from pycafe.solver.solver_helmholtz_1 import solve_helmholtz_frequency_sweep
from pycafe.build_matrices.assembly_cquad8 import expand_to_full


def run_analysis(
    *,
    system,
    nodes,
    boundaries,
    rho,
    interactive=True,
):
    """
    Run an acoustic analysis on a prepared FEM system.

    Interactive mode launches the legacy analysis driver.
    Non-interactive mode runs a default DIRECT frequency-domain analysis.
    """

    # =========================================================
    # INTERACTIVE (legacy, unchanged)
    # =========================================================
    if interactive:
        return run_acoustic_analysis(
            nodes=nodes,
            K_red=system["K_red"],
            M_red=system["M_red"],
            C_red=system["C_red"],
            idx_free=system["idx_free"],
            p0_nodes=system["p0_nodes"],
            boundaries=boundaries,
            pressure_nodes_red=system["pressure_nodes_red"],
            pressure_values=system["pressure_values"],
            bc_velocity=system["bc_velocity"],
            value_velocity_normal=system["value_velocity_normal"],
            rho=rho,
        )

    # =========================================================
    # NON-INTERACTIVE: DIRECT ANALYSIS (TEST / CI SAFE)
    # =========================================================

    # --- safe default frequency ---
    freq = 500.0  # Hz
    frequencies = np.array([freq])

    P_red = solve_helmholtz_frequency_sweep(
        system["K_red"],
        system["M_red"],
        system["C_red"],
        frequencies,
        system["pressure_nodes_red"],
        system["pressure_values"],
        nodes=nodes,
        boundary_velocity_nodes=system["bc_velocity"],
        idx_free=system["idx_free"],
        rho=rho,
        v_n=system["value_velocity_normal"],
        boundaries=boundaries,
    )

    p_full = expand_to_full(
        P_red,
        system["idx_free"],
        system["p0_nodes"],
        nodes.shape[0],
    )

    return {
        "type": "direct",
        "frequencies": frequencies,
        "p_full": p_full,
    }
