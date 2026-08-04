"""
Validation tests for the Helmholtz frequency-domain solver.

Tests verify:
- Single-frequency solver returns complex pressure array of correct size
- Prescribed pressure BCs are enforced exactly at the specified nodes
- System fully constrained raises RuntimeError
- Normal velocity RHS assembles non-zero contributions
- Frequency sweep returns correct shape output

Reviewers addressed: R1, R3 (Quality Control / solver testing)
"""

import numpy as np
import pytest
from scipy.sparse import lil_matrix, eye as speye

from pycafe.solver.solver_helmholtz_1 import (
    solve_helmholtz_single_frequency,
    solve_helmholtz_frequency_sweep,
    build_normal_velocity_rhs,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_trivial_system(n=10):
    """
    Build a trivial acoustic system: identity K and M, zero C.
    Useful for checking solver mechanics independent of assembly.
    """
    K = speye(n, format="csr") * 1.0
    M = speye(n, format="csr") * 1.0
    C = lil_matrix((n, n), dtype=complex)
    return K, M, C


# ---------------------------------------------------------------------------
# Tests: solve_helmholtz_single_frequency
# ---------------------------------------------------------------------------

class TestHelmholtzSingleFrequency:

    def test_output_shape(self):
        """Solver returns array of correct size."""
        n = 10
        K, M, C = make_trivial_system(n)
        omega = 2 * np.pi * 100.0

        # Prescribe pressure at node 0 only
        p_red = solve_helmholtz_single_frequency(
            K, M, C, omega,
            pressure_nodes_red=np.array([0]),
            pressure_values=np.array([1.0 + 0j]),
        )
        assert p_red.shape == (n,)

    def test_output_is_complex(self):
        """Solver returns complex-valued pressure array."""
        n = 6
        K, M, C = make_trivial_system(n)
        omega = 2 * np.pi * 50.0

        p_red = solve_helmholtz_single_frequency(
            K, M, C, omega,
            pressure_nodes_red=np.array([0]),
            pressure_values=np.array([1.0 + 0j]),
        )
        assert p_red.dtype == complex or np.iscomplexobj(p_red)

    def test_prescribed_pressure_exactly_enforced(self):
        """Prescribed pressure nodes must have exactly the imposed value."""
        n = 8
        K, M, C = make_trivial_system(n)
        omega = 2 * np.pi * 100.0
        p_imposed = 2.5 + 1.0j

        constrained = np.array([0, 3, 5])
        values = np.full(len(constrained), p_imposed)

        p_red = solve_helmholtz_single_frequency(
            K, M, C, omega,
            pressure_nodes_red=constrained,
            pressure_values=values,
        )

        for idx, val in zip(constrained, values):
            assert abs(p_red[idx] - val) < 1e-10, (
                f"Pressure not enforced at node {idx}: "
                f"p={p_red[idx]:.6f}, expected={val}"
            )

    def test_fully_constrained_raises(self):
        """If all DOFs are prescribed, solver raises RuntimeError."""
        n = 3
        K, M, C = make_trivial_system(n)
        omega = 100.0

        with pytest.raises(RuntimeError):
            solve_helmholtz_single_frequency(
                K, M, C, omega,
                pressure_nodes_red=np.array([0, 1, 2]),
                pressure_values=np.array([1.0, 1.0, 1.0]),
            )

    def test_zero_frequency_no_crash(self):
        """Solver should handle omega=0 without crashing (static limit)."""
        n = 5
        K, M, C = make_trivial_system(n)
        omega = 0.0  # DC / static case

        p_red = solve_helmholtz_single_frequency(
            K, M, C, omega,
            pressure_nodes_red=np.array([0]),
            pressure_values=np.array([1.0 + 0j]),
        )
        assert p_red.shape == (n,)

    def test_velocity_rhs_contributes(self):
        """Adding a velocity RHS changes the solution for free DOFs."""
        n = 5
        K, M, C = make_trivial_system(n)
        omega = 2 * np.pi * 100.0

        # Constrain only node 0
        p_nodes = np.array([0])
        p_vals = np.array([0.0 + 0j])

        # Without velocity
        p_no_vel = solve_helmholtz_single_frequency(
            K, M, C, omega,
            pressure_nodes_red=p_nodes,
            pressure_values=p_vals,
        )

        # With velocity excitation at node 2
        p_vel = solve_helmholtz_single_frequency(
            K, M, C, omega,
            pressure_nodes_red=p_nodes,
            pressure_values=p_vals,
            velocity_nodes_red=np.array([2]),
            velocity_values=np.array([1.0 + 0j]),
        )

        # Solution must differ at free DOFs when velocity is applied
        assert not np.allclose(p_no_vel, p_vel), (
            "Velocity RHS had no effect on the solution"
        )


# ---------------------------------------------------------------------------
# Tests: solve_helmholtz_frequency_sweep
# ---------------------------------------------------------------------------

class TestHelmholtzFrequencySweep:

    def test_output_shape(self):
        """Frequency sweep returns (N_red, N_freq) array."""
        n = 8
        K, M, C = make_trivial_system(n)
        frequencies = np.array([50.0, 100.0, 200.0])

        P_red = solve_helmholtz_frequency_sweep(
            K, M, C, frequencies,
            pressure_nodes_red=np.array([0]),
            pressure_values=np.array([1.0 + 0j]),
        )
        assert P_red.shape == (n, len(frequencies))

    def test_single_frequency_sweep_matches_single_solve(self):
        """Frequency sweep with one frequency matches single-frequency solve."""
        n = 6
        K, M, C = make_trivial_system(n)
        omega = 2 * np.pi * 100.0
        freq = np.array([100.0])

        P_sweep = solve_helmholtz_frequency_sweep(
            K, M, C, freq,
            pressure_nodes_red=np.array([0]),
            pressure_values=np.array([1.0 + 0j]),
        )

        p_single = solve_helmholtz_single_frequency(
            K, M, C, omega,
            pressure_nodes_red=np.array([0]),
            pressure_values=np.array([1.0 + 0j]),
        )

        np.testing.assert_allclose(P_sweep[:, 0], p_single, atol=1e-12)

    def test_prescribed_pressures_enforced_at_all_frequencies(self):
        """Prescribed pressure nodes are exactly enforced at every frequency."""
        n = 8
        K, M, C = make_trivial_system(n)
        frequencies = np.linspace(10, 500, 20)
        p_imposed = 3.0 + 0j
        constrained = np.array([1])
        values = np.array([p_imposed])

        P_red = solve_helmholtz_frequency_sweep(
            K, M, C, frequencies,
            pressure_nodes_red=constrained,
            pressure_values=values,
        )

        for i_f in range(len(frequencies)):
            for idx, val in zip(constrained, values):
                assert abs(P_red[idx, i_f] - val) < 1e-10, (
                    f"Pressure not enforced at node {idx}, freq idx {i_f}"
                )

    def test_output_is_complex(self):
        """Sweep output must be a complex array."""
        n = 5
        K, M, C = make_trivial_system(n)
        freqs = np.array([100.0, 200.0])
        P = solve_helmholtz_frequency_sweep(
            K, M, C, freqs,
            pressure_nodes_red=np.array([0]),
            pressure_values=np.array([1.0 + 0j]),
        )
        assert np.iscomplexobj(P)


# ---------------------------------------------------------------------------
# Tests: build_normal_velocity_rhs
# ---------------------------------------------------------------------------

class TestNormalVelocityRHS:

    def test_empty_boundary_returns_empty(self):
        """Empty velocity boundary returns empty arrays."""
        nodes = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        idx_free = np.array([0, 1, 2, 3])
        vel_nodes, vel_vals = build_normal_velocity_rhs(
            nodes=nodes,
            boundary_velocity_nodes=[],
            idx_free=idx_free,
            rho=1.2,
            omega=2 * np.pi * 100.0,
            v_n=0.01,
        )
        assert vel_nodes.size == 0
        assert vel_vals.size == 0

    def test_nonzero_rhs_for_active_boundary(self):
        """Non-empty velocity boundary generates non-zero RHS contributions."""
        nodes = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ])
        idx_free = np.array([0, 1, 2, 3])
        # Bottom edge: nodes 0 and 1 (1-based: 1 and 2)
        vel_nodes, vel_vals = build_normal_velocity_rhs(
            nodes=nodes,
            boundary_velocity_nodes=[1, 2],  # 1-based
            idx_free=idx_free,
            rho=1.2,
            omega=2 * np.pi * 100.0,
            v_n=0.01,
        )
        assert vel_nodes.size > 0
        assert np.any(np.abs(vel_vals) > 0)

    def test_rhs_scales_linearly_with_velocity(self):
        """Doubling v_n doubles the RHS contribution (linear dependence)."""
        nodes = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ])
        idx_free = np.array([0, 1, 2, 3])
        kwargs = dict(
            nodes=nodes,
            boundary_velocity_nodes=[1, 2],
            idx_free=idx_free,
            rho=1.2,
            omega=2 * np.pi * 100.0,
        )

        _, vals1 = build_normal_velocity_rhs(**kwargs, v_n=0.01)
        _, vals2 = build_normal_velocity_rhs(**kwargs, v_n=0.02)

        np.testing.assert_allclose(vals2, 2.0 * vals1, rtol=1e-12)

    def test_rhs_imaginary_at_nonzero_omega(self):
        """
        The velocity RHS contribution is proportional to i*omega*rho*v_n,
        so for a real v_n and real omega, the contribution is purely imaginary.
        """
        nodes = np.array([[0.0, 0.0], [1.0, 0.0]])
        idx_free = np.array([0, 1])
        _, vals = build_normal_velocity_rhs(
            nodes=nodes,
            boundary_velocity_nodes=[1, 2],
            idx_free=idx_free,
            rho=1.2,
            omega=2 * np.pi * 100.0,
            v_n=1.0,
        )
        # Imaginary part should dominate (real part numerically zero)
        if vals.size > 0:
            for v in vals:
                assert abs(v.real) < 1e-12, (
                    f"Expected purely imaginary RHS, got real part: {v.real}"
                )
