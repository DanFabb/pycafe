"""
Validation tests for volumetric acoustic sources and the modal
frequency-sweep solver.

Covered:
- volume load vector ``int N dV`` (area/volume, partition of unity);
- point-source shape functions: at a node, at an element centroid, at
  an arbitrary point (Newton inversion of the isoparametric map);
- source operator ``Q(omega) = j rho0 omega q g``: sign, scaling,
  distributed vs point;
- modal basis: mass-normalization, rigid mode handling, natural
  frequencies of a duct;
- modal damping: Rayleigh projection and direct modal ratios;
- modal vs direct sweep on a duct excited by a monopole and by a
  piston, with rigid and with impedance walls (mode coupling path).
"""

import numpy as np
import pytest

from pycafe.boundary_condition.acoustic_bc import (
    AcousticBC,
    build_impedance_operator,
    build_source_operator,
    build_velocity_operator,
)
from pycafe.build_matrices.assembly_dispatcher import build_KM_acoustic
from pycafe.build_matrices.source_volume import (
    point_source_shape,
    volume_load_vector,
)
from pycafe.solver.solver_helmholtz_1 import solve_helmholtz_frequency_sweep
from pycafe.solver.solver_modal_forced import (
    build_modal_basis,
    modal_damping_matrix,
    solve_modal_frequency_sweep,
)

from tests.test_bc_dynamic import duct_mesh_2d, duct_mesh_3d, RHO, C0


# ---------------------------------------------------------------------------
# Tests: volume load vector
# ---------------------------------------------------------------------------

class TestVolumeLoadVector:

    def test_2d_area(self):
        nodes, elements, _ = duct_mesh_2d(nx=8, ny=3)
        g = volume_load_vector(nodes, elements, nodes.shape[0])
        assert np.isclose(g.sum(), 1.0 * 0.1)      # Lx * H

    def test_3d_volume(self):
        nodes, elements, _ = duct_mesh_3d(nx=5, ny=2, nz=2)
        g = volume_load_vector(nodes, elements, nodes.shape[0])
        assert np.isclose(g.sum(), 1.0 * 0.1 * 0.1)

    def test_boundary_faces_are_skipped(self):
        """Quadrilateral 4 groups on a HEX8 mesh are walls, not domain."""
        nodes, elements, _ = duct_mesh_3d(nx=4)
        g_with = volume_load_vector(nodes, elements, nodes.shape[0])
        g_without = volume_load_vector(
            nodes, {"Hexahedron 8": elements["Hexahedron 8"]}, nodes.shape[0]
        )
        np.testing.assert_allclose(g_with, g_without)

    def test_no_domain_elements_raises(self):
        nodes = np.zeros((2, 3))
        with pytest.raises(ValueError, match="No registered acoustic domain"):
            volume_load_vector(nodes, {"Line 2": np.array([[1, 2]])}, 2)


# ---------------------------------------------------------------------------
# Tests: point source location
# ---------------------------------------------------------------------------

class TestPointSourceShape:

    def test_source_at_node_2d(self):
        nodes, elements, _ = duct_mesh_2d(nx=4, ny=2)
        target = 7
        idx, N = point_source_shape(nodes, elements, nodes[target, :2])
        f = np.zeros(nodes.shape[0])
        f[idx] = N
        expected = np.zeros(nodes.shape[0])
        expected[target] = 1.0
        np.testing.assert_allclose(f, expected, atol=1e-9)

    def test_source_at_centroid_splits_evenly(self):
        """Centroid of a rectangular QUAD4: N = 1/4 on each corner."""
        nodes, elements, _ = duct_mesh_2d(nx=2, ny=1, L=1.0, H=0.1)
        centroid = np.array([0.25, 0.05])
        idx, N = point_source_shape(nodes, elements, centroid)
        np.testing.assert_allclose(np.sort(N), np.full(4, 0.25), atol=1e-9)

    def test_partition_of_unity_3d(self):
        nodes, elements, _ = duct_mesh_3d(nx=4, ny=2, nz=2)
        x_s = np.array([0.37, 0.041, 0.083])
        idx, N = point_source_shape(nodes, elements, x_s)
        assert np.isclose(N.sum(), 1.0)
        assert np.all(N > -1e-9)
        # The interpolated position must be the source position.
        np.testing.assert_allclose(N @ nodes[idx, :3], x_s, atol=1e-9)

    def test_point_outside_raises(self):
        nodes, elements, _ = duct_mesh_2d(nx=3, ny=1)
        with pytest.raises(ValueError, match="No element contains"):
            point_source_shape(nodes, elements, [2.5, 0.05])


# ---------------------------------------------------------------------------
# Tests: source operator
# ---------------------------------------------------------------------------

class TestSourceOperator:

    def test_nodal_monopole_sign_and_value(self):
        """Q_i = +j rho0 omega q_s on the source node, zero elsewhere."""
        nodes, elements, _ = duct_mesh_2d(nx=3, ny=1)
        q_s = 1e-3
        omega = 2 * np.pi * 150.0

        op = build_source_operator(
            AcousticBC().add_monopole(q_s, node=5),
            nodes=nodes, rho=RHO, elements=elements,
        )
        Q = op.at(omega)

        expected = np.zeros(nodes.shape[0], dtype=complex)
        expected[5] = 1j * RHO * omega * q_s
        np.testing.assert_allclose(Q, expected)

    def test_monopole_at_position_conserves_volume_velocity(self):
        """sum_a Q_a = j rho omega q_s regardless of placement."""
        nodes, elements, _ = duct_mesh_3d(nx=4, ny=2, nz=2)
        q_s, omega = 2e-3, 2 * np.pi * 90.0
        op = build_source_operator(
            AcousticBC().add_monopole(q_s, position=[0.4, 0.05, 0.05]),
            nodes=nodes, rho=RHO, elements=elements,
        )
        assert np.isclose(op.at(omega).sum(), 1j * RHO * omega * q_s)

    def test_distributed_source_total(self):
        """sum_a Q_a = j rho omega q V for a constant density q."""
        nodes, elements, _ = duct_mesh_2d(nx=6, ny=2)
        q, omega = 3.0, 2 * np.pi * 60.0
        op = build_source_operator(
            AcousticBC().add_distributed_source(q),
            nodes=nodes, rho=RHO, elements=elements,
        )
        V = 1.0 * 0.1
        assert np.isclose(op.at(omega).sum(), 1j * RHO * omega * q * V)

    def test_frequency_dependent_strength(self):
        nodes, elements, _ = duct_mesh_2d(nx=3, ny=1)
        op = build_source_operator(
            AcousticBC().add_monopole(lambda w: 1.0 / w, node=2),
            nodes=nodes, rho=RHO, elements=elements,
        )
        # q ~ 1/omega cancels the omega factor.
        np.testing.assert_allclose(op.at(100.0), op.at(1000.0))

    def test_zero_strength_is_empty(self):
        nodes, elements, _ = duct_mesh_2d(nx=3, ny=1)
        op = build_source_operator(
            AcousticBC().add_monopole(0.0, node=2),
            nodes=nodes, rho=RHO, elements=elements,
        )
        assert op.is_empty

    def test_add_monopole_requires_one_placement(self):
        with pytest.raises(ValueError, match="exactly one"):
            AcousticBC().add_monopole(1.0)
        with pytest.raises(ValueError, match="exactly one"):
            AcousticBC().add_monopole(1.0, node=1, position=[0, 0])

    def test_legacy_tuple_cannot_hold_monopoles(self):
        bc = AcousticBC().add_monopole(1.0, node=0)
        with pytest.raises(ValueError, match="Monopole"):
            bc.to_legacy()

    def test_negative_node_is_rejected(self):
        """
        NumPy would happily read node=-1 as the last node, moving the
        source somewhere else instead of failing.
        """
        with pytest.raises(ValueError, match="non-negative"):
            AcousticBC().add_monopole(1.0, node=-1)
        with pytest.raises(ValueError, match="non-negative"):
            AcousticBC().add_point_pressure(-1, 1.0)

    def test_node_outside_the_mesh_is_rejected(self):
        nodes, elements, _ = duct_mesh_2d(nx=3, ny=1)
        bc = AcousticBC().add_monopole(1.0, node=0)
        # The mesh size is unknown when the BC is declared, so the range
        # is checked at assembly: edit the entry to reach that guard.
        bc.monopoles[0].node = nodes.shape[0]
        with pytest.raises(ValueError, match="outside the mesh"):
            build_source_operator(
                bc, nodes=nodes, rho=RHO, elements=elements
            )


# ---------------------------------------------------------------------------
# Tests: modal basis
# ---------------------------------------------------------------------------

class TestModalBasis:

    def test_mass_normalization(self):
        nodes, elements, _ = duct_mesh_2d(nx=20, ny=1)
        K, M, _, _ = build_KM_acoustic(nodes, elements, C0)
        omegas, Phi = build_modal_basis(K.tocsr(), M.tocsr(), 6)
        Mt = Phi.T @ (M.tocsr() @ Phi)
        np.testing.assert_allclose(Mt, np.eye(omegas.size), atol=1e-8)
        Kt = Phi.T @ (K.tocsr() @ Phi)
        # atol covers the rigid mode, whose omega^2 is numerically ~0.
        np.testing.assert_allclose(
            np.diagonal(Kt), omegas**2, rtol=1e-8, atol=1e-6
        )

    def test_rigid_mode_kept_by_default(self):
        nodes, elements, _ = duct_mesh_2d(nx=20, ny=1)
        K, M, _, _ = build_KM_acoustic(nodes, elements, C0)
        omegas, _ = build_modal_basis(K.tocsr(), M.tocsr(), 4)
        assert np.isclose(omegas[0], 0.0, atol=1e-3)

        omegas_nr, _ = build_modal_basis(
            K.tocsr(), M.tocsr(), 4, include_rigid=False
        )
        assert omegas_nr[0] > 1.0

    def test_duct_natural_frequencies(self):
        """Rigid-rigid duct: f_n = n c0 / (2 L)."""
        nodes, elements, _ = duct_mesh_2d(nx=80, ny=1)
        K, M, _, _ = build_KM_acoustic(nodes, elements, C0)
        omegas, _ = build_modal_basis(
            K.tocsr(), M.tocsr(), 4, include_rigid=False
        )
        f_num = omegas / (2 * np.pi)
        f_exact = C0 / 2.0 * np.array([1, 2, 3, 4])
        np.testing.assert_allclose(f_num, f_exact, rtol=2e-3)


class TestModalDamping:

    def test_rayleigh_projection(self):
        """zeta_m = alpha omega_m / 2 + beta / (2 omega_m)."""
        omegas = np.array([100.0, 400.0])
        alpha, beta = 1e-5, 2.0
        Ct = modal_damping_matrix(omegas, rayleigh=(alpha, beta))
        expected = alpha * omegas**2 + beta
        np.testing.assert_allclose(np.diagonal(Ct), expected)

    def test_rayleigh_regular_at_rigid_mode(self):
        Ct = modal_damping_matrix(np.array([0.0, 100.0]), rayleigh=(0.0, 3.0))
        assert np.isclose(Ct[0, 0], 3.0)      # beta*M projects to beta

    def test_modal_zeta(self):
        omegas = np.array([100.0, 200.0])
        Ct = modal_damping_matrix(omegas, modal_zeta=0.02)
        np.testing.assert_allclose(np.diagonal(Ct), 2 * 0.02 * omegas)


# ---------------------------------------------------------------------------
# Tests: modal vs direct sweep
# ---------------------------------------------------------------------------

def _system(nodes, elements, boundaries, bc):
    from pycafe.core.prepare_acoustic_system import prepare_acoustic_system
    return prepare_acoustic_system(
        nodes=nodes, elements=elements, boundaries=boundaries,
        rho=RHO, c0=C0, bc=bc,
    )


class TestModalVsDirect:

    def test_monopole_in_rigid_duct(self):
        """With the full basis the modal answer equals the direct one."""
        nodes, elements, boundaries = duct_mesh_2d(nx=30, ny=1)
        bc = AcousticBC().add_monopole(1e-3, position=[0.3, 0.05])
        system = _system(nodes, elements, boundaries, bc)

        freqs = [95.0, 240.0]
        n_red = system["K_red"].shape[0]

        P_direct = solve_helmholtz_frequency_sweep(
            system["K_red"].tocsr(), system["M_red"].tocsr(),
            system["C_red"], freqs,
            system["pressure_nodes_red"], system["pressure_values"],
            source_operator=system["source_red_op"],
        )
        P_modal = solve_modal_frequency_sweep(
            system["K_red"], system["M_red"], freqs,
            num_modes=n_red,                     # complete basis: exact
            source_operator=system["source_red_op"],
        )
        np.testing.assert_allclose(P_modal, P_direct, rtol=1e-6, atol=1e-9)

    def test_truncated_basis_approximates_direct(self):
        nodes, elements, boundaries = duct_mesh_2d(nx=60, ny=1)
        bc = AcousticBC().add_velocity("inlet", -0.01)
        system = _system(nodes, elements, boundaries, bc)

        freqs = np.array([80.0, 150.0])          # below f_1 = 170 Hz

        P_direct = solve_helmholtz_frequency_sweep(
            system["K_red"].tocsr(), system["M_red"].tocsr(),
            system["C_red"], freqs,
            system["pressure_nodes_red"], system["pressure_values"],
            velocity_operator=system["velocity_red_op"],
        )
        # Truncation error decays with the basis size (the out-of-band
        # modes only contribute their quasi-static tails ~ 1/omega_m^2).
        errors = []
        for num_modes in (6, 12, 48):
            P_modal = solve_modal_frequency_sweep(
                system["K_red"], system["M_red"], freqs,
                num_modes=num_modes,
                velocity_operator=system["velocity_red_op"],
            )
            errors.append(
                np.abs(P_modal - P_direct).max() / np.abs(P_direct).max()
            )

        assert errors[0] < 5e-2, f"truncation error {errors[0]:.2e}"
        assert errors[2] < errors[1] < errors[0]
        assert errors[2] < 1e-2

    def test_impedance_couples_modes_and_matches_direct(self):
        """Projected C is non-diagonal: dense modal solve path."""
        nodes, elements, boundaries = duct_mesh_2d(nx=30, ny=1)
        bc = (AcousticBC()
              .add_velocity("inlet", -0.01)
              .add_impedance("outlet", 1.0))
        system = _system(nodes, elements, boundaries, bc)

        freqs = [120.0, 260.0]
        n_red = system["K_red"].shape[0]

        P_direct = solve_helmholtz_frequency_sweep(
            system["K_red"].tocsr(), system["M_red"].tocsr(),
            system["C_red"], freqs,
            system["pressure_nodes_red"], system["pressure_values"],
            velocity_operator=system["velocity_red_op"],
            impedance_operator=system["C_red_op"],
        )
        P_modal = solve_modal_frequency_sweep(
            system["K_red"], system["M_red"], freqs,
            num_modes=n_red,
            velocity_operator=system["velocity_red_op"],
            impedance_operator=system["C_red_op"],
        )
        np.testing.assert_allclose(P_modal, P_direct, rtol=1e-6, atol=1e-9)

    def test_3d_monopole_modal_vs_direct(self):
        nodes, elements, boundaries = duct_mesh_3d(nx=12, ny=1, nz=1)
        bc = AcousticBC().add_monopole(1e-3, position=[0.31, 0.05, 0.05])
        system = _system(nodes, elements, boundaries, bc)

        freqs = [100.0]
        n_red = system["K_red"].shape[0]

        P_direct = solve_helmholtz_frequency_sweep(
            system["K_red"].tocsr(), system["M_red"].tocsr(),
            system["C_red"], freqs,
            system["pressure_nodes_red"], system["pressure_values"],
            source_operator=system["source_red_op"],
        )
        P_modal = solve_modal_frequency_sweep(
            system["K_red"], system["M_red"], freqs,
            num_modes=n_red,
            source_operator=system["source_red_op"],
        )
        np.testing.assert_allclose(P_modal, P_direct, rtol=1e-6, atol=1e-9)

    def test_participations_peak_at_resonant_mode(self):
        """Near f_1 the first axial mode dominates the participation."""
        nodes, elements, boundaries = duct_mesh_2d(nx=60, ny=1)
        bc = AcousticBC().add_velocity("inlet", -0.01)
        system = _system(nodes, elements, boundaries, bc)

        f1 = C0 / 2.0                             # 170 Hz
        _, modal = solve_modal_frequency_sweep(
            system["K_red"], system["M_red"], [f1 * 0.995],
            num_modes=8,
            velocity_operator=system["velocity_red_op"],
            modal_zeta=0.01,
            return_modal=True,
        )
        dominant = np.argmax(np.abs(modal["participations"][:, 0]))
        # Mode 0 is the rigid mode; the first axial mode is index 1.
        assert dominant == 1

    def test_no_excitation_raises(self):
        nodes, elements, boundaries = duct_mesh_2d(nx=10, ny=1)
        system = _system(nodes, elements, boundaries, AcousticBC())
        with pytest.raises(ValueError, match="No excitation"):
            solve_modal_frequency_sweep(
                system["K_red"], system["M_red"], [100.0], num_modes=4,
            )

    def test_basis_reuse(self):
        nodes, elements, boundaries = duct_mesh_2d(nx=20, ny=1)
        bc = AcousticBC().add_velocity("inlet", -0.01)
        system = _system(nodes, elements, boundaries, bc)

        basis = build_modal_basis(system["K_red"], system["M_red"], 6)
        P1 = solve_modal_frequency_sweep(
            system["K_red"], system["M_red"], [120.0],
            num_modes=6, basis=basis,
            velocity_operator=system["velocity_red_op"],
        )
        P2 = solve_modal_frequency_sweep(
            system["K_red"], system["M_red"], [120.0],
            num_modes=6,
            velocity_operator=system["velocity_red_op"],
        )
        np.testing.assert_allclose(P1, P2, rtol=1e-8)


# ---------------------------------------------------------------------------
# Tests: direct sweep with source operator through the system dict
# ---------------------------------------------------------------------------

class TestSourceThroughSystem:

    def test_prepare_exposes_source_operators(self):
        nodes, elements, boundaries = duct_mesh_2d(nx=10, ny=1)
        bc = AcousticBC().add_monopole(1e-3, node=4)
        system = _system(nodes, elements, boundaries, bc)
        assert not system["source_op"].is_empty
        assert system["source_red_op"].shape == (system["idx_free"].size,)

    def test_monopole_and_piston_superpose(self):
        """Linear system: responses to separate loads add up."""
        nodes, elements, boundaries = duct_mesh_2d(nx=20, ny=1)
        freqs = [130.0]

        bc_a = AcousticBC().add_monopole(1e-3, position=[0.4, 0.05])
        bc_b = AcousticBC().add_velocity("inlet", -0.01)
        bc_ab = (AcousticBC()
                 .add_monopole(1e-3, position=[0.4, 0.05])
                 .add_velocity("inlet", -0.01))

        def solve(bc):
            s = _system(nodes, elements, boundaries, bc)
            return solve_helmholtz_frequency_sweep(
                s["K_red"].tocsr(), s["M_red"].tocsr(), s["C_red"], freqs,
                s["pressure_nodes_red"], s["pressure_values"],
                velocity_operator=s["velocity_red_op"],
                source_operator=s["source_red_op"],
            )

        np.testing.assert_allclose(
            solve(bc_ab), solve(bc_a) + solve(bc_b), rtol=1e-9,
        )

    def test_point_pressure_outside_the_mesh_is_rejected(self):
        """
        An out-of-range node matches nothing in the reduced system, so
        without a check the condition would just disappear.
        """
        nodes, elements, boundaries = duct_mesh_2d(nx=4, ny=1)
        bc = AcousticBC().add_point_pressure(0, 1.0)
        bc.point_pressure[0].node = nodes.shape[0] + 3
        with pytest.raises(ValueError, match="outside the mesh"):
            _system(nodes, elements, boundaries, bc)
