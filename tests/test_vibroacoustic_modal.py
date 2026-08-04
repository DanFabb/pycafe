"""
Tests for the modal reduction of the coupled vibroacoustic problem.

Two bases are covered, both optional accelerations of the direct sweep:

- **coupled modes** — left modes obtained from the right ones without a
  second eigensolve, biorthogonality, projected matrices;
- **CMS** — dry structural modes plus rigid-wall acoustic modes, and the
  reduced block matrices ``[[Lambda_s, -A], [0, Lambda_a]]`` and
  ``[[I, 0], [rho0 A^T, I]]`` they produce;

plus the forced response of both against the direct solver, their
convergence, and the guards.
"""

import pathlib
import sys
import tempfile

import numpy as np
import pytest

from pycafe.core.prepare_vibroacoustic_system import (
    prepare_vibroacoustic_system,
)
from pycafe.solver.solver_vibroacoustic import (
    build_coupled_blocks,
    coupled_modal_matrices,
    solve_vibroacoustic_frequency_sweep,
)
from pycafe.solver.solver_vibroacoustic_modal import (
    build_cms_basis,
    build_coupled_modal_basis,
    left_modes_from_right,
    project_coupled_system,
    solve_coupled_modal_frequency_sweep,
)

LX, LY, LZ = 0.40, 0.30, 0.50
NX, NY, NZ = 6, 5, 6
RHO0, C0 = 1.204, 343.0
T, RHO_S, E, NU = 1.0e-3, 2700.0, 70.0e9, 0.33      # aluminium, 1 mm
ETA_S = 0.02


@pytest.fixture(scope="module")
def system():
    pytest.importorskip("gmsh")
    repo_root = pathlib.Path(__file__).parent.parent
    sys.path.insert(0, str(repo_root / "coupling_geometry"))
    from make_box_plate_mesh import make_box_plate_mesh
    from pycafe.create_geom.visualize_mesh import load_mesh_with_groups

    with tempfile.TemporaryDirectory() as tmp:
        msh = make_box_plate_mesh(
            Lx=LX, Ly=LY, Lz=LZ, nx=NX, ny=NY, nz=NZ,
            output_path=pathlib.Path(tmp) / "plate_cavity.msh", verbose=False,
        )
        nodes, _, _, groups = load_mesh_with_groups(str(msh), verbose=False)

    sys_ = prepare_vibroacoustic_system(
        nodes=nodes, groups=groups, rho0=RHO0, c0=C0,
        t=T, rho_s=RHO_S, E=E, nu=NU, support="clamped",
    )
    sys_["_nodes"] = nodes
    return sys_


@pytest.fixture(scope="module")
def blocks(system):
    return build_coupled_blocks(system)


@pytest.fixture(scope="module")
def load(system, blocks):
    """1 N normal at the node closest to the centre of the plate."""
    nodes = system["_nodes"]
    idx_s = system["structural"]["idx_free"]
    node_of_dof, component = idx_s // 6, idx_s % 6
    plate = np.unique(node_of_dof)
    centre = np.array([LX / 2, LY / 2, LZ])
    target = plate[np.argmin(np.linalg.norm(nodes[plate] - centre, axis=1))]
    dof = int(np.where((node_of_dof == target) & (component == 2))[0][0])
    F_s = np.zeros(blocks["n_s"])
    F_s[dof] = 1.0
    return F_s, dof


@pytest.fixture(scope="module")
def frequencies():
    return np.arange(20.0, 300.0, 10.0)


@pytest.fixture(scope="module")
def direct(system, load, frequencies):
    F_s, _ = load
    damped = build_coupled_blocks(system, eta_s=ETA_S)
    return solve_vibroacoustic_frequency_sweep(
        system, frequencies, F_s=F_s, blocks=damped, verbose=False,
    )


def _global_error(modal, direct):
    return (np.linalg.norm(modal["w"] - direct["w"])
            / np.linalg.norm(direct["w"]))


# ---------------------------------------------------------------------------
# Coupled modes: left modes and biorthogonality
# ---------------------------------------------------------------------------

class TestCoupledBasis:

    def test_left_modes_solve_the_transposed_problem(self, blocks):
        basis = build_coupled_modal_basis(blocks=blocks, num_modes=8)
        K, M = coupled_modal_matrices(blocks)

        lhs = K.T @ basis["Phi_L"]
        rhs = M.T @ (basis["Phi_L"] * basis["omegas"] ** 2)
        scales = np.maximum(np.linalg.norm(lhs, axis=0),
                            np.linalg.norm(rhs, axis=0))
        # The omega = 0 mode makes both sides vanish (its left vector is
        # [0; uniform pressure]); scaling on the basis as a whole keeps
        # that null case from being measured against itself.
        residual = np.linalg.norm(lhs - rhs, axis=0) / scales.max()
        assert residual.max() < 1e-6

    def test_left_modes_differ_from_the_right_ones(self, blocks):
        basis = build_coupled_modal_basis(blocks=blocks, num_modes=6)
        # The structural halves are scaled by rho0 omega^2: on a coupled
        # problem the two families are genuinely different.
        assert not np.allclose(basis["Phi_L"], basis["Phi_R"])

    def test_transform_is_applied_where_expected(self):
        Phi_R = np.arange(12, dtype=float).reshape(6, 2)
        omegas = np.array([2.0, 3.0])
        Phi_L = left_modes_from_right(Phi_R, omegas, n_s=4, rho0=1.2)
        assert np.allclose(Phi_L[4:], Phi_R[4:])                  # acoustic
        assert np.allclose(Phi_L[:4, 0], 1.2 * 4.0 * Phi_R[:4, 0])
        assert np.allclose(Phi_L[:4, 1], 1.2 * 9.0 * Phi_R[:4, 1])

    def test_basis_is_biorthonormal(self, blocks):
        basis = build_coupled_modal_basis(blocks=blocks, num_modes=10)
        assert basis["biorthogonality"] < 1e-3
        _, _, Mt = project_coupled_system(basis)
        assert np.allclose(np.real(Mt), np.eye(Mt.shape[0]), atol=1e-3)

    def test_projected_stiffness_is_the_squared_frequencies(self, blocks):
        basis = build_coupled_modal_basis(blocks=blocks, num_modes=10)
        Kt, _, _ = project_coupled_system(basis)
        omega2 = basis["omegas"] ** 2
        assert np.allclose(np.real(np.diag(Kt)), omega2,
                           rtol=1e-6, atol=1e-6 * omega2.max())


# ---------------------------------------------------------------------------
# CMS basis
# ---------------------------------------------------------------------------

class TestCMSBasis:

    def test_components_are_the_uncoupled_systems(self, system, blocks):
        """Dry plate and rigid cavity, exactly as solved on their own."""
        from scipy.sparse.linalg import eigsh
        from pycafe.solver.solver_modale import solve_modal_acoustic_reduced

        basis = build_cms_basis(blocks=blocks, num_structural=4,
                                num_acoustic=5)

        vals, _ = eigsh(blocks["Ks"].tocsc(), k=4,
                        M=blocks["Ms"].tocsc(), sigma=0.0, which="LM")
        f_plate = np.sqrt(np.abs(np.sort(vals))) / (2 * np.pi)
        assert np.allclose(basis["omegas_s"] / (2 * np.pi), f_plate, rtol=1e-6)

        f_cav, _ = solve_modal_acoustic_reduced(
            blocks["Ka"], blocks["Ma"], num_modes=4
        )
        # The first acoustic vector is the uniform-pressure mode at 0 Hz.
        assert basis["omegas_a"][0] / (2 * np.pi) < 1e-6
        assert np.allclose(basis["omegas_a"][1:5] / (2 * np.pi), f_cav,
                           rtol=1e-6)

    def test_basis_is_block_diagonal(self, blocks):
        basis = build_cms_basis(blocks=blocks, num_structural=4,
                                num_acoustic=6)
        n_s, m_s = basis["n_s"], basis["num_structural"]
        Phi = basis["Phi_R"]
        assert np.all(Phi[n_s:, :m_s] == 0.0)     # structural modes: no p
        assert np.all(Phi[:n_s, m_s:] == 0.0)     # acoustic modes: no w
        assert basis["Phi_L"] is basis["Phi_R"]   # Galerkin

    def test_reduced_matrices_have_the_cms_block_form(self, blocks):
        basis = build_cms_basis(blocks=blocks, num_structural=5,
                                num_acoustic=7)
        Kt, _, Mt = project_coupled_system(basis)

        m_s, m_a = basis["num_structural"], basis["num_acoustic"]
        A = basis["modal_coupling"]
        rho0 = blocks["rho0"]
        Kt_theory = np.block([
            [np.diag(basis["omegas_s"] ** 2), -A],
            [np.zeros((m_a, m_s)), np.diag(basis["omegas_a"] ** 2)],
        ])
        Mt_theory = np.block([
            [np.eye(m_s), np.zeros((m_s, m_a))],
            [rho0 * A.T, np.eye(m_a)],
        ])
        assert np.abs(Kt - Kt_theory).max() < 1e-8 * np.abs(Kt_theory).max()
        assert np.abs(Mt - Mt_theory).max() < 1e-8 * np.abs(Mt_theory).max()

    def test_reduced_system_stays_coupled_and_unsymmetric(self, blocks):
        basis = build_cms_basis(blocks=blocks, num_structural=5,
                                num_acoustic=7)
        Kt, _, Mt = project_coupled_system(basis)
        assert np.abs(Kt - Kt.T).max() > 0.0
        assert np.abs(Mt - Mt.T).max() > 0.0
        # Normalizing each component diagonalizes its own block only.
        assert np.abs(basis["modal_coupling"]).max() > 0.0

    def test_modal_coupling_shape(self, blocks):
        basis = build_cms_basis(blocks=blocks, num_structural=5,
                                num_acoustic=7)
        assert basis["modal_coupling"].shape == (5, basis["num_acoustic"])

    def test_rigid_pressure_mode_can_be_dropped(self, blocks):
        with_rigid = build_cms_basis(blocks=blocks, num_structural=3,
                                     num_acoustic=5)
        without = build_cms_basis(blocks=blocks, num_structural=3,
                                  num_acoustic=5, include_rigid=False)
        assert with_rigid["omegas_a"][0] / (2 * np.pi) < 1e-6
        assert without["omegas_a"][0] / (2 * np.pi) > 1.0


# ---------------------------------------------------------------------------
# Forced response
# ---------------------------------------------------------------------------

class TestModalVsDirect:

    def test_coupled_basis_converges(self, blocks, load, frequencies, direct):
        F_s, _ = load
        errors = []
        for m in (10, 30, 60):
            basis = build_coupled_modal_basis(blocks=blocks, num_modes=m)
            modal = solve_coupled_modal_frequency_sweep(
                basis, frequencies, F_s=F_s, eta_s=ETA_S,
            )
            errors.append(_global_error(modal, direct))
        assert errors[-1] < errors[0]
        assert errors[-1] < 0.05

    def test_cms_basis_converges(self, blocks, load, frequencies, direct):
        F_s, _ = load
        errors = []
        for m_a in (10, 40, 120):
            basis = build_cms_basis(blocks=blocks, num_structural=12,
                                    num_acoustic=m_a)
            modal = solve_coupled_modal_frequency_sweep(
                basis, frequencies, F_s=F_s, eta_s=ETA_S,
            )
            errors.append(_global_error(modal, direct))
        assert errors[-1] < errors[0]

    def test_coupled_basis_is_the_more_efficient_one(
        self, blocks, load, frequencies, direct
    ):
        """
        Same basis size, different quality: rigid-wall acoustic modes
        cannot follow a flexible wall, so CMS needs more vectors.
        """
        F_s, _ = load
        coupled = build_coupled_modal_basis(blocks=blocks, num_modes=30)
        cms = build_cms_basis(blocks=blocks, num_structural=10,
                              num_acoustic=20)
        assert cms["Phi_R"].shape[1] == coupled["Phi_R"].shape[1]

        err = {}
        for name, basis in (("coupled", coupled), ("cms", cms)):
            modal = solve_coupled_modal_frequency_sweep(
                basis, frequencies, F_s=F_s, eta_s=ETA_S,
            )
            err[name] = _global_error(modal, direct)
        assert err["coupled"] < err["cms"]

    def test_shapes_and_keys_match_the_direct_solver(
        self, blocks, load, frequencies, direct
    ):
        F_s, _ = load
        basis = build_coupled_modal_basis(blocks=blocks, num_modes=20)
        modal = solve_coupled_modal_frequency_sweep(
            basis, frequencies, F_s=F_s, eta_s=ETA_S, return_modal=True,
        )
        assert modal["w"].shape == direct["w"].shape
        assert modal["p"].shape == direct["p"].shape
        assert np.array_equal(modal["idx_a"], direct["idx_a"])
        assert modal["participations"].shape == (20, frequencies.size)

    def test_resonances_land_on_the_coupled_modes(self, blocks, load):
        """The peak of the response must sit on a natural frequency."""
        F_s, dof = load
        basis = build_coupled_modal_basis(blocks=blocks, num_modes=20)
        fine = np.arange(60.0, 140.0, 0.5)
        modal = solve_coupled_modal_frequency_sweep(
            basis, fine, F_s=F_s, eta_s=ETA_S,
        )
        peak = fine[np.argmax(np.abs(modal["w"][dof]))]
        flexible = basis["freqs"][basis["freqs"] > 1.0]
        assert np.min(np.abs(flexible - peak)) < 1.0

    @pytest.mark.parametrize("kind", ["coupled", "cms"])
    def test_a_load_on_the_fluid_also_works(self, blocks, frequencies, kind):
        F_a = np.zeros(blocks["n_a"])
        F_a[0] = 1.0
        basis = (build_coupled_modal_basis(blocks=blocks, num_modes=15)
                 if kind == "coupled"
                 else build_cms_basis(blocks=blocks, num_structural=8,
                                      num_acoustic=15))
        modal = solve_coupled_modal_frequency_sweep(
            basis, frequencies, F_a=F_a, eta_s=ETA_S,
        )
        assert np.abs(modal["p"]).max() > 0.0
        # The fluid pushes the plate: the structure cannot stay still.
        assert np.abs(modal["w"]).max() > 0.0


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

class TestGuards:

    def test_no_excitation_raises(self, blocks):
        basis = build_coupled_modal_basis(blocks=blocks, num_modes=6)
        with pytest.raises(ValueError, match="No excitation"):
            solve_coupled_modal_frequency_sweep(basis, [100.0])

    @pytest.mark.parametrize("builder", [build_coupled_modal_basis,
                                         build_cms_basis])
    def test_damped_blocks_are_refused(self, system, builder):
        damped = build_coupled_blocks(system, eta_s=ETA_S)
        with pytest.raises(ValueError, match="complex stiffness"):
            builder(blocks=damped)

    @pytest.mark.parametrize("builder", [build_coupled_modal_basis,
                                         build_cms_basis])
    def test_system_or_blocks_required(self, builder):
        with pytest.raises(ValueError, match="'system' or 'blocks'"):
            builder()

    def test_short_basis_warns(self, blocks, load):
        F_s, _ = load
        basis = build_coupled_modal_basis(blocks=blocks, num_modes=4)
        f_top = basis["freqs"].max()
        with pytest.warns(RuntimeWarning, match="2 x f_max"):
            solve_coupled_modal_frequency_sweep(
                basis, [f_top], F_s=F_s, eta_s=ETA_S,
            )
