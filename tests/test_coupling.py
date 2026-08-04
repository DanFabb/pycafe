"""
Tests for the vibroacoustic coupling: matrix, sign convention, solvers.

Covered:
- the coupling matrix transfers a uniform pressure as the vector area
  of the interface, loads only the translational DOFs, and orients its
  normals out of the fluid;
- ``Mc = -rho0 Kc^T``, the two faces of the same interface;
- the coupled eigenproblem degenerates into the two uncoupled ones when
  the fluid density vanishes;
- with air in a sealed cavity the first plate mode goes **up**, and by
  the amount predicted by the compact-cavity stiffness
  ``k = rho0 c0^2 (dV)^2 / V`` -- the check that fixes the sign and the
  magnitude of the coupling at once;
- the coupled frequency sweep resonates at the coupled frequency, and
  the reduced solution can be scattered back onto the mesh.
"""

import pathlib
import sys

import numpy as np
import pytest
from scipy.sparse.linalg import eigsh

from pycafe.build_matrices.coupling import (
    acoustic_coupling_matrix,
    build_coupling_matrix,
    interface_area_vector,
    interface_normals,
)
from pycafe.core.prepare_vibroacoustic_system import (
    prepare_vibroacoustic_system,
)
from pycafe.solver.solver_vibroacoustic import (
    build_coupled_blocks,
    coupled_dynamic_stiffness,
    expand_pressure,
    solve_vibroacoustic_frequency_sweep,
    solve_vibroacoustic_modal,
    structural_displacement_field,
)

LX, LY, LZ = 0.8, 0.6, 0.5
NX, NY, NZ = 8, 6, 5
V_CAVITY = LX * LY * LZ
A_PLATE = LX * LY

RHO0, C0 = 1.204, 343.0
T, RHO_S, E, NU = 0.002, 7800.0, 210e9, 0.3


@pytest.fixture(scope="module")
def mesh(tmp_path_factory):
    pytest.importorskip("gmsh")
    repo_root = pathlib.Path(__file__).parent.parent
    sys.path.insert(0, str(repo_root / "coupling_geometry"))
    from make_box_plate_mesh import make_box_plate_mesh

    msh = make_box_plate_mesh(
        Lx=LX, Ly=LY, Lz=LZ, nx=NX, ny=NY, nz=NZ,
        output_path=tmp_path_factory.mktemp("mesh") / "box_plate.msh",
    )
    from pycafe.create_geom.visualize_mesh import load_mesh_with_groups

    nodes, _, _, groups = load_mesh_with_groups(str(msh))
    return nodes, groups


def _system(mesh, **overrides):
    nodes, groups = mesh
    kwargs = dict(
        nodes=nodes, groups=groups, rho0=RHO0, c0=C0,
        t=T, rho_s=RHO_S, E=E, nu=NU,
    )
    kwargs.update(overrides)
    return prepare_vibroacoustic_system(**kwargs)


@pytest.fixture(scope="module")
def system(mesh):
    return _system(mesh)


# ------------------------------------------------------------
#  COUPLING MATRIX
# ------------------------------------------------------------
class TestCouplingMatrix:

    def test_built_by_default(self, system):
        coupling = system["coupling"]
        assert coupling is not None
        assert set(coupling) == {"Kc", "Mc", "area_vector"}

    def test_can_be_skipped(self, mesh):
        assert _system(mesh, build_coupling=False)["coupling"] is None

    def test_shape_spans_all_nodes(self, mesh, system):
        n = mesh[0].shape[0]
        assert system["coupling"]["Kc"].shape == (6 * n, n)

    def test_uniform_pressure_gives_the_vector_area(self, system):
        # The plate closes the cavity from above: the normal points out
        # of the fluid, i.e. +z, and the resultant of a unit pressure is
        # the plate area.
        area = system["coupling"]["area_vector"]
        assert np.allclose(area, [0.0, 0.0, A_PLATE], atol=1e-12)
        assert np.allclose(
            interface_area_vector(system["coupling"]["Kc"]), area
        )

    def test_only_translations_are_loaded(self, system):
        Kc = system["coupling"]["Kc"].tocsr()
        rotational = np.concatenate([
            np.arange(i, Kc.shape[0], 6) for i in (3, 4, 5)
        ])
        # An inviscid fluid applies a normal traction, never a couple.
        assert Kc[rotational].nnz == 0

    def test_normals_point_out_of_the_fluid(self, mesh, system):
        normals = system["interface"]["normals"]
        assert normals.shape == (NX * NY, 3)
        assert np.allclose(normals, [0.0, 0.0, 1.0], atol=1e-12)

    def test_forced_sign_flips_the_normals(self, mesh):
        nodes, groups = mesh
        flipped = _system(mesh, interface_sign=-1)
        assert np.allclose(
            flipped["interface"]["normals"], [0.0, 0.0, -1.0], atol=1e-12
        )
        assert np.allclose(
            flipped["coupling"]["area_vector"], [0.0, 0.0, -A_PLATE]
        )

    def test_acoustic_side_is_the_scaled_transpose(self, system):
        Kc = system["coupling"]["Kc"]
        Mc = system["coupling"]["Mc"]
        assert abs(Mc + RHO0 * Kc.T).max() == 0.0
        assert abs(acoustic_coupling_matrix(Kc, RHO0) - Mc).max() == 0.0

    def test_non_conforming_interface_is_reported(self, mesh):
        nodes, _ = mesh
        # A face whose nodes are not a face of any fluid element.
        rogue = np.array([[0, 1, 2, 3]])
        fluid = np.array([[10, 11, 12, 13, 14, 15, 16, 17]])
        with pytest.raises(ValueError, match="not conforming"):
            interface_normals(nodes, rogue, fluid)

    def test_orientation_needs_a_reference(self, mesh):
        nodes, _ = mesh
        with pytest.raises(ValueError, match="fluid_conn0"):
            interface_normals(nodes, np.array([[0, 1, 2, 3]]))

    def test_only_quad_faces_are_supported(self, mesh):
        nodes, _ = mesh
        with pytest.raises(ValueError, match="4-node"):
            build_coupling_matrix(nodes, np.array([[0, 1, 2]]), sign=1)

    def test_interior_face_has_no_outward_normal(self):
        """
        A face shared by two fluid elements is inside the fluid: the two
        owners orient it in opposite directions, so deducing a normal
        from "the" element behind it would silently pick one side.
        """
        # Two unit hexes stacked along z; their shared face is z = 1.
        coords = np.array([[x, y, z]
                           for z in (0.0, 1.0, 2.0)
                           for y in (0.0, 1.0)
                           for x in (0.0, 1.0)], dtype=float)
        bottom = [0, 1, 3, 2, 4, 5, 7, 6]
        top = [4, 5, 7, 6, 8, 9, 11, 10]
        fluid = np.array([bottom, top])
        shared = np.array([[4, 5, 7, 6]])

        with pytest.raises(ValueError, match="shared by 2 fluid elements"):
            interface_normals(coords, shared, fluid)

        # An explicit sign is the documented way out (embedded structure).
        n = interface_normals(coords, shared, fluid, sign=1)
        assert np.allclose(np.abs(n), [0.0, 0.0, 1.0])

    def test_external_face_of_the_same_stack_still_works(self):
        """The interior-face check must not reject genuine boundary faces."""
        coords = np.array([[x, y, z]
                           for z in (0.0, 1.0, 2.0)
                           for y in (0.0, 1.0)
                           for x in (0.0, 1.0)], dtype=float)
        fluid = np.array([[0, 1, 3, 2, 4, 5, 7, 6],
                          [4, 5, 7, 6, 8, 9, 11, 10]])
        top_face = np.array([[8, 9, 11, 10]])       # z = 2, one owner only
        assert np.allclose(
            interface_normals(coords, top_face, fluid), [[0.0, 0.0, 1.0]]
        )


# ------------------------------------------------------------
#  SUPPORT OF THE PLATE
# ------------------------------------------------------------
class TestSupport:

    def test_simply_supported_keeps_the_edge_rotations(self, mesh):
        clamped = _system(mesh, support="clamped")
        pinned = _system(mesh, support="simply_supported")

        n_edge = 2 * NX + 2 * NY
        # A simple support blocks the 3 translations only, so the 3
        # rotations of every edge node survive.
        assert (pinned["structural"]["K_red"].shape[0]
                - clamped["structural"]["K_red"].shape[0]) == 3 * n_edge
        assert list(pinned["structural"]["blocked_dofs"]) == [0, 1, 2]

    def test_simply_supported_matches_the_navier_solution(self, mesh):
        """
        S-S-S-S rectangular plate, Navier:
        f_11 = pi/2 sqrt(D / (rho t)) (1/a^2 + 1/b^2).
        """
        pinned = _system(mesh, support="simply_supported")
        f_fem, _ = _in_vacuo_plate(pinned, k=1)

        D = E * T ** 3 / (12 * (1 - NU ** 2))
        f_ref = (np.pi / 2) * np.sqrt(D / (RHO_S * T)) * (
            1 / LX ** 2 + 1 / LY ** 2
        )
        assert abs(f_fem[0] - f_ref) / f_ref < 0.03

    def test_softer_support_gives_lower_frequencies(self, mesh):
        f_pinned, _ = _in_vacuo_plate(
            _system(mesh, support="simply_supported"), k=1
        )
        f_clamped, _ = _in_vacuo_plate(_system(mesh, support="clamped"), k=1)
        assert f_pinned[0] < f_clamped[0]

    def test_explicit_dofs_override_the_named_support(self, mesh):
        # Only the transverse translation blocked: softer than S-S-S-S.
        custom = _system(mesh, clamp_dofs=[2])
        assert list(custom["structural"]["blocked_dofs"]) == [2]
        assert custom["structural"]["support"] == "custom"

    def test_unknown_support_is_reported(self, mesh):
        with pytest.raises(ValueError, match="Unknown support"):
            _system(mesh, support="welded")
        with pytest.raises(ValueError, match="local DOF indices"):
            _system(mesh, clamp_dofs=[7])

    def test_clamped_plate_does_not_float(self, mesh):
        """C-C-C-C: edge fully blocked, so no rigid-body motion at all."""
        clamped = _system(mesh, support="clamped")
        f_plate, _ = _in_vacuo_plate(clamped, k=6)
        assert np.all(f_plate > 1.0)

        # Every edge node is blocked in all 6 DOF: none of its DOFs
        # survives among the unknowns.
        edge = clamped["structural"]["clamped_nodes0"]
        idx_free = clamped["structural"]["idx_free"]
        assert not np.intersect1d(idx_free // 6, edge).size

    def test_simply_supported_plate_does_not_float_either(self, mesh):
        """S-S-S-S: softer than a clamp, still no rigid-body motion."""
        pinned = _system(mesh, support="simply_supported")
        f_plate, _ = _in_vacuo_plate(pinned, k=6)
        assert np.all(f_plate > 1.0)

    def test_unconstrained_plate_floats(self, mesh):
        """Only `support="free"` leaves the plate floating in space."""
        free = _system(mesh, support="free")
        assert free["structural"]["K_red"].shape[0] == 6 * (NX + 1) * (NY + 1)

        K = free["structural"]["K_red"].tocsc()
        M = free["structural"]["M_red"].tocsc()
        vals, _ = eigsh(K, k=7, M=M, sigma=-1e3, which="LM")
        freqs = np.sqrt(np.abs(np.sort(vals))) / (2 * np.pi)

        # 5 of the 6 rigid-body motions come out at zero: the two
        # in-plane translations, the transverse one and the two
        # rotations about x and y. The in-plane rotation about z is
        # held by the drilling stiffness of CQUAD4F (Nastran K6ROT),
        # which is an artificial stiffness added to keep the drilling
        # DOF from being singular -- so it is not in the null space.
        assert np.sum(freqs < 1e-3) == 5
        assert freqs[6] > freqs[5]


# ------------------------------------------------------------
#  COUPLED EIGENPROBLEM
# ------------------------------------------------------------
def _in_vacuo_plate(system, k=4):
    K = system["structural"]["K_red"].tocsc()
    M = system["structural"]["M_red"].tocsc()
    vals, vecs = eigsh(K, k=k, M=M, sigma=0.0, which="LM")
    order = np.argsort(vals)
    return np.sqrt(np.abs(vals[order])) / (2 * np.pi), vecs[:, order]


class TestCoupledModes:

    def test_vanishing_density_recovers_the_uncoupled_problems(self, mesh):
        light = _system(mesh, rho0=1e-9)
        freqs, _ = solve_vibroacoustic_modal(light, num_modes=5)
        f_plate, _ = _in_vacuo_plate(light)

        # First: the uniform-pressure mode of the closed cavity, the
        # acoustic rigid-body mode.
        assert freqs[0] < 1e-3
        assert np.allclose(freqs[1:5], f_plate[:4], rtol=1e-4)

    def test_sealed_cavity_stiffens_the_first_mode(self, mesh, system):
        freqs, _ = solve_vibroacoustic_modal(system, num_modes=3)
        f_plate, _ = _in_vacuo_plate(system, k=1)
        # The trapped air is a spring: the frequency must go up. A sign
        # error in the coupling shows up here as a drop.
        assert freqs[1] > f_plate[0]

    def test_stiffening_matches_the_compact_cavity_prediction(self, system):
        """
        Below the first cavity resonance the fluid acts on the plate mode
        as the spring ``k = rho0 c0^2 dV^2 / V``, with ``dV`` the volume
        swept by the mode. That fixes the magnitude, not just the sign.
        """
        blocks = build_coupled_blocks(system)
        Ks, Ms, Kc = blocks["Ks"], blocks["Ms"], blocks["Kc"]

        vals, vecs = eigsh(Ks.tocsc(), k=1, M=Ms.tocsc(), sigma=0.0,
                           which="LM")
        phi, omega_p2 = vecs[:, 0], vals[0]
        modal_mass = phi @ (Ms @ phi)
        swept_volume = (Kc @ np.ones(blocks["n_a"])) @ phi

        k_air = RHO0 * C0 ** 2 * swept_volume ** 2 / V_CAVITY
        f_lumped = np.sqrt(omega_p2 + k_air / modal_mass) / (2 * np.pi)

        freqs, _ = solve_vibroacoustic_modal(system, num_modes=3)
        assert abs(freqs[1] - f_lumped) / f_lumped < 0.02

    def test_zero_mode_is_the_cavity_pressure_not_a_floating_plate(
        self, system
    ):
        """
        The ~0 Hz mode of the coupled clamped case is acoustic, not
        structural: the pressure is uniform over the sealed cavity and
        the plate merely sits at its static deflection under it. The
        plate itself has no rigid-body motion -- it is clamped.
        """
        from scipy.sparse.linalg import spsolve

        blocks = build_coupled_blocks(system)
        freqs, modes = solve_vibroacoustic_modal(system, num_modes=2,
                                                 blocks=blocks)
        assert freqs[0] < 1e-3

        n_s = blocks["n_s"]
        w0 = np.real(modes[:n_s, 0])
        p0 = np.real(modes[n_s:, 0])

        # Uniform pressure over the whole cavity.
        assert np.ptp(p0 / p0.mean()) < 1e-9

        # The plate shape in that mode is the static deflection under
        # that same uniform pressure, not a rigid displacement.
        w_static = spsolve(
            blocks["Ks"].tocsc(),
            blocks["Kc"] @ np.full(blocks["n_a"], p0.mean()),
        )
        assert abs(np.corrcoef(w0, w_static)[0, 1]) > 0.999

    def test_modes_have_both_fields(self, system):
        blocks = build_coupled_blocks(system)
        freqs, modes = solve_vibroacoustic_modal(system, num_modes=3,
                                                 blocks=blocks)
        assert modes.shape == (blocks["n_s"] + blocks["n_a"], 3)
        # The first flexible coupled mode moves the plate and the fluid.
        w, p = modes[:blocks["n_s"], 1], modes[blocks["n_s"]:, 1]
        assert np.abs(w).max() > 0.0
        assert np.abs(p).max() > 0.0


# ------------------------------------------------------------
#  COUPLED FREQUENCY RESPONSE
# ------------------------------------------------------------
def _centre_plate_dof(system, nodes):
    """Index, in the reduced structural unknowns, of uz at the centre."""
    idx_s = system["structural"]["idx_free"]
    node_of_dof, component = idx_s // 6, idx_s % 6
    plate = np.unique(node_of_dof)
    centre = np.array([LX / 2, LY / 2, LZ])
    target = plate[np.argmin(np.linalg.norm(nodes[plate] - centre, axis=1))]
    return int(np.where((node_of_dof == target) & (component == 2))[0][0])


class TestCoupledSweep:

    def test_response_peaks_at_the_coupled_frequency(self, mesh, system):
        nodes, _ = mesh
        blocks = build_coupled_blocks(system)
        dof = _centre_plate_dof(system, nodes)

        freqs, _ = solve_vibroacoustic_modal(system, num_modes=3,
                                             blocks=blocks)
        f_res = freqs[1]

        F_s = np.zeros(blocks["n_s"])
        F_s[dof] = 1.0
        sweep = np.array([0.5 * f_res, f_res, 1.5 * f_res])
        out = solve_vibroacoustic_frequency_sweep(
            system, sweep, F_s=F_s, blocks=blocks, verbose=False,
        )

        amplitude = np.abs(out["w"][dof])
        assert amplitude[1] > 50 * amplitude[0]
        assert amplitude[1] > 50 * amplitude[2]
        # A vibrating plate pressurises the cavity: no silent response.
        assert np.abs(out["p"][:, 1]).max() > 0.0

    def test_shapes_and_index_sets(self, system):
        blocks = build_coupled_blocks(system)
        out = solve_vibroacoustic_frequency_sweep(
            system, [100.0, 200.0], blocks=blocks, verbose=False,
        )
        assert out["w"].shape == (blocks["n_s"], 2)
        assert out["p"].shape == (blocks["n_a"], 2)
        assert out["idx_s"].size == blocks["n_s"]
        assert out["idx_a"].size == blocks["n_a"]

    def test_pinned_pressure_is_removed_from_the_unknowns(self, system):
        full = build_coupled_blocks(system)
        pinned = build_coupled_blocks(system, pressure_zero_nodes0=[0, 1, 2])
        assert pinned["n_a"] == full["n_a"] - 3
        assert pinned["Kc"].shape == (full["n_s"], full["n_a"] - 3)

    def test_dynamic_stiffness_blocks_are_unsymmetric(self, system):
        blocks = build_coupled_blocks(system)
        A = coupled_dynamic_stiffness(blocks, 2 * np.pi * 100.0)
        n_s = blocks["n_s"]
        assert A.shape == (n_s + blocks["n_a"],) * 2
        # Pressure -> force sits in the stiffness block, structural
        # acceleration -> fluid in the mass block: the two off-diagonal
        # blocks cannot be each other's transpose.
        upper = A[:n_s, n_s:]
        lower = A[n_s:, :n_s]
        assert abs(upper - lower.T).max() > 0.0

    def test_wrong_load_size_is_reported(self, system):
        blocks = build_coupled_blocks(system)
        with pytest.raises(ValueError, match="F_s has size"):
            solve_vibroacoustic_frequency_sweep(
                system, [100.0], F_s=np.zeros(3), blocks=blocks, verbose=False,
            )

    def test_missing_coupling_is_reported(self, mesh):
        uncoupled = _system(mesh, build_coupling=False)
        with pytest.raises(RuntimeError, match="system\\['coupling'\\]"):
            build_coupled_blocks(uncoupled)


# ------------------------------------------------------------
#  BACK TO THE MESH
# ------------------------------------------------------------
def test_solution_scatters_back_onto_the_mesh(mesh, system):
    nodes, _ = mesh
    n = nodes.shape[0]
    blocks = build_coupled_blocks(system, pressure_zero_nodes0=[0])
    out = solve_vibroacoustic_frequency_sweep(
        system, [120.0], F_s=np.ones(blocks["n_s"]), blocks=blocks,
        verbose=False,
    )

    p_full = expand_pressure(out["p"], out["idx_a"], n)
    assert p_full.shape == (n, 1)
    assert p_full[0, 0] == 0.0                      # pinned node
    assert np.allclose(p_full[out["idx_a"], 0], out["p"][:, 0])

    w_full = structural_displacement_field(out["w"][:, 0], out["idx_s"], n)
    assert w_full.shape == (n, 6)
    clamped = system["structural"]["clamped_nodes0"]
    assert np.allclose(w_full[clamped], 0.0)
