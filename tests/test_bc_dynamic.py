"""
Validation tests for the boundary conditions of the dynamic response.

Covered:
- boundary integrals ``int N dOmega`` and ``int N_a N_b dOmega`` on
  edges (2D fluid) and faces (3D fluid), including skewed faces;
- impedance specification: normalized, absolute, frequency dependent;
- normal velocity load vector: sign convention, linearity, frequency
  scaling, and assembly on 3D faces;
- impedance boundary conditions given by physical-group *name*
  through the public ``prepare_acoustic_system`` entry point;
- plane wave in a duct with a vibrating piston and an anechoic
  termination, in 2D and in 3D, against the analytical solution.

The duct case exercises the Neumann and the Robin condition at once and
is the reference check of the sign convention: with ``exp(+j omega t)``
and the normal velocity taken along the **outward** normal, a piston at
x = 0 moving toward +x with velocity u imposes ``v_n = -u`` and radiates
``p(x) = rho c u exp(-j k x)`` into an anechoic duct.
"""

import numpy as np
import pytest

from pycafe.boundary_condition.acoustic_bc import (
    AcousticBC,
    build_impedance_operator,
    build_velocity_operator,
    make_admittance,
)
from pycafe.build_matrices.assembly_dispatcher import build_KM_acoustic
from pycafe.build_matrices.bc_surface import (
    boundary_integrals,
    resolve_boundary_faces,
)
from pycafe.core.prepare_acoustic_system import prepare_acoustic_system
from pycafe.solver.solver_helmholtz_1 import (
    build_normal_velocity_rhs,
    solve_helmholtz_frequency_sweep,
    solve_helmholtz_single_frequency,
)

RHO = 1.2
C0 = 340.0


# Mesh helpers (structured, built in place: no Gmsh needed)

def duct_mesh_2d(L=1.0, H=0.1, nx=40, ny=2):
    """CQUAD4 duct along x, with Line 2 elements on the two ends."""
    xs = np.linspace(0.0, L, nx + 1)
    ys = np.linspace(0.0, H, ny + 1)
    nodes = np.array([[x, y, 0.0] for y in ys for x in xs])

    def tag(i, j):
        return j * (nx + 1) + i + 1

    quads = np.array([
        [tag(i, j), tag(i + 1, j), tag(i + 1, j + 1), tag(i, j + 1)]
        for j in range(ny) for i in range(nx)
    ])
    inlet = np.array([[tag(0, j), tag(0, j + 1)] for j in range(ny)])
    outlet = np.array([[tag(nx, j), tag(nx, j + 1)] for j in range(ny)])

    elements = {
        "Quadrilateral 4": quads,
        "Line 2": np.vstack([inlet, outlet]),
    }
    boundaries = {
        "inlet": sorted(set(inlet.ravel().tolist())),
        "outlet": sorted(set(outlet.ravel().tolist())),
    }
    return nodes, elements, boundaries


def duct_mesh_3d(L=1.0, H=0.1, W=0.1, nx=30, ny=1, nz=1):
    """CHEXA8 duct along x, with Quadrilateral 4 faces on the two ends."""
    xs = np.linspace(0.0, L, nx + 1)
    ys = np.linspace(0.0, H, ny + 1)
    zs = np.linspace(0.0, W, nz + 1)
    nodes = np.array([[x, y, z] for z in zs for y in ys for x in xs])

    nxp, nyp = nx + 1, ny + 1

    def tag(i, j, k):
        return k * nxp * nyp + j * nxp + i + 1

    hexes = np.array([
        [tag(i, j, k), tag(i + 1, j, k), tag(i + 1, j + 1, k), tag(i, j + 1, k),
         tag(i, j, k + 1), tag(i + 1, j, k + 1),
         tag(i + 1, j + 1, k + 1), tag(i, j + 1, k + 1)]
        for k in range(nz) for j in range(ny) for i in range(nx)
    ])

    def face_x(i):
        return np.array([
            [tag(i, j, k), tag(i, j + 1, k),
             tag(i, j + 1, k + 1), tag(i, j, k + 1)]
            for k in range(nz) for j in range(ny)
        ])

    inlet, outlet = face_x(0), face_x(nx)
    elements = {
        "Hexahedron 8": hexes,
        "Quadrilateral 4": np.vstack([inlet, outlet]),
    }
    boundaries = {
        "inlet": sorted(set(inlet.ravel().tolist())),
        "outlet": sorted(set(outlet.ravel().tolist())),
    }
    return nodes, elements, boundaries


# Tests: boundary integrals

class TestBoundaryIntegrals:

    def test_line2_length(self):
        """The load vector of an edge sums to its length."""
        nodes = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
        g, S = boundary_integrals(nodes, {"Line 2": np.array([[1, 2]])}, 2)
        assert np.isclose(g.sum(), 5.0)
        assert np.isclose(S.sum(), 5.0)

    def test_line3_length(self):
        """A quadratic edge integrates exactly as well."""
        nodes = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        g, _ = boundary_integrals(nodes, {"Line 3": np.array([[1, 2, 3]])}, 3)
        assert np.isclose(g.sum(), 2.0)

    def test_quad4_face_area(self):
        """The load vector of a face sums to its area."""
        nodes = np.array([
            [0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 3.0, 0.0], [0.0, 3.0, 0.0],
        ])
        g, S = boundary_integrals(
            nodes, {"Quadrilateral 4": np.array([[1, 2, 3, 4]])}, 4
        )
        assert np.isclose(g.sum(), 6.0)
        assert np.isclose(S.sum(), 6.0)
        # Bilinear face, uniform: the area splits evenly over the corners.
        np.testing.assert_allclose(g, np.full(4, 1.5))

    def test_face_area_is_orientation_independent(self):
        """A face tilted in 3D keeps its true area (no projection)."""
        # Unit square rotated 45 degrees about the y axis.
        s = np.sqrt(0.5)
        nodes = np.array([
            [0.0, 0.0, 0.0], [s, 0.0, s], [s, 1.0, s], [0.0, 1.0, 0.0],
        ])
        g, _ = boundary_integrals(
            nodes, {"Quadrilateral 4": np.array([[1, 2, 3, 4]])}, 4
        )
        assert np.isclose(g.sum(), 1.0)

    def test_triangle_area(self):
        nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        g, _ = boundary_integrals(nodes, {"Triangle 3": np.array([[1, 2, 3]])}, 3)
        assert np.isclose(g.sum(), 0.5)

    def test_mass_matrix_is_symmetric(self):
        nodes, elements, boundaries = duct_mesh_3d(nx=3)
        faces = resolve_boundary_faces(
            ["outlet"], boundaries=boundaries, elements=elements, boundary_dim=2
        )
        _, S = boundary_integrals(nodes, faces, nodes.shape[0])
        S = S.toarray()
        np.testing.assert_allclose(S, S.T, atol=1e-14)

    def test_mass_matrix_row_sums_equal_load_vector(self):
        """Partition of unity: sum_b S_ab = int N_a dOmega = g_a."""
        nodes, elements, boundaries = duct_mesh_3d(nx=3, ny=2, nz=2)
        faces = resolve_boundary_faces(
            ["inlet"], boundaries=boundaries, elements=elements, boundary_dim=2
        )
        g, S = boundary_integrals(nodes, faces, nodes.shape[0])
        np.testing.assert_allclose(np.asarray(S.sum(axis=1)).ravel(), g, atol=1e-14)


# Tests: boundary resolution

class TestBoundaryResolution:

    def test_3d_boundary_resolves_to_faces_not_edges(self):
        """On a HEXA8 mesh an impedance wall must be a set of faces."""
        nodes, elements, boundaries = duct_mesh_3d(nx=4, ny=2, nz=2)
        faces = resolve_boundary_faces(
            ["outlet"], nodes=nodes, boundaries=boundaries,
            elements=elements, boundary_dim=2,
        )
        assert set(faces) == {"Quadrilateral 4"}
        assert faces["Quadrilateral 4"].shape == (4, 4)   # 2 x 2 face patch

    def test_2d_boundary_resolves_to_edges(self):
        nodes, elements, boundaries = duct_mesh_2d(nx=5, ny=2)
        faces = resolve_boundary_faces(
            ["inlet"], nodes=nodes, boundaries=boundaries,
            elements=elements, boundary_dim=1,
        )
        assert set(faces) == {"Line 2"}
        assert faces["Line 2"].shape == (2, 2)

    def test_unknown_boundary_name_raises(self):
        nodes, elements, boundaries = duct_mesh_2d(nx=3, ny=1)
        with pytest.raises(ValueError, match="not found"):
            resolve_boundary_faces(
                ["nope"], boundaries=boundaries, elements=elements
            )

    def test_polyline_fallback_without_elements(self):
        """A bare node list on a 2D model still yields segments."""
        nodes = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0]])
        faces = resolve_boundary_faces([1, 2, 3], nodes=nodes, boundary_dim=1)
        g, _ = boundary_integrals(nodes, faces, 3)
        assert np.isclose(g.sum(), 2.0)


# Tests: impedance specification

class TestAdmittance:

    def test_bare_value_is_normalized(self):
        adm = make_admittance(2.0, RHO, C0)
        assert np.isclose(adm(0.0), 1.0 / (2.0 * RHO * C0))

    def test_absolute_unit(self):
        adm = make_admittance((1200.0, "abs"), RHO, C0)
        assert np.isclose(adm(0.0), 1.0 / 1200.0)

    def test_absolute_matches_normalized(self):
        zeta = 3.0
        a_norm = make_admittance(zeta, RHO, C0)
        a_abs = make_admittance((zeta * RHO * C0, "rayl"), RHO, C0)
        assert np.isclose(a_norm(0.0), a_abs(0.0))

    def test_callable_impedance(self):
        adm = make_admittance(lambda w: 1.0 + 1j * w, RHO, C0)
        assert not np.isclose(adm(10.0), adm(1000.0))

    def test_rigid_wall_gives_no_operator(self):
        assert make_admittance(0.0, RHO, C0) is None
        assert make_admittance(None, RHO, C0) is None

    def test_zero_impedance_from_callable_raises(self):
        adm = make_admittance(lambda w: 0.0, RHO, C0)
        with pytest.raises(ValueError, match="infinite admittance"):
            adm(1.0)

    def test_unknown_unit_raises(self):
        with pytest.raises(ValueError, match="Unknown impedance unit"):
            make_admittance((1.0, "furlongs"), RHO, C0)


class TestImpedanceOperator:

    def test_named_boundary_on_3d_faces(self):
        nodes, elements, boundaries = duct_mesh_3d(nx=4, ny=2, nz=2)
        bc = AcousticBC().add_impedance("outlet", 1.0)
        op = build_impedance_operator(
            bc, nodes=nodes, rho=RHO, c0=C0,
            boundaries=boundaries, elements=elements, boundary_dim=2,
        )
        C = op.at(0.0)
        # C = rho * A * S, and S sums to the face area (0.1 x 0.1).
        expected = RHO * (1.0 / (RHO * C0)) * 0.01
        assert np.isclose(C.sum(), expected)

    def test_rigid_wall_operator_is_empty(self):
        nodes, elements, boundaries = duct_mesh_2d(nx=3, ny=1)
        bc = AcousticBC().add_impedance("outlet", 0.0)
        op = build_impedance_operator(
            bc, nodes=nodes, rho=RHO, c0=C0,
            boundaries=boundaries, elements=elements,
        )
        assert op.is_empty
        assert np.allclose(op.at(100.0).toarray(), 0.0)

    def test_frequency_dependent_impedance(self):
        nodes, elements, boundaries = duct_mesh_2d(nx=3, ny=1)
        bc = AcousticBC().add_impedance("outlet", lambda w: 1.0 + 1j * w / 1e3)
        op = build_impedance_operator(
            bc, nodes=nodes, rho=RHO, c0=C0,
            boundaries=boundaries, elements=elements,
        )
        assert op.is_frequency_dependent
        assert not np.isclose(op.at(50.0).sum(), op.at(5000.0).sum())

    def test_constant_impedance_not_flagged_frequency_dependent(self):
        nodes, elements, boundaries = duct_mesh_2d(nx=3, ny=1)
        bc = AcousticBC().add_impedance("outlet", 2.0 - 1j)
        op = build_impedance_operator(
            bc, nodes=nodes, rho=RHO, c0=C0,
            boundaries=boundaries, elements=elements,
        )
        assert not op.is_frequency_dependent

    def test_distinct_impedances_on_distinct_walls(self):
        """The per-boundary values the legacy tuple could not express."""
        nodes, elements, boundaries = duct_mesh_2d(nx=3, ny=1)
        bc = (AcousticBC()
              .add_impedance("inlet", 1.0)
              .add_impedance("outlet", 4.0))
        op = build_impedance_operator(
            bc, nodes=nodes, rho=RHO, c0=C0,
            boundaries=boundaries, elements=elements,
        )
        assert len(op.terms) == 2
        # Same edge length, admittances in ratio 4:1.
        assert np.isclose(op.terms[0][1](0.0), 4.0 * op.terms[1][1](0.0))

    def test_reduce_matches_submatrix(self):
        nodes, elements, boundaries = duct_mesh_2d(nx=4, ny=2)
        bc = AcousticBC().add_impedance("outlet", 1.5)
        op = build_impedance_operator(
            bc, nodes=nodes, rho=RHO, c0=C0,
            boundaries=boundaries, elements=elements,
        )
        idx_free = np.arange(3, nodes.shape[0])
        C_full = op.at(0.0).toarray()
        C_red = op.reduce(idx_free).at(0.0).toarray()
        np.testing.assert_allclose(C_red, C_full[np.ix_(idx_free, idx_free)])


# Tests: normal velocity load vector

class TestVelocityOperator:

    def test_sign_and_magnitude(self):
        """V_n = -j rho omega vbar_n int N dOmega, outward normal."""
        nodes, elements, boundaries = duct_mesh_2d(nx=3, ny=1)
        v_n = 0.02
        omega = 2 * np.pi * 100.0

        op = build_velocity_operator(
            AcousticBC().add_velocity("inlet", v_n),
            nodes=nodes, rho=RHO, boundaries=boundaries, elements=elements,
        )
        f = op.at(omega)

        g = op.terms[0][0]
        np.testing.assert_allclose(f, -1j * RHO * omega * v_n * g)
        # The inlet edge is 0.1 m long.
        assert np.isclose(g.sum(), 0.1)

    def test_linear_in_velocity(self):
        nodes, elements, boundaries = duct_mesh_2d(nx=3, ny=1)
        omega = 2 * np.pi * 100.0
        f1 = build_velocity_operator(
            AcousticBC().add_velocity("inlet", 0.01),
            nodes=nodes, rho=RHO, boundaries=boundaries, elements=elements,
        ).at(omega)
        f2 = build_velocity_operator(
            AcousticBC().add_velocity("inlet", 0.02),
            nodes=nodes, rho=RHO, boundaries=boundaries, elements=elements,
        ).at(omega)
        np.testing.assert_allclose(f2, 2.0 * f1, rtol=1e-12)

    def test_linear_in_frequency(self):
        nodes, elements, boundaries = duct_mesh_2d(nx=3, ny=1)
        op = build_velocity_operator(
            AcousticBC().add_velocity("inlet", 0.01),
            nodes=nodes, rho=RHO, boundaries=boundaries, elements=elements,
        )
        np.testing.assert_allclose(op.at(200.0), 2.0 * op.at(100.0), rtol=1e-12)

    def test_frequency_dependent_velocity(self):
        nodes, elements, boundaries = duct_mesh_2d(nx=3, ny=1)
        op = build_velocity_operator(
            AcousticBC().add_velocity("inlet", lambda w: 1.0 / max(w, 1.0)),
            nodes=nodes, rho=RHO, boundaries=boundaries, elements=elements,
        )
        # v_n ~ 1/omega cancels the omega factor: constant load vector.
        np.testing.assert_allclose(op.at(100.0), op.at(500.0), rtol=1e-12)

    def test_3d_face_area(self):
        nodes, elements, boundaries = duct_mesh_3d(nx=4, ny=2, nz=2)
        op = build_velocity_operator(
            AcousticBC().add_velocity("inlet", 1.0),
            nodes=nodes, rho=RHO, boundaries=boundaries,
            elements=elements, boundary_dim=2,
        )
        assert np.isclose(op.terms[0][0].sum(), 0.1 * 0.1)

    def test_rigid_wall_is_empty(self):
        nodes, elements, boundaries = duct_mesh_2d(nx=3, ny=1)
        op = build_velocity_operator(
            AcousticBC().add_velocity("inlet", 0.0),
            nodes=nodes, rho=RHO, boundaries=boundaries, elements=elements,
        )
        assert op.is_empty


class TestVelocityRHSCompatibility:
    """The legacy ``build_normal_velocity_rhs`` entry point."""

    def test_matches_operator(self):
        nodes, elements, boundaries = duct_mesh_2d(nx=3, ny=1)
        idx_free = np.arange(nodes.shape[0])
        omega = 2 * np.pi * 250.0

        idx, vals = build_normal_velocity_rhs(
            nodes=nodes,
            boundary_velocity_nodes=["inlet"],
            idx_free=idx_free,
            rho=RHO,
            omega=omega,
            v_n=0.03,
            boundaries=boundaries,
            elements=elements,
        )
        f = build_velocity_operator(
            AcousticBC().add_velocity("inlet", 0.03),
            nodes=nodes, rho=RHO, boundaries=boundaries, elements=elements,
        ).at(omega)

        expected = np.zeros(nodes.shape[0], dtype=complex)
        expected[idx] = vals
        np.testing.assert_allclose(expected, f, atol=1e-18)

    def test_only_free_dofs_are_returned(self):
        nodes, elements, boundaries = duct_mesh_2d(nx=3, ny=1)
        idx_free = np.array([0, 1, 2])       # excludes most inlet nodes
        idx, vals = build_normal_velocity_rhs(
            nodes=nodes,
            boundary_velocity_nodes=["inlet"],
            idx_free=idx_free,
            rho=RHO,
            omega=100.0,
            v_n=1.0,
            boundaries=boundaries,
            elements=elements,
        )
        assert idx.size == 0 or idx.max() < idx_free.size


# Tests: public entry point with named boundaries

class TestPrepareAcousticSystemBoundaries:

    def test_impedance_by_name_does_not_crash(self):
        """
        Regression: ``choose_boundary_conditions`` returns boundary
        *names*, which used to reach the impedance assembly as strings
        and blow up in ``int(...)``.
        """
        nodes, elements, boundaries = duct_mesh_2d(nx=4, ny=2)
        bc = ([], [], 0.0, ["outlet"], 1.0 + 0j, [], 0.0, None, None)

        system = prepare_acoustic_system(
            nodes=nodes, elements=elements, boundaries=boundaries,
            rho=RHO, c0=C0, bc=bc,
        )
        assert system["C"].shape == (nodes.shape[0], nodes.shape[0])
        assert np.abs(system["C"]).sum() > 0.0

    def test_impedance_by_name_3d_uses_faces(self):
        nodes, elements, boundaries = duct_mesh_3d(nx=4, ny=2, nz=2)
        bc = AcousticBC().add_impedance("outlet", 1.0)

        system = prepare_acoustic_system(
            nodes=nodes, elements=elements, boundaries=boundaries,
            rho=RHO, c0=C0, bc=bc,
        )
        expected = RHO * (1.0 / (RHO * C0)) * 0.01     # rho * A * area
        assert np.isclose(system["C"].sum(), expected)
        assert system["boundary_dim"] == 2

    def test_velocity_operator_is_exposed(self):
        nodes, elements, boundaries = duct_mesh_2d(nx=4, ny=2)
        bc = AcousticBC().add_velocity("inlet", -0.01)

        system = prepare_acoustic_system(
            nodes=nodes, elements=elements, boundaries=boundaries,
            rho=RHO, c0=C0, bc=bc,
        )
        assert not system["velocity_op"].is_empty
        assert system["velocity_red_op"].shape == (system["idx_free"].size,)


# Tests: analytical validation on a duct

def _solve_duct(nodes, elements, boundaries, freq, u):
    """Piston at x = 0, anechoic termination at x = L."""
    K, M, _, _ = build_KM_acoustic(nodes, elements, C0)
    omega = 2 * np.pi * freq

    # Outward normal at the inlet points toward -x, so a piston moving
    # toward +x with velocity u imposes v_n = -u.
    bc = (AcousticBC()
          .add_velocity("inlet", -u)
          .add_impedance("outlet", 1.0))

    C = build_impedance_operator(
        bc, nodes=nodes, rho=RHO, c0=C0,
        boundaries=boundaries, elements=elements,
    ).at(omega)
    f = build_velocity_operator(
        bc, nodes=nodes, rho=RHO, boundaries=boundaries, elements=elements,
    ).at(omega)

    return solve_helmholtz_single_frequency(
        K.tocsr(), M.tocsr(), C, omega,
        pressure_nodes_red=np.array([], dtype=int),
        pressure_values=np.array([], dtype=complex),
        f_red=f,
    )


class TestDuctPlaneWave:

    @pytest.mark.parametrize("freq", [120.0, 200.0])
    def test_2d_duct_matches_analytical(self, freq):
        u = 0.01
        nodes, elements, boundaries = duct_mesh_2d(nx=60, ny=2)
        p = _solve_duct(nodes, elements, boundaries, freq, u)

        k = 2 * np.pi * freq / C0
        p_exact = RHO * C0 * u * np.exp(-1j * k * nodes[:, 0])

        err = np.abs(p - p_exact).max() / np.abs(p_exact).max()
        assert err < 5e-3, f"relative error {err:.2e}"

    def test_3d_duct_matches_analytical(self):
        u, freq = 0.01, 200.0
        nodes, elements, boundaries = duct_mesh_3d(nx=40)
        p = _solve_duct(nodes, elements, boundaries, freq, u)

        k = 2 * np.pi * freq / C0
        p_exact = RHO * C0 * u * np.exp(-1j * k * nodes[:, 0])

        err = np.abs(p - p_exact).max() / np.abs(p_exact).max()
        assert err < 5e-3, f"relative error {err:.2e}"

    def test_converges_with_refinement(self):
        u, freq = 0.01, 200.0
        k = 2 * np.pi * freq / C0

        errors = []
        for nx in (20, 40, 80):
            nodes, elements, boundaries = duct_mesh_2d(nx=nx, ny=1)
            p = _solve_duct(nodes, elements, boundaries, freq, u)
            p_exact = RHO * C0 * u * np.exp(-1j * k * nodes[:, 0])
            errors.append(np.abs(p - p_exact).max() / np.abs(p_exact).max())

        assert errors[1] < errors[0] and errors[2] < errors[1]

    def test_pressure_is_flat_across_the_section(self):
        """Below the first cut-on frequency only the plane mode travels."""
        nodes, elements, boundaries = duct_mesh_2d(nx=40, ny=4)
        p = _solve_duct(nodes, elements, boundaries, 200.0, 0.01)

        for x0 in (0.0, 0.5, 1.0):
            column = np.isclose(nodes[:, 0], x0)
            spread = np.abs(p[column] - p[column].mean()).max()
            assert spread < 1e-6 * np.abs(p).max()


# Tests: frequency sweep wiring

class TestFrequencySweep:

    def test_sweep_matches_single_frequency(self):
        nodes, elements, boundaries = duct_mesh_2d(nx=20, ny=2)
        freq = 200.0
        bc = (AcousticBC()
              .add_velocity("inlet", -0.01)
              .add_impedance("outlet", 1.0))

        system = prepare_acoustic_system(
            nodes=nodes, elements=elements, boundaries=boundaries,
            rho=RHO, c0=C0, bc=bc,
        )

        P = solve_helmholtz_frequency_sweep(
            system["K_red"].tocsr(),
            system["M_red"].tocsr(),
            system["C_red"],
            [freq],
            system["pressure_nodes_red"],
            system["pressure_values"],
            velocity_operator=system["velocity_red_op"],
            impedance_operator=system["C_red_op"],
        )

        p_ref = _solve_duct(nodes, elements, boundaries, freq, 0.01)
        np.testing.assert_allclose(P[:, 0], p_ref[system["idx_free"]], rtol=1e-9)

    def test_frequency_dependent_impedance_changes_the_answer(self):
        nodes, elements, boundaries = duct_mesh_2d(nx=20, ny=2)
        frequencies = [100.0, 400.0]

        def liner(omega):
            return 1.0 + 1j * omega / 5e3

        bc = (AcousticBC()
              .add_velocity("inlet", -0.01)
              .add_impedance("outlet", liner))

        system = prepare_acoustic_system(
            nodes=nodes, elements=elements, boundaries=boundaries,
            rho=RHO, c0=C0, bc=bc,
        )
        assert system["C_red_op"].is_frequency_dependent

        P_var = solve_helmholtz_frequency_sweep(
            system["K_red"].tocsr(), system["M_red"].tocsr(), system["C_red"],
            frequencies,
            system["pressure_nodes_red"], system["pressure_values"],
            velocity_operator=system["velocity_red_op"],
            impedance_operator=system["C_red_op"],
        )
        # Same model with the impedance frozen at its omega = 0 value.
        P_fixed = solve_helmholtz_frequency_sweep(
            system["K_red"].tocsr(), system["M_red"].tocsr(),
            system["C_red_op"].at(0.0),
            frequencies,
            system["pressure_nodes_red"], system["pressure_values"],
            velocity_operator=system["velocity_red_op"],
        )
        assert not np.allclose(P_var, P_fixed)
