"""
Spherical wave radiation on a spherical boundary.

Three questions, in order of how much they prove:

* does the boundary integrate what it claims (area, sphere fit, the two
  halves of ``jk + 1/r``)?
* with an incident field and nothing to scatter it, does the solution
  come back as the incident plane wave? That is an exact answer the
  discrete problem should reproduce to discretization error alone, and
  it checks the matrix and the load vector against each other: get one
  sign wrong and the field collapses.
* against the plane-wave (anechoic) condition, does the ``1/r`` term
  earn its place? A monopole at the centre of the sphere radiates a
  known field, and the plane-wave condition reflects a fraction
  ``1 / sqrt(1 + (2kR)^2)`` of it back.
"""

import numpy as np
import pytest

from pycafe.boundary_condition.acoustic_bc import (
    AcousticBC,
    IncidentPlaneWave,
    build_radiation_operator,
)
from pycafe.build_matrices.bc_radiation import SphericalRadiation, fit_sphere
from pycafe.core.prepare_acoustic_system import prepare_acoustic_system
from pycafe.solver.solver_helmholtz_1 import solve_helmholtz_frequency_sweep

RHO, C0 = 997.0, 1500.0
RADIUS = 0.03


def sphere_mesh(tmp_path, elements_per_wavelength, frequency):
    """Tetrahedral ball of radius RADIUS, with the surface named."""
    gmsh = pytest.importorskip("gmsh")
    from pycafe.create_geom.visualize_mesh import load_mesh_with_groups

    h = (C0 / frequency) / elements_per_wavelength
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "ball.msh"

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("ball")
        volume = gmsh.model.occ.addSphere(0.0, 0.0, 0.0, RADIUS)
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(3, [volume], name="fluid")
        gmsh.model.addPhysicalGroup(
            2, [s for _dim, s in gmsh.model.getEntities(2)], name="radiation"
        )
        gmsh.option.setNumber("Mesh.MeshSizeMin", h)
        gmsh.option.setNumber("Mesh.MeshSizeMax", h)
        gmsh.model.mesh.generate(3)
        gmsh.write(str(path))
    finally:
        gmsh.finalize()

    return load_mesh_with_groups(str(path))


def solve(mesh, bc, frequency):
    """Pressure on the free DOFs, and the prepared system."""
    nodes, elements, boundaries, groups = mesh
    system = prepare_acoustic_system(
        nodes=nodes, elements=elements, boundaries=boundaries,
        rho=RHO, c0=C0, bc=bc, groups=groups,
    )
    P = solve_helmholtz_frequency_sweep(
        system["K_red"], system["M_red"], system["C_red"],
        np.array([frequency]),
        system["pressure_nodes_red"], system["pressure_values"],
        nodes=nodes, idx_free=system["idx_free"], rho=RHO,
        impedance_operator=system["C_red_op"],
        source_operator=system["source_red_op"],
        radiation_operator=system["radiation_red_op"],
    )
    return P[:, 0], system


class TestSphereFit:
    """The geometry the condition needs is read off the mesh."""

    def test_fits_a_sphere_exactly(self):
        rng = np.random.default_rng(0)
        direction = rng.normal(size=(200, 3))
        direction /= np.linalg.norm(direction, axis=1)[:, None]
        centre_in = np.array([0.1, -0.2, 0.3])
        points = centre_in + 0.05 * direction

        centre, radius, spread = fit_sphere(points)

        assert np.allclose(centre, centre_in, atol=1e-12)
        assert radius == pytest.approx(0.05, rel=1e-12)
        assert spread < 1e-10

    def test_a_cube_is_not_a_sphere(self):
        corners = np.array([[x, y, z]
                            for x in (0.0, 1.0)
                            for y in (0.0, 1.0)
                            for z in (0.0, 1.0)])
        face_centres = np.array([[0.5, 0.5, 0.0], [0.5, 0.5, 1.0],
                                 [0.5, 0.0, 0.5], [0.5, 1.0, 0.5],
                                 [0.0, 0.5, 0.5], [1.0, 0.5, 0.5]])
        _centre, _radius, spread = fit_sphere(
            np.vstack([corners, face_centres])
        )
        # Corners and face centres cannot lie on one sphere: 0.87 vs 0.5.
        assert spread > 0.2

    def test_too_few_points_is_refused(self):
        with pytest.raises(ValueError, match="at least 4 points"):
            fit_sphere(np.zeros((3, 3)))


class TestAssembly:
    """What the boundary integrates, before any solve."""

    @pytest.fixture(scope="class")
    def mesh(self, tmp_path_factory):
        return sphere_mesh(tmp_path_factory.mktemp("rad"), 8, 20e3)

    def test_area_and_radius_match_the_sphere(self, mesh):
        nodes, elements, boundaries, groups = mesh
        operator = build_radiation_operator(
            AcousticBC().add_spherical_radiation("radiation"),
            nodes=nodes, c0=C0,
            boundaries=boundaries, groups=groups, elements=elements,
        )
        term, = operator.terms

        assert term.radius == pytest.approx(RADIUS, rel=1e-6)
        assert term.spread < 1e-6
        assert np.allclose(term.centre, 0.0, atol=1e-9)
        # A faceted sphere is inscribed, so its area falls just short.
        assert term.area == pytest.approx(4 * np.pi * RADIUS ** 2, rel=0.05)
        assert term.area < 4 * np.pi * RADIUS ** 2

    def test_the_two_halves_of_the_condition(self, mesh):
        """``matrix`` is ``jk S + S/r``, and on a sphere ``S/r = S/R``."""
        nodes, elements, boundaries, groups = mesh
        operator = build_radiation_operator(
            AcousticBC().add_spherical_radiation("radiation"),
            nodes=nodes, c0=C0,
            boundaries=boundaries, groups=groups, elements=elements,
        )
        term, = operator.terms
        omega = 2 * np.pi * 20e3
        k = omega / C0

        A = term.matrix(omega)
        expected_imag = k * term._S.toarray()
        # The quadrature points sit slightly inside the sphere, so 1/r is
        # a little above 1/R: the agreement is geometric, not exact.
        expected_real = term._S.toarray() / RADIUS

        assert np.allclose(A.toarray().imag, expected_imag)
        assert np.allclose(A.toarray().real, expected_real, rtol=0.02)

    def test_without_an_incident_field_the_load_is_zero(self, mesh):
        nodes, elements, boundaries, groups = mesh
        operator = build_radiation_operator(
            AcousticBC().add_spherical_radiation("radiation"),
            nodes=nodes, c0=C0,
            boundaries=boundaries, groups=groups, elements=elements,
        )
        assert not operator.has_incident_field
        assert np.all(operator.load(2 * np.pi * 20e3) == 0.0)

    def test_a_non_spherical_boundary_warns(self, mesh):
        """The condition assumes a sphere and says so when it is not one."""
        nodes, elements, boundaries, groups = mesh
        with pytest.warns(RuntimeWarning, match="strays from the sphere"):
            build_radiation_operator(
                AcousticBC().add_spherical_radiation(
                    "radiation", centre=(0.5 * RADIUS, 0.0, 0.0)
                ),
                nodes=nodes, c0=C0,
                boundaries=boundaries, groups=groups, elements=elements,
            )


class TestIncidentPlaneWave:
    """With nothing to scatter it, the incident wave is the answer."""

    def test_direction_is_normalized(self):
        wave = IncidentPlaneWave(p0=2.0, direction=(0.0, 0.0, 3.0))
        assert np.allclose(wave.direction, [0.0, 0.0, 1.0])

    def test_the_zero_direction_is_refused(self):
        with pytest.raises(ValueError, match="zero vector"):
            IncidentPlaneWave(direction=(0.0, 0.0, 0.0))

    @pytest.mark.parametrize(
        "per_wavelength, tolerance", [(8, 0.06), (16, 0.02)]
    )
    def test_empty_domain_returns_the_incident_wave(
        self, tmp_path, per_wavelength, tolerance
    ):
        frequency = 20e3
        mesh = sphere_mesh(tmp_path, per_wavelength, frequency)
        nodes = mesh[0]

        wave = IncidentPlaneWave(p0=1.0, direction=(0.75, -0.433, -0.5))
        bc = AcousticBC().add_spherical_radiation("radiation", incident=wave)
        p, system = solve(mesh, bc, frequency)

        exact = system["radiation_op"].incident_field(
            nodes, 2 * np.pi * frequency
        )[system["idx_free"]]
        error = np.linalg.norm(p - exact) / np.linalg.norm(exact)
        assert error < tolerance

    def test_refining_the_mesh_reduces_the_error(self, tmp_path):
        """The error is discretization, not a modelling mistake."""
        frequency = 20e3
        errors = []
        for per_wavelength in (8, 16):
            mesh = sphere_mesh(
                tmp_path / f"h{per_wavelength}", per_wavelength, frequency
            )
            wave = IncidentPlaneWave(direction=(1.0, 0.0, 0.0))
            bc = AcousticBC().add_spherical_radiation(
                "radiation", incident=wave
            )
            p, system = solve(mesh, bc, frequency)
            exact = system["radiation_op"].incident_field(
                mesh[0], 2 * np.pi * frequency
            )[system["idx_free"]]
            errors.append(
                np.linalg.norm(p - exact) / np.linalg.norm(exact)
            )

        assert errors[1] < 0.5 * errors[0]


class TestAgainstThePlaneWaveCondition:
    """A monopole at the centre: what the ``1/r`` term is worth."""

    def test_the_curvature_term_earns_its_place(self, tmp_path):
        frequency, q = 20e3, 1e-6
        mesh = sphere_mesh(tmp_path, 20, frequency)
        nodes = mesh[0]
        omega = 2 * np.pi * frequency
        k = omega / C0
        centre_node = int(np.argmin(np.linalg.norm(nodes, axis=1)))

        def error(bc):
            p, system = solve(mesh, bc, frequency)
            r = np.linalg.norm(nodes[system["idx_free"]], axis=1)
            band = (r > 0.5 * RADIUS) & (r < 0.98 * RADIUS)
            exact = (1j * omega * RHO * q
                     * np.exp(-1j * k * r[band]) / (4 * np.pi * r[band]))
            return np.linalg.norm(p[band] - exact) / np.linalg.norm(exact)

        spherical = error(
            AcousticBC()
            .add_spherical_radiation("radiation")
            .add_monopole(q, node=centre_node)
        )
        plane = error(
            AcousticBC()
            .add_anechoic("radiation")
            .add_monopole(q, node=centre_node)
        )

        # kR = 2.5 here, so the plane-wave condition sends back about a
        # fifth of the amplitude: 1 / sqrt(1 + (2kR)^2) = 0.19.
        assert k * RADIUS == pytest.approx(2.51, abs=0.01)
        assert spherical < 0.05
        assert plane > 5 * spherical


class TestOperatorMechanics:
    """Reduction and the empty case, which the solvers rely on."""

    def test_an_empty_operator_contributes_nothing(self):
        operator = build_radiation_operator(
            AcousticBC(), nodes=np.zeros((4, 3)), c0=C0
        )
        assert operator.is_empty
        assert not operator.has_incident_field
        assert operator.matrix(1.0).nnz == 0
        assert np.all(operator.load(1.0) == 0.0)

    def test_reduce_restricts_both_the_matrix_and_the_load(self, tmp_path):
        mesh = sphere_mesh(tmp_path, 8, 20e3)
        nodes, elements, boundaries, groups = mesh
        wave = IncidentPlaneWave(direction=(0.0, 0.0, 1.0))
        operator = build_radiation_operator(
            AcousticBC().add_spherical_radiation("radiation", incident=wave),
            nodes=nodes, c0=C0,
            boundaries=boundaries, groups=groups, elements=elements,
        )
        omega = 2 * np.pi * 20e3
        keep = np.arange(0, nodes.shape[0], 2)

        reduced = operator.reduce(keep)

        assert reduced.shape == (keep.size, keep.size)
        assert np.allclose(
            reduced.matrix(omega).toarray(),
            operator.matrix(omega).toarray()[np.ix_(keep, keep)],
        )
        assert np.allclose(reduced.load(omega), operator.load(omega)[keep])

    def test_reduce_twice_composes(self, tmp_path):
        mesh = sphere_mesh(tmp_path, 8, 20e3)
        nodes, elements, boundaries, groups = mesh
        operator = build_radiation_operator(
            AcousticBC().add_spherical_radiation("radiation"),
            nodes=nodes, c0=C0,
            boundaries=boundaries, groups=groups, elements=elements,
        )
        first = np.arange(0, nodes.shape[0], 2)
        second = np.arange(0, first.size, 3)

        twice = operator.reduce(first).reduce(second)
        once = operator.reduce(first[second])

        assert np.allclose(twice.matrix(1e5).toarray(),
                           once.matrix(1e5).toarray())
