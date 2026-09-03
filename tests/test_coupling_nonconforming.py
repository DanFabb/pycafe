"""
Tests for the coupling of interfaces whose meshes do not share nodes.

The method under test is the interpolation route of Mi & Zheng (2018),
CMA 338:264-297, implemented in
``pycafe_vibro.coupling_nonconforming``.

Covered:
- a **conforming** interface is untouched: the fallback never runs, and
  the matrix is bit-for-bit the one built before this module existed;
- an interface whose two sides are meshed independently is detected as
  non-conforming and still transfers a uniform pressure as the vector
  area of the surface -- the identity that has to hold whatever the two
  grids are;
- the interpolation reproduces a constant pressure at the virtual layer
  (rows of ``A`` summing to one) for both variants of the paper;
- refusing is still possible (``nonconforming=False``) and raises the
  old error;
- a linear pressure field is reproduced by the moving least squares
  variant with a linear basis, which is what its polynomial degree
  promises;
- the structural side does not need to know: ``Kc`` keeps its shape,
  its sign convention and its zero rotational rows.
"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from pycafe_vibro.coupling import (
    build_coupling_matrix,
    interface_area_vector,
    interface_is_conforming,
)
from pycafe_vibro.coupling_nonconforming import (
    build_nonconforming_coupling,
    interpolation_matrix,
    shape_parameter,
    wet_boundary,
)

LX, LY, LZ = 0.4, 0.3, 0.2


def _hex_block(nx, ny, nz, Lx=LX, Ly=LY, Lz=LZ, z0=0.0):
    """A structured hexahedral block, nodes and connectivity (0-based)."""
    xs = np.linspace(0.0, Lx, nx + 1)
    ys = np.linspace(0.0, Ly, ny + 1)
    zs = np.linspace(z0, z0 + Lz, nz + 1)
    nodes = np.array([(x, y, z) for z in zs for y in ys for x in xs])

    def nid(i, j, k):
        return k * (nx + 1) * (ny + 1) + j * (nx + 1) + i

    conn = np.array([
        [nid(i, j, k), nid(i + 1, j, k), nid(i + 1, j + 1, k), nid(i, j + 1, k),
         nid(i, j, k + 1), nid(i + 1, j, k + 1), nid(i + 1, j + 1, k + 1),
         nid(i, j + 1, k + 1)]
        for k in range(nz) for j in range(ny) for i in range(nx)
    ])
    return nodes, conn


def _quad_grid(nx, ny, z, Lx=LX, Ly=LY, offset=0):
    """A quadrilateral grid on the plane ``z``, numbered from ``offset``."""
    xs = np.linspace(0.0, Lx, nx + 1)
    ys = np.linspace(0.0, Ly, ny + 1)
    nodes = np.array([(x, y, z) for y in ys for x in xs])
    conn = np.array([
        [offset + j * (nx + 1) + i,
         offset + j * (nx + 1) + i + 1,
         offset + (j + 1) * (nx + 1) + i + 1,
         offset + (j + 1) * (nx + 1) + i]
        for j in range(ny) for i in range(nx)
    ])
    return nodes, conn


@pytest.fixture(scope="module")
def conforming():
    """Plate meshed as the top faces of the fluid: nodes are shared."""
    nodes, fluid = _hex_block(4, 3, 2)
    top = np.isclose(nodes[:, 2], LZ)
    plate = np.array([
        [e[4], e[5], e[6], e[7]] for e in fluid
        if top[e[4]] and top[e[5]] and top[e[6]] and top[e[7]]
    ])
    return nodes, fluid, plate


@pytest.fixture(scope="module")
def nonconforming():
    """Same interface, meshed twice: 4x3 on the fluid, 5x4 on the plate."""
    fluid_nodes, fluid = _hex_block(4, 3, 2)
    plate_nodes, plate = _quad_grid(5, 4, LZ, offset=fluid_nodes.shape[0])
    nodes = np.vstack([fluid_nodes, plate_nodes])
    return nodes, fluid, plate


# the conforming case must not notice any of this
def test_conforming_interface_is_detected(conforming):
    nodes, fluid, plate = conforming
    ok, orphans, interior = interface_is_conforming(plate, fluid)
    assert ok
    assert orphans == [] and interior == []


def test_conforming_matrix_is_unchanged(conforming, recwarn):
    """The fallback must not run, and must not perturb the result."""
    nodes, fluid, plate = conforming

    strict = build_coupling_matrix(nodes, plate, fluid, nonconforming=False)
    auto = build_coupling_matrix(nodes, plate, fluid)          # default

    assert (strict - auto).nnz == 0
    assert np.array_equal(strict.toarray(), auto.toarray())
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


def test_conforming_report_says_so(conforming):
    nodes, fluid, plate = conforming
    report = {}
    build_coupling_matrix(nodes, plate, fluid, report=report)
    assert report == {"conforming": True}


# the interpolation itself
@pytest.mark.parametrize("method", ["mls", "bim"])
def test_interpolation_reproduces_a_constant(method):
    master = _quad_grid(4, 3, 0.0)[0]
    slave = _quad_grid(5, 4, 0.0)[0]
    A, info = interpolation_matrix(slave, master, method=method)

    p = np.full(master.shape[0], 2.5)
    assert np.allclose(A @ p, 2.5)
    assert info["n_master"] == master.shape[0]
    assert info["n_slave"] == slave.shape[0]


def test_mls_reproduces_a_linear_field():
    """Degree 1 is a linear fit, so a linear pressure comes back exactly."""
    master = _quad_grid(6, 5, 0.0)[0]
    slave = _quad_grid(7, 6, 0.0)[0]
    A, _ = interpolation_matrix(slave, master, method="mls", degree=1)

    def field(x):
        return 3.0 + 2.0 * x[:, 0] - 1.5 * x[:, 1]

    assert np.allclose(A @ field(master), field(slave), atol=1e-8)


def test_shape_parameter_follows_the_spacing():
    coarse = _quad_grid(2, 2, 0.0)[0]
    fine = _quad_grid(8, 8, 0.0)[0]
    assert shape_parameter(fine) > shape_parameter(coarse)


def test_wet_boundary_keeps_only_the_faces_near_the_plate(nonconforming):
    nodes, fluid, plate = nonconforming
    wet = wet_boundary(nodes, fluid, plate)

    assert np.allclose(nodes[wet["nodes0"], 2], LZ)
    assert np.allclose(wet["normals"], np.array([0.0, 0.0, 1.0]))


# the coupling matrix built through the interpolation
def test_nonconforming_is_detected_and_warned(nonconforming):
    nodes, fluid, plate = nonconforming
    ok, orphans, _ = interface_is_conforming(plate, fluid)
    assert not ok
    assert len(orphans) == plate.shape[0]

    with pytest.warns(RuntimeWarning, match="Mi & Zheng"):
        build_coupling_matrix(nodes, plate, fluid)


def test_nonconforming_can_still_be_refused(nonconforming):
    nodes, fluid, plate = nonconforming
    with pytest.raises(ValueError, match="not conforming"):
        build_coupling_matrix(nodes, plate, fluid, nonconforming=False)


@pytest.mark.parametrize("method", ["mls", "bim"])
def test_uniform_pressure_transfers_the_vector_area(nonconforming, method):
    nodes, fluid, plate = nonconforming
    Kc, report = build_nonconforming_coupling(nodes, plate, fluid,
                                              method=method)

    area = interface_area_vector(Kc)
    assert np.allclose(area, [0.0, 0.0, LX * LY], atol=1e-10)
    assert report["consistency"]["relative_error"] < 1e-10


def test_only_the_translations_are_loaded(nonconforming):
    nodes, fluid, plate = nonconforming
    Kc, _ = build_nonconforming_coupling(nodes, plate, fluid)

    dense = Kc.toarray()
    rotations = np.concatenate([np.arange(3, 6) + 6 * n
                                for n in range(nodes.shape[0])])
    assert np.allclose(dense[rotations], 0.0)


def test_pressure_columns_are_fluid_nodes(nonconforming):
    """The columns of Kc are the fluid unknowns, not the virtual ones."""
    nodes, fluid, plate = nonconforming
    Kc, report = build_nonconforming_coupling(nodes, plate, fluid)

    loaded = np.unique(csr_matrix(Kc).nonzero()[1])
    assert set(loaded.tolist()) <= set(report["master_nodes0"].tolist())
    assert not set(loaded.tolist()) & set(np.unique(plate).tolist())


def test_shape_and_normals(nonconforming):
    nodes, fluid, plate = nonconforming
    Kc, report = build_nonconforming_coupling(nodes, plate, fluid)

    assert Kc.shape == (6 * nodes.shape[0], nodes.shape[0])
    assert np.allclose(report["normals"], np.array([0.0, 0.0, 1.0]))


class TestCoupledModel:
    """
    The same cavity solved twice: plate meshed as the fluid faces, and
    plate meshed on its own with the same density. The second mesh
    shares no node with the fluid, so it goes through the interpolation;
    the coupled frequencies must be the ones of the first.
    """

    LX, LY, LZ = 0.8, 0.6, 0.5
    NX, NY, NZ = 8, 6, 5
    RHO0, C0 = 1.204, 343.0
    T, RHO_S, E, NU = 0.002, 7800.0, 210e9, 0.3

    def _system(self, path):
        from pycafe_vibro.prepare_vibroacoustic_system import (
            prepare_vibroacoustic_system,
        )
        from pycafe.create_geom.visualize_mesh import load_mesh_with_groups

        nodes, _, _, groups = load_mesh_with_groups(str(path), verbose=False)
        return prepare_vibroacoustic_system(
            nodes=nodes, groups=groups, rho0=self.RHO0, c0=self.C0,
            t=self.T, rho_s=self.RHO_S, E=self.E, nu=self.NU,
        )

    @pytest.fixture(scope="class")
    def conforming_mesh(self, tmp_path_factory):
        pytest.importorskip("gmsh")
        from pycafe.create_geom.library import box_with_plate

        return box_with_plate(
            self.LX, self.LY, self.LZ, self.NX, self.NY, self.NZ,
            output_path=tmp_path_factory.mktemp("c") / "box_plate.msh",
        )

    @pytest.fixture(scope="class")
    def split_mesh(self, tmp_path_factory):
        """Cavity and plate meshed independently, node for node apart."""
        pytest.importorskip("gmsh")
        from pycafe.create_geom.gmsh_workflow import GmshModel

        divisions = {0: self.NX, 1: self.NY, 2: self.NZ}
        with GmshModel("split", recombine=True) as m:
            box = m.occ.addBox(0, 0, 0, self.LX, self.LY, self.LZ)
            rect = m.occ.addRectangle(0, 0, self.LZ, self.LX, self.LY)
            m.occ.synchronize()
            for _d, curve in m.model.getEntities(1):
                bbox = np.asarray(m.model.getBoundingBox(1, curve))
                axis = int(np.argmax(np.abs(bbox[3:] - bbox[:3])))
                m.mesh.setTransfiniteCurve(curve, divisions[axis] + 1)
            for _d, surface in m.model.getEntities(2):
                m.mesh.setTransfiniteSurface(surface)
                m.mesh.setRecombine(2, surface)
            for _d, volume in m.model.getEntities(3):
                m.mesh.setTransfiniteVolume(volume)
            m.physical(3, [box], "fluid")
            m.physical(2, [rect], "plate")
            m.physical(1, [abs(t) for _d, t in m.model.getBoundary(
                [(2, rect)], oriented=False, combined=True)], "plate_clamp")
            m.generate(3, kernel="occ")
            return m.write(tmp_path_factory.mktemp("s") / "split.msh")

    def test_the_split_mesh_is_reported_as_non_conforming(self, split_mesh):
        from pycafe.create_geom.validation import validate_mesh

        report = validate_mesh(str(split_mesh))
        assert report.ok, "a non-conforming mesh is usable, with a warning"
        assert any("not conforming" in w for w in report.warnings)

    def test_the_interface_is_flagged_in_the_system(self, split_mesh):
        with pytest.warns(RuntimeWarning):
            system = self._system(split_mesh)

        assert system["interface"]["conforming"] is False
        assert np.allclose(system["coupling"]["area_vector"],
                           [0.0, 0.0, self.LX * self.LY], atol=1e-9)
        consistency = system["coupling"]["nonconforming"]["consistency"]
        assert consistency["relative_error"] < 1e-9

    def test_coupled_frequencies_match_the_conforming_model(
            self, conforming_mesh, split_mesh):
        from pycafe_vibro.solver_vibroacoustic import (
            solve_vibroacoustic_modal,
        )

        f_ref, *_ = solve_vibroacoustic_modal(
            self._system(conforming_mesh), num_modes=6)
        with pytest.warns(RuntimeWarning):
            split = self._system(split_mesh)
        f_split, *_ = solve_vibroacoustic_modal(split, num_modes=6)

        # The first eigenvalue is the constant-pressure mode at 0 Hz.
        assert np.allclose(f_ref[1:], f_split[1:], rtol=0.01)


def test_matches_the_conforming_matrix_when_the_grids_agree(conforming):
    """
    Same interface, same grid on both sides, but with the plate
    renumbered so that no node is shared: the interpolation has nothing
    to interpolate, and must return the conforming coupling.
    """
    nodes, fluid, plate = conforming
    n0 = nodes.shape[0]
    duplicate = nodes[np.unique(plate)]
    renumber = {int(old): n0 + i for i, old in enumerate(np.unique(plate))}
    plate_dup = np.vectorize(renumber.get)(plate)
    nodes_dup = np.vstack([nodes, duplicate])

    Kc_ref = build_coupling_matrix(nodes, plate, fluid, num_nodes=n0)
    Kc_nc, report = build_nonconforming_coupling(
        nodes_dup, plate_dup, fluid, num_nodes=nodes_dup.shape[0],
    )

    # Compare the loads a uniform pressure produces on the plate nodes,
    # each one read at its own row on the two sides.
    load_ref = np.asarray(Kc_ref.sum(axis=1)).ravel()
    load_nc = np.asarray(Kc_nc.sum(axis=1)).ravel()
    for old, new in renumber.items():
        assert np.allclose(load_ref[6 * old:6 * old + 3],
                           load_nc[6 * new:6 * new + 3], atol=1e-9)
    assert report["consistency"]["relative_error"] < 1e-10
