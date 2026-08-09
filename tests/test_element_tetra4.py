"""
Validation of the CTETRA4 linear tetrahedral acoustic element.

The formulation follows Felippa, "Advanced Finite Element Methods",
Chapter 15; the tests check it against the closed forms of that chapter
(volume, gradients, the mass integral (15.35)), against an independent
Jacobian route, and against the analytical modes of a rigid rectangular
cavity.
"""

import numpy as np
import pytest

from pycafe.build_matrices.assembly import assemble_KM
from pycafe.build_matrices.element_hex8 import element_matrices_hex8
from pycafe.build_matrices.element_registry import (
    ELEMENT_TYPES,
    find_acoustic_elements,
)
from pycafe.build_matrices.element_tetra4 import (
    element_matrices_tetra4,
    tetra4_geometry,
    tetra4_shape,
)
from pycafe.build_matrices.source_volume import (
    point_source_shape,
    volume_load_vector,
)
from pycafe.solver.solver_modale import solve_modal_acoustic_reduced


C0 = 343.0

# The reference tetrahedron of §15.2: unit legs along the three axes,
# volume 1/6.
REF_TET = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
])


# Helpers

def make_box_mesh_tetra4(Lx, Ly, Lz, nx, ny, nz):
    """
    Structured box mesh, each brick split into six CTETRA4.

    The node set is exactly that of ``make_box_mesh_hex8`` with the same
    counts, which is what lets the two elements be compared at equal
    number of degrees of freedom. Every tetrahedron is oriented so that
    the signed volume of §15.2.1 is positive.

    Returns (nodes, elements) with 1-based Gmsh-ordered connectivity.
    """
    x = np.linspace(0.0, Lx, nx + 1)
    y = np.linspace(0.0, Ly, ny + 1)
    z = np.linspace(0.0, Lz, nz + 1)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    nodes = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    def nid(i, j, k):
        return i * (ny + 1) * (nz + 1) + j * (nz + 1) + k

    # Kuhn decomposition of the brick into six tetrahedra, on the local
    # corner numbering used by the Gmsh "Hexahedron 8" ordering. Every
    # tetrahedron spans the main diagonal 0 -> 6; the two middle corners
    # are one of the six monotone edge paths between them. Any other
    # choice of diagonal leaves gaps and overlaps that still add up to
    # the right volume, so it is worth stating the rule rather than the
    # list: see test_the_split_tiles_the_brick.
    corners = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
               (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
    split = [(0, 1, 2, 6), (0, 1, 5, 6), (0, 3, 2, 6),
             (0, 3, 7, 6), (0, 4, 5, 6), (0, 4, 7, 6)]

    conn = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                v = [nid(i + a, j + b, k + c) for a, b, c in corners]
                for tet in split:
                    q = [v[m] for m in tet]
                    p = nodes[q]
                    if np.linalg.det(np.array([p[1] - p[0], p[2] - p[0],
                                               p[3] - p[0]])) < 0.0:
                        q[2], q[3] = q[3], q[2]
                    conn.append([m + 1 for m in q])

    return nodes, {"Tetrahedron 4": np.array(conn, dtype=int)}


def make_box_mesh_hex8(Lx, Ly, Lz, nx, ny, nz):
    """Same box, one HEXA8 per brick — the reference to compare against."""
    x = np.linspace(0.0, Lx, nx + 1)
    y = np.linspace(0.0, Ly, ny + 1)
    z = np.linspace(0.0, Lz, nz + 1)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    nodes = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    def nid(i, j, k):
        return i * (ny + 1) * (nz + 1) + j * (nz + 1) + k

    conn = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                conn.append([m + 1 for m in [
                    nid(i, j, k), nid(i + 1, j, k),
                    nid(i + 1, j + 1, k), nid(i, j + 1, k),
                    nid(i, j, k + 1), nid(i + 1, j, k + 1),
                    nid(i + 1, j + 1, k + 1), nid(i, j + 1, k + 1),
                ]])

    return nodes, {"Hexahedron 8": np.array(conn, dtype=int)}


def analytical_box_freqs(Lx, Ly, Lz, c0, n_modes):
    freqs = []
    for l in range(4):
        for m in range(4):
            for n in range(4):
                if (l, m, n) == (0, 0, 0):
                    continue
                freqs.append(0.5 * c0 * np.sqrt(
                    (l / Lx) ** 2 + (m / Ly) ** 2 + (n / Lz) ** 2
                ))
    return np.sort(freqs)[:n_modes]


def build_KM(nodes, elements, kernel, key):
    K, M, _ = assemble_KM(nodes, elements[key] - 1, kernel, C0)
    return K, M


# Geometry: the closed forms of Chapter 15

class TestGeometry:

    def test_reference_volume(self):
        volume, _ = tetra4_geometry(REF_TET)
        assert volume == pytest.approx(1.0 / 6.0)

    def test_gradients_match_the_jacobian_route(self):
        """(15.7)-(15.10) against the inverse-Jacobian computation."""
        rng = np.random.default_rng(7)
        for _ in range(20):
            x_e = REF_TET + 0.3 * rng.standard_normal((4, 3))
            try:
                volume, grad_N = tetra4_geometry(x_e)
            except ValueError:
                continue                      # inverted by the perturbation

            J = np.column_stack([x_e[1] - x_e[0], x_e[2] - x_e[0],
                                 x_e[3] - x_e[0]])
            dN_ref = np.array([[-1.0, -1.0, -1.0], [1.0, 0.0, 0.0],
                               [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
            assert np.allclose(grad_N, np.linalg.inv(J).T @ dN_ref.T)
            assert volume == pytest.approx(abs(np.linalg.det(J)) / 6.0)

    def test_gradients_sum_to_zero(self):
        """Partition of unity: a constant pressure has no gradient."""
        _, grad_N = tetra4_geometry(REF_TET * np.array([2.0, 3.0, 0.5]))
        assert np.allclose(grad_N.sum(axis=1), 0.0)

    def test_flat_tetrahedron_is_rejected(self):
        coplanar = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
        with pytest.raises(ValueError, match="Non-positive"):
            tetra4_geometry(coplanar)

    def test_wrong_orientation_is_rejected(self):
        """Swapping two corners flips the sign of the volume (§15.2.1)."""
        flipped = REF_TET[[0, 2, 1, 3]]
        with pytest.raises(ValueError, match="Non-positive"):
            tetra4_geometry(flipped)

    def test_shape_functions_are_the_tetrahedral_coordinates(self):
        assert np.allclose(tetra4_shape([1.0, 0.0, 0.0, 0.0]),
                           [1.0, 0.0, 0.0, 0.0])
        assert np.allclose(tetra4_shape([0.25] * 4), [0.25] * 4)
        with pytest.raises(ValueError, match="sum to 1"):
            tetra4_shape([0.5, 0.0, 0.0, 0.0])


# Element matrices

class TestElementMatrices:

    def test_symmetry(self):
        K_e, M_e = element_matrices_tetra4(REF_TET, C0)
        assert np.allclose(K_e, K_e.T)
        assert np.allclose(M_e, M_e.T)

    def test_constant_pressure_is_in_the_kernel_of_K(self):
        K_e, _ = element_matrices_tetra4(REF_TET * 1.7, C0)
        assert np.allclose(K_e @ np.ones(4), 0.0, atol=1e-12)

    def test_mass_matrix_is_the_closed_form_15_35(self):
        x_e = REF_TET * np.array([2.0, 3.0, 0.5])
        volume, _ = tetra4_geometry(x_e)
        _, M_e = element_matrices_tetra4(x_e, C0)

        expected = (volume / 20.0) * (np.ones((4, 4)) + np.eye(4)) / C0 ** 2
        assert np.allclose(M_e, expected)
        # int zeta_i dV = V/4, equation (15.34)
        assert np.allclose(M_e.sum(axis=1), volume / 4.0 / C0 ** 2)
        # and the total is the volume of the element
        assert M_e.sum() * C0 ** 2 == pytest.approx(volume)

    def test_matrices_scale_with_the_element(self):
        """
        Under a uniform dilation by ``s``: the gradients go as 1/s and
        the volume as s^3, so K grows linearly with s, while M follows
        the volume.
        """
        K1, M1 = element_matrices_tetra4(REF_TET, C0)
        K2, M2 = element_matrices_tetra4(2.0 * REF_TET, C0)
        assert np.allclose(2.0 * K1, K2)
        assert np.allclose(8.0 * M1, M2)

    def test_bad_shape_is_rejected(self):
        with pytest.raises(ValueError, match=r"shape \(4, 3\)"):
            element_matrices_tetra4(REF_TET[:3], C0)


# Registry wiring

class TestRegistry:

    def test_registered_as_a_3d_acoustic_element(self):
        spec = ELEMENT_TYPES["CTETRA4"]
        assert spec.dim == 3
        assert spec.field == "acoustic"
        assert spec.n_nodes == 4
        assert spec.dofs_per_node == 1
        assert spec.kernel is element_matrices_tetra4

    def test_detected_in_a_mesh(self):
        _, elements = make_box_mesh_tetra4(1.0, 1.0, 1.0, 2, 2, 2)
        elem_type, elem_key, spec = find_acoustic_elements(elements)
        assert elem_type == "CTETRA4"
        assert elem_key == "Tetrahedron 4"
        assert spec.kernel is element_matrices_tetra4

    def test_volume_faces_do_not_win_over_the_tetrahedra(self):
        """A tet mesh carrying its Triangle 3 skin still assembles as 3D."""
        _, elements = make_box_mesh_tetra4(1.0, 1.0, 1.0, 2, 2, 2)
        elements = dict(elements)
        elements["Triangle 3"] = np.array([[1, 2, 3]], dtype=int)
        elem_type, _, _ = find_acoustic_elements(elements)
        assert elem_type == "CTETRA4"


# Analytical validation: rigid rectangular cavity

class TestMeshHelper:
    """
    The helper is the reference mesh of the whole file, so it is checked
    rather than assumed.
    """

    def test_the_split_tiles_the_brick(self):
        """
        Every point of the box lies in exactly one tetrahedron — no gap,
        no overlap. Summing the volumes is not enough to see this: a
        wrong choice of diagonal produces a gap and an overlap of equal
        measure, which the volume total hides.
        """
        nodes, elements = make_box_mesh_tetra4(1.0, 1.0, 1.0, 1, 1, 1)
        conn = elements["Tetrahedron 4"] - 1

        rng = np.random.default_rng(0)
        pts = rng.random((4000, 3))
        counts = np.zeros(len(pts), dtype=int)
        for elem in conn:
            p = nodes[elem]
            J = np.column_stack([p[1] - p[0], p[2] - p[0], p[3] - p[0]])
            loc = np.linalg.solve(J, (pts - p[0]).T).T
            N = np.column_stack([1.0 - loc.sum(axis=1), loc])
            counts += np.all(N >= -1e-12, axis=1)

        assert np.all(counts == 1)

    def test_volumes_add_up(self):
        Lx, Ly, Lz = 0.9, 0.4, 0.6
        nodes, elements = make_box_mesh_tetra4(Lx, Ly, Lz, 2, 2, 2)
        total = sum(tetra4_geometry(nodes[elem - 1])[0]
                    for elem in elements["Tetrahedron 4"])
        assert total == pytest.approx(Lx * Ly * Lz)


class TestCavityModes:
    """Cavity 1.0 x 0.7 x 0.5 m, rigid walls. f_100 = 171.5 Hz."""

    LX, LY, LZ = 1.0, 0.7, 0.5

    def _freqs(self, nx, ny, nz, n_modes=3):
        nodes, elements = make_box_mesh_tetra4(
            self.LX, self.LY, self.LZ, nx, ny, nz
        )
        K, M = build_KM(nodes, elements, element_matrices_tetra4,
                        "Tetrahedron 4")
        freqs, _ = solve_modal_acoustic_reduced(K, M, num_modes=n_modes)
        return freqs

    def test_first_three_modes(self):
        # 16 x 11 x 8: the mesh a brick needs (12 x 8 x 6) leaves the
        # (1,1,0) mode at 1.3%, so the tetrahedra get a finer one to
        # meet the same 1% bar the CHEXA8 is held to.
        freqs = self._freqs(16, 11, 8)
        exact = analytical_box_freqs(self.LX, self.LY, self.LZ, C0, 3)
        for f_fem, f_ex in zip(freqs, exact):
            rel = abs(f_fem - f_ex) / f_ex
            assert rel < 0.01, (
                f"FEM={f_fem:.3f} Hz, exact={f_ex:.3f} Hz, err={rel*100:.2f}%"
            )

    def test_convergence_is_second_order(self):
        exact = C0 / (2.0 * self.LX)
        errs = [abs(self._freqs(n, max(2, round(n * 0.7)),
                                max(2, round(n * 0.5)), 1)[0] - exact) / exact
                for n in (4, 8, 16)]
        assert errs[0] > errs[1] > errs[2]
        # halving h divides the error by about four
        assert errs[0] / errs[1] > 2.0
        assert errs[1] / errs[2] > 2.0

    def test_against_hexa8_on_the_same_nodes(self):
        """
        Six tetrahedra per brick against one brick, same node set.

        A regression guard on the relative behaviour of the two
        elements, not a statement about which to use: the bounds below
        are tolerances chosen around what the code currently does.
        """
        nx, ny, nz = 12, 8, 6
        nodes_t, el_t = make_box_mesh_tetra4(self.LX, self.LY, self.LZ,
                                             nx, ny, nz)
        nodes_h, el_h = make_box_mesh_hex8(self.LX, self.LY, self.LZ,
                                           nx, ny, nz)
        assert np.allclose(nodes_t, nodes_h)
        assert el_t["Tetrahedron 4"].shape[0] == 6 * el_h["Hexahedron 8"].shape[0]

        K_t, M_t = build_KM(nodes_t, el_t, element_matrices_tetra4,
                            "Tetrahedron 4")
        K_h, M_h = build_KM(nodes_h, el_h, element_matrices_hex8,
                            "Hexahedron 8")
        f_t, _ = solve_modal_acoustic_reduced(K_t, M_t, num_modes=3)
        f_h, _ = solve_modal_acoustic_reduced(K_h, M_h, num_modes=3)

        exact = analytical_box_freqs(self.LX, self.LY, self.LZ, C0, 3)
        err_t = np.abs(f_t - exact) / exact
        err_h = np.abs(f_h - exact) / exact

        # the two axis-aligned modes agree between the elements
        assert np.allclose(f_t[:2], f_h[:2], rtol=1e-3)
        # on the (1,1,0) mode they do not; the ratio is bracketed loosely
        assert 1.5 < err_t[2] / err_h[2] < 4.0
        assert err_t[2] < 0.02

    def test_uniform_pressure_is_in_the_kernel_of_the_global_K(self):
        nodes, elements = make_box_mesh_tetra4(
            self.LX, self.LY, self.LZ, 3, 3, 3
        )
        K, _ = build_KM(nodes, elements, element_matrices_tetra4,
                        "Tetrahedron 4")
        assert np.allclose(K @ np.ones(nodes.shape[0]), 0.0, atol=1e-10)


# Volume sources on a tetrahedral mesh

class TestVolumeSources:

    def test_load_vector_integrates_to_the_volume(self):
        Lx, Ly, Lz = 0.9, 0.4, 0.6
        nodes, elements = make_box_mesh_tetra4(Lx, Ly, Lz, 3, 2, 2)
        g = volume_load_vector(nodes, elements, nodes.shape[0])
        assert g.sum() == pytest.approx(Lx * Ly * Lz)
        assert np.all(g > 0.0)

    def test_point_source_lands_in_the_right_element(self):
        nodes, elements = make_box_mesh_tetra4(1.0, 1.0, 1.0, 2, 2, 2)
        x_s = np.array([0.31, 0.22, 0.13])
        idx, N = point_source_shape(nodes, elements, x_s)

        assert N.sum() == pytest.approx(1.0)
        assert np.all(N >= -1e-9)
        # the shape functions interpolate the source position back
        assert np.allclose(N @ nodes[idx], x_s)

    def test_source_on_a_node_is_a_pure_nodal_source(self):
        nodes, elements = make_box_mesh_tetra4(1.0, 1.0, 1.0, 2, 2, 2)
        target = 5
        idx, N = point_source_shape(nodes, elements, nodes[target])
        assert np.allclose(N[idx == target], 1.0)
        assert np.allclose(N[idx != target], 0.0, atol=1e-9)

    def test_point_outside_the_mesh_is_refused(self):
        """
        The tensor containment test would have accepted this: the point
        sits in the bounding box of a corner tetrahedron but outside the
        tetrahedron itself.
        """
        nodes, elements = make_box_mesh_tetra4(1.0, 1.0, 1.0, 1, 1, 1)
        with pytest.raises(ValueError, match="No element contains"):
            point_source_shape(nodes, elements, np.array([1.4, 1.4, 1.4]))


# End to end: an unstructured gmsh mesh through the dispatcher

class TestUnstructuredMesh:
    """
    The structured split above is a favourable case. This is the mesh a
    tetrahedral mesher actually produces, read back through the loader
    and assembled through the registry, with no hand-built connectivity
    anywhere.
    """

    def test_gmsh_cavity_modes(self, tmp_path):
        gmsh = pytest.importorskip("gmsh")
        from pycafe.build_matrices.assembly_dispatcher import build_KM_acoustic
        from pycafe.create_geom.visualize_mesh import load_mesh_and_elements

        Lx, Ly, Lz = 1.0, 0.7, 0.5
        path = tmp_path / "cavity_tet.msh"

        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("cavity")
            gmsh.model.occ.addBox(0.0, 0.0, 0.0, Lx, Ly, Lz)
            gmsh.model.occ.synchronize()
            gmsh.model.addPhysicalGroup(3, [1], name="fluid")
            gmsh.option.setNumber("Mesh.MeshSizeMax", 0.06)
            gmsh.model.mesh.generate(3)
            gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
            gmsh.write(str(path))
        finally:
            gmsh.finalize()

        nodes, elements, _ = load_mesh_and_elements(str(path))
        assert "Tetrahedron 4" in elements

        K, M = build_KM_acoustic(nodes, elements, C0)[:2]
        freqs, _ = solve_modal_acoustic_reduced(K, M, num_modes=3)

        exact = analytical_box_freqs(Lx, Ly, Lz, C0, 3)
        for f_fem, f_ex in zip(freqs, exact):
            rel = abs(f_fem - f_ex) / f_ex
            assert rel < 0.01, (
                f"FEM={f_fem:.3f} Hz, exact={f_ex:.3f} Hz, err={rel*100:.2f}%"
            )
