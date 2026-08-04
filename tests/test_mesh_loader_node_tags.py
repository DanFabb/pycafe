"""
Tests for Gmsh node-tag handling in the mesh loader (visualize_mesh.py).

Gmsh node tags are arbitrary labels; only a mesh generated in one shot
has them contiguous, 1-based and sorted. Using a tag directly as an
array index therefore reads the wrong coordinates whenever a mesh has
been merged, refined or renumbered, which corrupts K and M silently.
The loader must remap tags to row indices of the coordinate array
(:class:`NodeIndex`) while keeping the 1-based connectivity contract.
"""

import numpy as np
import pytest

from pycafe.create_geom.visualize_mesh import (
    NodeIndex,
    load_mesh_and_elements,
    load_mesh_with_groups,
)

# Unit square, one quad + the four boundary lines, with node tags that
# are sparse and not in geometric order.
SPARSE_MSH = """$MeshFormat
2.2 0 8
$EndMeshFormat
$PhysicalNames
2
1 1 "edge"
2 2 "domain"
$EndPhysicalNames
$Nodes
4
101 0.0 0.0 0.0
57 1.0 0.0 0.0
2003 1.0 1.0 0.0
9 0.0 1.0 0.0
$EndNodes
$Elements
2
1 3 2 2 1 101 57 2003 9
2 1 2 1 2 101 57
$EndElements
"""

# Same mesh, same geometry, tags 1..4 in order: the reference the
# sparse-tag version must reproduce exactly.
DENSE_MSH = SPARSE_MSH.replace("101 0.0", "1 0.0").replace(
    "57 1.0 0.0", "2 1.0 0.0"
).replace("2003 1.0 1.0", "3 1.0 1.0").replace("9 0.0 1.0", "4 0.0 1.0").replace(
    "1 3 2 2 1 101 57 2003 9", "1 3 2 2 1 1 2 3 4"
).replace("2 1 2 1 2 101 57", "2 1 2 1 2 1 2")

QUAD_COORDS = np.array(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
)


@pytest.fixture(scope="module")
def sparse_mesh(tmp_path_factory):
    pytest.importorskip("gmsh")
    path = tmp_path_factory.mktemp("meshes") / "sparse_tags.msh"
    path.write_text(SPARSE_MSH)
    return str(path)


@pytest.fixture(scope="module")
def dense_mesh(tmp_path_factory):
    pytest.importorskip("gmsh")
    path = tmp_path_factory.mktemp("meshes") / "dense_tags.msh"
    path.write_text(DENSE_MSH)
    return str(path)


class TestNodeIndex:
    def test_identity_for_contiguous_sorted_tags(self):
        idx = NodeIndex(np.arange(1, 6))
        assert idx.identity
        conn = np.array([[1, 2, 3], [3, 4, 5]])
        assert np.array_equal(idx(conn), conn)

    def test_sparse_tags_map_to_rows(self):
        # rows 0..3 -> tags 101, 57, 2003, 9
        idx = NodeIndex([101, 57, 2003, 9])
        assert not idx.identity
        assert np.array_equal(idx([101, 57, 2003, 9]), [1, 2, 3, 4])

    def test_shape_is_preserved(self):
        idx = NodeIndex([10, 20, 30, 40])
        out = idx(np.array([[40, 10], [20, 30]]))
        assert out.shape == (2, 2)
        assert np.array_equal(out, [[4, 1], [2, 3]])

    def test_unknown_tag_raises(self):
        idx = NodeIndex([10, 20, 30])
        with pytest.raises(KeyError):
            idx([10, 999])

    def test_empty_input(self):
        idx = NodeIndex([10, 20, 30])
        assert idx(np.empty((0, 4), dtype=int)).shape == (0, 4)


class TestLoaderWithSparseTags:
    def test_connectivity_points_at_the_right_coordinates(self, sparse_mesh):
        nodes, elements, _ = load_mesh_and_elements(sparse_mesh, verbose=False)
        conn = elements["Quadrilateral 4"]
        assert conn.shape == (1, 4)
        # The quad must still be the unit square, in order.
        assert np.allclose(nodes[conn[0] - 1], QUAD_COORDS)

    def test_indices_are_within_range(self, sparse_mesh):
        nodes, elements, boundaries = load_mesh_and_elements(
            sparse_mesh, verbose=False
        )
        for conn in elements.values():
            assert conn.min() >= 1 and conn.max() <= nodes.shape[0]
        for name, nds in boundaries.items():
            assert min(nds) >= 1 and max(nds) <= nodes.shape[0], name

    def test_boundary_nodes_lie_on_their_edge(self, sparse_mesh):
        nodes, _, boundaries = load_mesh_and_elements(sparse_mesh, verbose=False)
        edge = np.array(boundaries["edge"]) - 1
        assert np.allclose(nodes[edge, 1], 0.0)  # y = 0 side

    def test_group_connectivity_is_remapped(self, sparse_mesh):
        nodes, _, _, groups = load_mesh_with_groups(sparse_mesh, verbose=False)
        line = groups["edge"]["elements"]["Line 2"]
        assert np.allclose(nodes[line[0] - 1, 1], 0.0)
        quad = groups["domain"]["elements"]["Quadrilateral 4"]
        assert np.allclose(nodes[quad[0] - 1], QUAD_COORDS)

    def test_matches_the_equivalent_dense_mesh(self, sparse_mesh, dense_mesh):
        """Sparse and contiguous tags must give the same mesh, node order apart."""
        n_s, e_s, b_s = load_mesh_and_elements(sparse_mesh, verbose=False)
        n_d, e_d, b_d = load_mesh_and_elements(dense_mesh, verbose=False)
        assert set(e_s) == set(e_d)
        for name in e_s:
            assert np.allclose(n_s[e_s[name] - 1], n_d[e_d[name] - 1])
        for name in b_s:
            assert np.allclose(
                np.sort(n_s[np.array(b_s[name]) - 1], axis=0),
                np.sort(n_d[np.array(b_d[name]) - 1], axis=0),
            )
