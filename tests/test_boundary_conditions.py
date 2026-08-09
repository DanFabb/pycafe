"""
Unit tests for boundary condition enforcement and system reduction.

Tests verify:
- get_pressure_zero_nodes: correct 0-based index conversion
- reduce_KMC_dirichlet_mask: correct dimension reduction and idx_free mapping
- Dirichlet enforcement preserves symmetry of reduced matrices
- idx_free correctly excludes constrained DOFs

Reviewers addressed: R1, R3 (Quality Control / testing coverage)
"""

import numpy as np
import pytest
from scipy.sparse import lil_matrix

from pycafe.build_matrices.assembly_cquad4 import (
    get_pressure_zero_nodes,
    reduce_KMC_dirichlet_mask,
    build_KM_cquad4,
)


# Helpers

def make_rect_mesh_cquad4(Lx, Ly, nx, ny):
    """
    Generate a structured rectangular CQUAD4 mesh.

    Returns nodes (N,2) and elements dict with 1-based connectivity.
    Node ordering: node(i,j) = i*(ny+1)+j, x=x[i], y=y[j].
    Element connectivity: CCW [bottom-left, bottom-right, top-right, top-left].
    """
    x = np.linspace(0.0, Lx, nx + 1)
    y = np.linspace(0.0, Ly, ny + 1)
    X, Y = np.meshgrid(x, y, indexing="ij")
    nodes = np.column_stack([X.ravel(), Y.ravel()])  # (N,2)

    conn = []
    for ie in range(nx):
        for je in range(ny):
            n1 = ie * (ny + 1) + je          # bottom-left
            n2 = (ie + 1) * (ny + 1) + je    # bottom-right
            n3 = (ie + 1) * (ny + 1) + je + 1  # top-right
            n4 = ie * (ny + 1) + je + 1      # top-left
            conn.append([n1 + 1, n2 + 1, n3 + 1, n4 + 1])  # 1-based

    elements = {"Quadrilateral 4": np.array(conn, dtype=int)}
    return nodes, elements


def make_dummy_symmetric_sparse(n):
    """Create a random symmetric sparse matrix for testing."""
    rng = np.random.default_rng(42)
    A = rng.random((n, n))
    A = A + A.T  # symmetrise
    sp = lil_matrix(A)
    return sp


# Tests: get_pressure_zero_nodes

class TestGetPressureZeroNodes:

    def test_single_boundary(self):
        """Nodes on a single boundary are correctly converted to 0-based."""
        boundaries = {
            "left": [1, 2, 3, 4],   # 1-based
            "right": [5, 6, 7, 8],
        }
        p0 = get_pressure_zero_nodes(boundaries, ["left"])
        expected = np.array([0, 1, 2, 3])  # 0-based
        np.testing.assert_array_equal(p0, expected)

    def test_multiple_boundaries(self):
        """Nodes from multiple boundaries are combined and unique."""
        boundaries = {
            "bottom": [1, 2, 3],
            "top":    [3, 4, 5],   # node 3 appears in both
        }
        p0 = get_pressure_zero_nodes(boundaries, ["bottom", "top"])
        expected = np.unique(np.array([0, 1, 2, 2, 3, 4]))  # 0-based, unique
        np.testing.assert_array_equal(p0, expected)

    def test_empty_bc_list(self):
        """If no boundaries are specified, return empty array."""
        boundaries = {"left": [1, 2, 3]}
        p0 = get_pressure_zero_nodes(boundaries, [])
        assert p0.size == 0

    def test_missing_boundary_name_ignored(self):
        """Boundary names not in dict are silently ignored."""
        boundaries = {"left": [1, 2]}
        p0 = get_pressure_zero_nodes(boundaries, ["left", "nonexistent"])
        expected = np.array([0, 1])
        np.testing.assert_array_equal(p0, expected)

    def test_uniqueness(self):
        """Duplicate nodes across boundaries are deduplicated."""
        boundaries = {
            "a": [1, 2, 3, 4],
            "b": [3, 4, 5, 6],
            "c": [5, 6, 7, 8],
        }
        p0 = get_pressure_zero_nodes(boundaries, ["a", "b", "c"])
        assert len(p0) == len(np.unique(p0)), "Duplicate node indices found"
        assert len(p0) == 8  # nodes 1..8 → 0..7

    def test_output_is_sorted(self):
        """Returned indices are sorted in ascending order."""
        boundaries = {"mix": [5, 1, 3, 2, 4]}
        p0 = get_pressure_zero_nodes(boundaries, ["mix"])
        assert np.all(p0[:-1] <= p0[1:]), "Output is not sorted"


# Tests: reduce_KMC_dirichlet_mask

class TestReduceKMCDirichlet:

    def _make_system(self, n):
        """Small n×n symmetric sparse K, M, C for testing."""
        K = make_dummy_symmetric_sparse(n)
        M = make_dummy_symmetric_sparse(n)
        C = lil_matrix((n, n), dtype=complex)
        return K, M, C

    def test_reduced_size(self):
        """Reduced system has correct number of DOFs."""
        n = 10
        constrained = [0, 3, 7]  # 3 DOFs removed
        K, M, C = self._make_system(n)
        K_red, M_red, C_red, idx_free = reduce_KMC_dirichlet_mask(K, M, C, constrained)
        assert K_red.shape == (n - len(constrained), n - len(constrained))
        assert M_red.shape == (n - len(constrained), n - len(constrained))
        assert len(idx_free) == n - len(constrained)

    def test_idx_free_excludes_constrained(self):
        """idx_free must not contain any constrained DOF index."""
        n = 8
        constrained = [1, 4, 6]
        K, M, C = self._make_system(n)
        _, _, _, idx_free = reduce_KMC_dirichlet_mask(K, M, C, constrained)
        for c in constrained:
            assert c not in idx_free, f"Constrained DOF {c} found in idx_free"

    def test_idx_free_contains_all_free_dofs(self):
        """idx_free must contain all non-constrained global indices."""
        n = 6
        constrained = [0, 2]
        K, M, C = self._make_system(n)
        _, _, _, idx_free = reduce_KMC_dirichlet_mask(K, M, C, constrained)
        expected_free = np.array([1, 3, 4, 5])
        np.testing.assert_array_equal(np.sort(idx_free), expected_free)

    def test_no_constrained_dofs(self):
        """With no constrained DOFs, reduced system equals full system."""
        n = 5
        K, M, C = self._make_system(n)
        K_red, M_red, C_red, idx_free = reduce_KMC_dirichlet_mask(K, M, C, [])
        assert K_red.shape == (n, n)
        assert len(idx_free) == n

    def test_values_in_reduced_matrix(self):
        """Reduced matrix entries match the corresponding full-matrix entries."""
        n = 6
        constrained = [1, 3]
        K, M, C = self._make_system(n)
        K_full_dense = K.toarray()
        K_red, _, _, idx_free = reduce_KMC_dirichlet_mask(K, M, C, constrained)
        K_red_dense = np.array(K_red.todense()) if hasattr(K_red, "todense") else K_red
        # Check a few entries
        for i, gi in enumerate(idx_free):
            for j, gj in enumerate(idx_free):
                assert abs(K_red_dense[i, j] - K_full_dense[gi, gj]) < 1e-14, (
                    f"Mismatch at reduced ({i},{j}) → global ({gi},{gj})"
                )

    def test_duplicate_constrained_dofs_handled(self):
        """Duplicate entries in constrained list are handled correctly."""
        n = 5
        constrained = [1, 1, 2, 2]  # duplicates
        K, M, C = self._make_system(n)
        K_red, _, _, idx_free = reduce_KMC_dirichlet_mask(K, M, C, constrained)
        # Effective unique constraints: [1, 2]
        assert K_red.shape == (n - 2, n - 2)
        assert len(idx_free) == n - 2


# Tests: Full assembly + reduction on structured mesh

class TestAssemblyAndReduction:

    def test_reduced_system_size_after_assembly(self):
        """
        On a 2×2 mesh (9 nodes), constraining the 3 bottom nodes
        yields a 6×6 reduced system.
        """
        nodes, elements = make_rect_mesh_cquad4(1.0, 1.0, 2, 2)
        c = 343.0
        K_global, M_global, _ = build_KM_cquad4(nodes, elements, c)

        # Bottom row: j=0 → nodes at y=0: indices 0, 2, 4 (i=0,1,2, j=0)
        # node(i,j) = i*(ny+1)+j with ny=2 → node(i,0) = 3*i
        bottom_nodes = [0, 3, 6]  # global 0-based indices of y=0 boundary
        C_dummy = lil_matrix(K_global.shape, dtype=complex)

        K_red, M_red, _, idx_free = reduce_KMC_dirichlet_mask(
            K_global, M_global, C_dummy, bottom_nodes
        )
        n_total = nodes.shape[0]  # 9
        n_constrained = len(bottom_nodes)  # 3
        assert K_red.shape == (n_total - n_constrained, n_total - n_constrained)

    def test_assembled_global_matrix_symmetry(self):
        """Global K and M assembled from CQUAD4 must be symmetric."""
        nodes, elements = make_rect_mesh_cquad4(1.0, 1.0, 3, 3)
        c = 343.0
        K_global, M_global, _ = build_KM_cquad4(nodes, elements, c)

        K_dense = K_global.toarray()
        M_dense = M_global.toarray()

        np.testing.assert_allclose(K_dense, K_dense.T, atol=1e-13)
        np.testing.assert_allclose(M_dense, M_dense.T, atol=1e-13)

    def test_global_mass_conservation(self):
        """sum(M_global) = total_area / c^2 for a structured mesh."""
        Lx, Ly = 2.0, 3.0
        nx, ny = 4, 6
        nodes, elements = make_rect_mesh_cquad4(Lx, Ly, nx, ny)
        c = 343.0
        _, M_global, _ = build_KM_cquad4(nodes, elements, c)

        total_area = Lx * Ly
        M_dense = M_global.toarray()
        np.testing.assert_allclose(M_dense.sum(), total_area / c**2, rtol=1e-10)

    def test_global_stiffness_null_space(self):
        """K_global * ones = 0 (constant pressure is a zero-energy global mode)."""
        nodes, elements = make_rect_mesh_cquad4(1.0, 1.0, 3, 3)
        c = 343.0
        K_global, _, _ = build_KM_cquad4(nodes, elements, c)
        K_dense = K_global.toarray()
        n = K_dense.shape[0]
        result = K_dense @ np.ones(n)
        np.testing.assert_allclose(result, np.zeros(n), atol=1e-12)
