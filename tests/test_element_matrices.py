"""
Unit tests for element-level stiffness and mass matrices.

Tests verify:
- Symmetry of K_e and M_e
- K_e * ones = 0 (constant pressure is a free mode for Neumann BC)
- Mass conservation: sum(M_e) = element_area / c^2
- Positive definiteness of M_e
- Partition of unity for shape functions: sum(N_i) = 1
- Kronecker delta at element nodes: N_i(x_j) = delta_ij
- Correct Jacobian for rectangular elements

Reviewers addressed: R1, R3, R5 (Quality Control / testing coverage)
"""

import numpy as np
import pytest

from pycafe.build_matrices.element_cquad4 import (
    element_matrices_cquad4,
    cquad4_shape,
    jacobian_2d,
    gauss_rule_quad_2x2,
)
from pycafe.build_matrices.element_cquad8 import (
    element_matrices_cquad8,
    cquad8_shape,
    gauss_rule_quad_3x3,
)

# Helpers

def unit_square_cquad4():
    """Unit square [0,1]^2 — 4 corner nodes, CCW ordering."""
    return np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ], dtype=float)


def rect_cquad4(Lx, Ly):
    """Rectangular element [0,Lx] x [0,Ly]."""
    return np.array([
        [0.0,  0.0],
        [Lx,   0.0],
        [Lx,   Ly],
        [0.0,  Ly],
    ], dtype=float)


def unit_square_cquad8():
    """Unit square with mid-side nodes (CQUAD8)."""
    return np.array([
        [0.0,  0.0],   # corner 1
        [1.0,  0.0],   # corner 2
        [1.0,  1.0],   # corner 3
        [0.0,  1.0],   # corner 4
        [0.5,  0.0],   # mid 5 (bottom)
        [1.0,  0.5],   # mid 6 (right)
        [0.5,  1.0],   # mid 7 (top)
        [0.0,  0.5],   # mid 8 (left)
    ], dtype=float)


# CQUAD4 Shape Function Tests

class TestCQUAD4ShapeFunctions:

    def test_partition_of_unity(self):
        """sum(N_i) = 1 at arbitrary points inside the element."""
        test_points = [
            (0.0, 0.0), (0.5, 0.3), (-0.7, 0.2),
            (1.0, 1.0), (-1.0, -1.0), (0.0, 1.0),
        ]
        for xi, eta in test_points:
            N, _, _ = cquad4_shape(xi, eta)
            assert abs(N.sum() - 1.0) < 1e-14, (
                f"Partition of unity failed at (xi={xi}, eta={eta}): sum={N.sum()}"
            )

    def test_kronecker_delta_at_nodes(self):
        """N_i(x_j) = delta_ij at the four corner nodes."""
        # Reference-space coordinates of each node
        node_coords = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
        for j, (xi_j, eta_j) in enumerate(node_coords):
            N, _, _ = cquad4_shape(xi_j, eta_j)
            for i in range(4):
                expected = 1.0 if i == j else 0.0
                assert abs(N[i] - expected) < 1e-14, (
                    f"Kronecker delta failed: N_{i}(node {j}) = {N[i]}, expected {expected}"
                )

    def test_sum_of_derivatives_is_zero(self):
        """sum(dN_i/dxi) = 0 and sum(dN_i/deta) = 0 (compatibility)."""
        test_points = [(0.0, 0.0), (0.3, -0.5), (-0.9, 0.7)]
        for xi, eta in test_points:
            _, dN_dxi, dN_deta = cquad4_shape(xi, eta)
            assert abs(dN_dxi.sum()) < 1e-14
            assert abs(dN_deta.sum()) < 1e-14


# CQUAD8 Shape Function Tests

class TestCQUAD8ShapeFunctions:

    def test_partition_of_unity(self):
        """sum(N_i) = 1 at arbitrary points."""
        test_points = [
            (0.0, 0.0), (0.5, 0.3), (-0.7, 0.2), (0.0, 1.0),
        ]
        for xi, eta in test_points:
            N, _, _ = cquad8_shape(xi, eta)
            assert abs(N.sum() - 1.0) < 1e-12, (
                f"Partition of unity failed at (xi={xi}, eta={eta}): sum={N.sum()}"
            )

    def test_kronecker_delta_at_corner_nodes(self):
        """N_i(x_j) = delta_ij at the four corner nodes (i=0..3)."""
        corner_coords = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
        for j, (xi_j, eta_j) in enumerate(corner_coords):
            N, _, _ = cquad8_shape(xi_j, eta_j)
            assert abs(N[j] - 1.0) < 1e-12, (
                f"N_{j}(corner {j}) = {N[j]}, expected 1.0"
            )
            for i in range(8):
                if i != j:
                    assert abs(N[i]) < 1e-12, (
                        f"N_{i}(corner {j}) = {N[i]}, expected 0.0"
                    )

    def test_kronecker_delta_at_midside_nodes(self):
        """N_i(x_j) = delta_ij at the four mid-side nodes (i=4..7)."""
        mid_coords = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # nodes 5,6,7,8
        for j, (xi_j, eta_j) in enumerate(mid_coords):
            N, _, _ = cquad8_shape(xi_j, eta_j)
            node_idx = j + 4  # nodes are 4..7 in 0-based indexing
            assert abs(N[node_idx] - 1.0) < 1e-12, (
                f"N_{node_idx}(midside {j}) = {N[node_idx]}, expected 1.0"
            )
            for i in range(8):
                if i != node_idx:
                    assert abs(N[i]) < 1e-12, (
                        f"N_{i}(midside {j}) = {N[i]}, expected 0.0"
                    )

    def test_sum_of_derivatives_is_zero(self):
        """sum(dN_i/dxi) = 0 and sum(dN_i/deta) = 0."""
        test_points = [(0.0, 0.0), (0.3, -0.5), (-0.2, 0.7)]
        for xi, eta in test_points:
            _, dN_dxi, dN_deta = cquad8_shape(xi, eta)
            assert abs(dN_dxi.sum()) < 1e-12
            assert abs(dN_deta.sum()) < 1e-12


# CQUAD4 Jacobian Tests

class TestCQUAD4Jacobian:

    def test_unit_square_jacobian(self):
        """For a unit square [0,1]^2, Jacobian must be 0.5*I and detJ=0.25."""
        x2d = unit_square_cquad4()
        # Test at center of reference element
        N, dN_dxi, dN_deta = cquad4_shape(0.0, 0.0)
        J, detJ, invJ = jacobian_2d(dN_dxi, dN_deta, x2d)

        np.testing.assert_allclose(J, 0.5 * np.eye(2), atol=1e-14)
        assert abs(detJ - 0.25) < 1e-14

    def test_rectangular_element_jacobian(self):
        """For a rectangle [0,Lx]x[0,Ly], detJ = Lx*Ly/4 everywhere."""
        Lx, Ly = 2.0, 3.0
        x2d = rect_cquad4(Lx, Ly)
        xi_pts, eta_pts, _ = gauss_rule_quad_2x2()
        for xi, eta in zip(xi_pts, eta_pts):
            _, dN_dxi, dN_deta = cquad4_shape(xi, eta)
            _, detJ, _ = jacobian_2d(dN_dxi, dN_deta, x2d)
            assert abs(detJ - Lx * Ly / 4.0) < 1e-12

    def test_jacobian_positive(self):
        """detJ must be positive for a valid (non-inverted) element."""
        x2d = unit_square_cquad4()
        xi_pts, eta_pts, _ = gauss_rule_quad_2x2()
        for xi, eta in zip(xi_pts, eta_pts):
            _, dN_dxi, dN_deta = cquad4_shape(xi, eta)
            _, detJ, _ = jacobian_2d(dN_dxi, dN_deta, x2d)
            assert detJ > 0.0


# CQUAD4 Element Matrix Tests

class TestCQUAD4ElementMatrices:

    @pytest.fixture
    def unit_matrices(self):
        c = 343.0
        x_e = unit_square_cquad4()
        K_e, M_e = element_matrices_cquad4(x_e, c)
        return K_e, M_e, c, 1.0  # 1.0 = area of unit square

    def test_stiffness_symmetry(self, unit_matrices):
        K_e, _, _, _ = unit_matrices
        np.testing.assert_allclose(K_e, K_e.T, atol=1e-14)

    def test_mass_symmetry(self, unit_matrices):
        _, M_e, _, _ = unit_matrices
        np.testing.assert_allclose(M_e, M_e.T, atol=1e-14)

    def test_stiffness_null_space_constant_pressure(self, unit_matrices):
        """K_e * [1,1,1,1] = 0: constant pressure is a zero-energy mode."""
        K_e, _, _, _ = unit_matrices
        result = K_e @ np.ones(4)
        np.testing.assert_allclose(result, np.zeros(4), atol=1e-13)

    def test_mass_conservation(self, unit_matrices):
        """sum(M_e) = area / c^2."""
        _, M_e, c, area = unit_matrices
        np.testing.assert_allclose(M_e.sum(), area / c**2, rtol=1e-12)

    def test_mass_positive_definite(self, unit_matrices):
        """All eigenvalues of M_e must be positive."""
        _, M_e, _, _ = unit_matrices
        eigvals = np.linalg.eigvalsh(M_e)
        assert np.all(eigvals > 0), f"M_e has non-positive eigenvalues: {eigvals}"

    def test_stiffness_positive_semidefinite(self, unit_matrices):
        """K_e eigenvalues must be >= 0 (one zero for constant pressure mode)."""
        K_e, _, _, _ = unit_matrices
        eigvals = np.linalg.eigvalsh(K_e)
        assert np.all(eigvals >= -1e-14), (
            f"K_e has negative eigenvalues: {eigvals}"
        )

    def test_stiffness_exact_rank(self, unit_matrices):
        """K_e must have exactly one zero eigenvalue (null space = constant)."""
        K_e, _, _, _ = unit_matrices
        eigvals = np.linalg.eigvalsh(K_e)
        n_zero = np.sum(np.abs(eigvals) < 1e-12)
        assert n_zero == 1, (
            f"K_e should have exactly 1 zero eigenvalue, found {n_zero}"
        )

    def test_mass_scales_correctly_with_c(self):
        """Doubling c reduces all M_e entries by factor of 4."""
        x_e = unit_square_cquad4()
        c1 = 100.0
        c2 = 200.0
        _, M1 = element_matrices_cquad4(x_e, c1)
        _, M2 = element_matrices_cquad4(x_e, c2)
        np.testing.assert_allclose(M1, M2 * (c2/c1)**2, rtol=1e-12)

    def test_stiffness_invariant_with_c(self):
        """K_e must not depend on the speed of sound c."""
        x_e = unit_square_cquad4()
        K1, _ = element_matrices_cquad4(x_e, 100.0)
        K2, _ = element_matrices_cquad4(x_e, 500.0)
        np.testing.assert_allclose(K1, K2, atol=1e-14)

    def test_mass_scales_with_element_area(self):
        """sum(M_e) = area / c^2 for arbitrary rectangular elements."""
        c = 343.0
        for Lx, Ly in [(0.5, 0.5), (2.0, 3.0), (0.1, 10.0)]:
            x_e = rect_cquad4(Lx, Ly)
            area = Lx * Ly
            _, M_e = element_matrices_cquad4(x_e, c)
            np.testing.assert_allclose(
                M_e.sum(), area / c**2, rtol=1e-10,
                err_msg=f"Mass conservation failed for Lx={Lx}, Ly={Ly}"
            )


# CQUAD8 Element Matrix Tests

class TestCQUAD8ElementMatrices:

    @pytest.fixture
    def unit_matrices(self):
        c = 343.0
        x_e = unit_square_cquad8()
        K_e, M_e = element_matrices_cquad8(x_e, c)
        return K_e, M_e, c, 1.0  # area = 1.0

    def test_stiffness_symmetry(self, unit_matrices):
        K_e, _, _, _ = unit_matrices
        np.testing.assert_allclose(K_e, K_e.T, atol=1e-12)

    def test_mass_symmetry(self, unit_matrices):
        _, M_e, _, _ = unit_matrices
        np.testing.assert_allclose(M_e, M_e.T, atol=1e-12)

    def test_stiffness_null_space_constant_pressure(self, unit_matrices):
        """K_e * ones = 0 for CQUAD8 (constant pressure, zero gradient)."""
        K_e, _, _, _ = unit_matrices
        result = K_e @ np.ones(8)
        np.testing.assert_allclose(result, np.zeros(8), atol=1e-11)

    def test_mass_conservation(self, unit_matrices):
        """sum(M_e) = area / c^2 for CQUAD8."""
        _, M_e, c, area = unit_matrices
        np.testing.assert_allclose(M_e.sum(), area / c**2, rtol=1e-10)

    def test_mass_positive_definite(self, unit_matrices):
        """All eigenvalues of M_e must be positive."""
        _, M_e, _, _ = unit_matrices
        eigvals = np.linalg.eigvalsh(M_e)
        assert np.all(eigvals > 0), f"M_e has non-positive eigenvalues: {eigvals}"

    def test_stiffness_positive_semidefinite(self, unit_matrices):
        """K_e must be positive semi-definite."""
        K_e, _, _, _ = unit_matrices
        eigvals = np.linalg.eigvalsh(K_e)
        assert np.all(eigvals >= -1e-11)

    def test_stiffness_exact_rank(self, unit_matrices):
        """K_e for CQUAD8 must have exactly 1 zero eigenvalue."""
        K_e, _, _, _ = unit_matrices
        eigvals = np.linalg.eigvalsh(K_e)
        n_zero = np.sum(np.abs(eigvals) < 1e-10)
        assert n_zero == 1, (
            f"K_e CQUAD8 should have 1 zero eigenvalue, found {n_zero}"
        )
