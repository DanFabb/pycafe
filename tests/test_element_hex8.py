"""
Element-level tests for the CHEXA8 trilinear hexahedral acoustic element.

Mirrors tests/test_element_matrices.py (CQUAD4/CQUAD8): shape function
identities, Jacobian, and the mathematical properties of K_e / M_e
(symmetry, null space, rank, positive definiteness, volume and scaling
laws), on both regular and distorted elements.
"""

import numpy as np
import pytest

from pycafe.build_matrices.element_hex8 import (
    gauss_rule_hex_2x2x2,
    hex8_shape,
    jacobian_3d,
    B_stiffness_3d,
    element_matrices_hex8,
)


# Helpers

NAT_COORDS = np.array([
    [-1, -1, -1],
    [ 1, -1, -1],
    [ 1,  1, -1],
    [-1,  1, -1],
    [-1, -1,  1],
    [ 1, -1,  1],
    [ 1,  1,  1],
    [-1,  1,  1],
], dtype=float)


def box_hex8(Lx=1.0, Ly=1.0, Lz=1.0):
    """Single HEXA8 element spanning [0,Lx]x[0,Ly]x[0,Lz], Gmsh ordering."""
    return 0.5 * (NAT_COORDS + 1.0) * np.array([Lx, Ly, Lz])


def distorted_hex8():
    """A valid but non-parallelepiped hexahedron."""
    x = box_hex8()
    x = x.copy()
    x[6] += [0.15, 0.10, 0.20]
    x[1] += [0.05, -0.02, 0.0]
    return x


# Shape functions

class TestHex8ShapeFunctions:

    def test_partition_of_unity(self):
        rng = np.random.default_rng(42)
        for _ in range(10):
            xi, eta, zeta = rng.uniform(-1, 1, 3)
            N, _, _, _ = hex8_shape(xi, eta, zeta)
            assert np.isclose(N.sum(), 1.0, atol=1e-14)

    def test_kronecker_delta_at_nodes(self):
        for i, (xi, eta, zeta) in enumerate(NAT_COORDS):
            N, _, _, _ = hex8_shape(xi, eta, zeta)
            expected = np.zeros(8)
            expected[i] = 1.0
            assert np.allclose(N, expected, atol=1e-14)

    def test_derivative_sums_zero(self):
        """Sum of shape function derivatives vanishes (constant reproduced)."""
        rng = np.random.default_rng(7)
        for _ in range(10):
            xi, eta, zeta = rng.uniform(-1, 1, 3)
            _, dxi, deta, dzeta = hex8_shape(xi, eta, zeta)
            assert np.isclose(dxi.sum(), 0.0, atol=1e-14)
            assert np.isclose(deta.sum(), 0.0, atol=1e-14)
            assert np.isclose(dzeta.sum(), 0.0, atol=1e-14)

    def test_linear_field_reproduction(self):
        """Trilinear shape functions reproduce a linear field exactly."""
        x_e = distorted_hex8()
        a, b, c, d = 0.3, 1.2, -0.7, 2.1
        p_nodes = a + b * x_e[:, 0] + c * x_e[:, 1] + d * x_e[:, 2]

        rng = np.random.default_rng(3)
        for _ in range(5):
            xi, eta, zeta = rng.uniform(-1, 1, 3)
            N, dxi, deta, dzeta = hex8_shape(xi, eta, zeta)
            x_p = N @ x_e
            p_interp = N @ p_nodes
            p_exact = a + b * x_p[0] + c * x_p[1] + d * x_p[2]
            assert np.isclose(p_interp, p_exact, atol=1e-12)


# Jacobian

class TestJacobian3D:

    def test_unit_cube_detJ(self):
        """[-1,1]^3 -> [0,1]^3 mapping has detJ = 1/8 everywhere."""
        x_e = box_hex8(1.0, 1.0, 1.0)
        _, dxi, deta, dzeta = hex8_shape(0.2, -0.3, 0.5)
        _, detJ, _ = jacobian_3d(dxi, deta, dzeta, x_e)
        assert np.isclose(detJ, 0.125, atol=1e-14)

    def test_box_detJ_scales_with_volume(self):
        x_e = box_hex8(2.0, 3.0, 0.5)
        _, dxi, deta, dzeta = hex8_shape(0.0, 0.0, 0.0)
        _, detJ, _ = jacobian_3d(dxi, deta, dzeta, x_e)
        assert np.isclose(detJ, 2.0 * 3.0 * 0.5 / 8.0, atol=1e-13)

    def test_inverted_element_raises(self):
        x_e = box_hex8()
        x_inv = x_e.copy()
        x_inv = np.vstack([x_e[4:], x_e[:4]])
        _, dxi, deta, dzeta = hex8_shape(0.0, 0.0, 0.0)
        with pytest.raises(ValueError, match="Jacobian"):
            jacobian_3d(dxi, deta, dzeta, x_inv)

    def test_gradient_of_linear_field(self):
        """B = invJ @ grad_nat reproduces the exact physical gradient."""
        x_e = distorted_hex8()
        b, c, d = 1.5, -0.4, 0.9
        p_nodes = b * x_e[:, 0] + c * x_e[:, 1] + d * x_e[:, 2]

        _, dxi, deta, dzeta = hex8_shape(0.3, -0.6, 0.1)
        _, _, invJ = jacobian_3d(dxi, deta, dzeta, x_e)
        B = B_stiffness_3d(dxi, deta, dzeta, invJ)
        grad = B @ p_nodes
        assert np.allclose(grad, [b, c, d], atol=1e-12)


# Element matrices

class TestHex8ElementMatrices:

    C0 = 343.0

    def test_shapes(self):
        K, M = element_matrices_hex8(box_hex8(), self.C0)
        assert K.shape == (8, 8)
        assert M.shape == (8, 8)

    def test_symmetry(self):
        for x_e in (box_hex8(), distorted_hex8()):
            K, M = element_matrices_hex8(x_e, self.C0)
            assert np.allclose(K, K.T, atol=1e-12)
            assert np.allclose(M, M.T, atol=1e-14)

    def test_stiffness_nullspace_constant_pressure(self):
        """K_e @ ones = 0 (constant pressure produces no 'force')."""
        for x_e in (box_hex8(2.0, 1.0, 0.5), distorted_hex8()):
            K, _ = element_matrices_hex8(x_e, self.C0)
            assert np.allclose(K @ np.ones(8), 0.0, atol=1e-12)

    def test_stiffness_rank(self):
        """rank(K_e) = 7 (only the constant mode is in the null space)."""
        for x_e in (box_hex8(), distorted_hex8()):
            K, _ = element_matrices_hex8(x_e, self.C0)
            assert np.linalg.matrix_rank(K, tol=1e-10) == 7

    def test_stiffness_positive_semidefinite(self):
        K, _ = element_matrices_hex8(distorted_hex8(), self.C0)
        eigvals = np.linalg.eigvalsh(K)
        assert np.all(eigvals > -1e-12)

    def test_mass_positive_definite(self):
        for x_e in (box_hex8(), distorted_hex8()):
            _, M = element_matrices_hex8(x_e, self.C0)
            eigvals = np.linalg.eigvalsh(M)
            assert np.all(eigvals > 0.0)

    def test_mass_total_volume(self):
        """sum(M_e) = V / c^2."""
        Lx, Ly, Lz = 2.0, 1.5, 0.4
        _, M = element_matrices_hex8(box_hex8(Lx, Ly, Lz), self.C0)
        V = Lx * Ly * Lz
        assert np.isclose(M.sum(), V / self.C0**2, rtol=1e-12)

    def test_mass_scales_inverse_c_squared(self):
        x_e = box_hex8()
        _, M1 = element_matrices_hex8(x_e, 100.0)
        _, M2 = element_matrices_hex8(x_e, 200.0)
        assert np.allclose(M1, 4.0 * M2, rtol=1e-12)

    def test_stiffness_independent_of_c(self):
        x_e = distorted_hex8()
        K1, _ = element_matrices_hex8(x_e, 100.0)
        K2, _ = element_matrices_hex8(x_e, 343.0)
        assert np.allclose(K1, K2, atol=1e-13)

    def test_mass_scales_with_volume(self):
        _, M1 = element_matrices_hex8(box_hex8(1, 1, 1), self.C0)
        _, M2 = element_matrices_hex8(box_hex8(2, 2, 2), self.C0)
        assert np.allclose(8.0 * M1, M2, rtol=1e-12)

    def test_patch_linear_pressure_energy(self):
        """
        For p = b*x + c*y + d*z, the stiffness energy is
        p^T K p = |grad p|^2 * V (B'B convention, no 1/rho).
        """
        Lx, Ly, Lz = 1.2, 0.8, 0.6
        x_e = box_hex8(Lx, Ly, Lz)
        b, c, d = 2.0, -1.0, 0.5
        p = b * x_e[:, 0] + c * x_e[:, 1] + d * x_e[:, 2]
        K, _ = element_matrices_hex8(x_e, self.C0)
        V = Lx * Ly * Lz
        assert np.isclose(p @ K @ p, (b**2 + c**2 + d**2) * V, rtol=1e-12)

    def test_bad_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            element_matrices_hex8(np.zeros((8, 2)), self.C0)
