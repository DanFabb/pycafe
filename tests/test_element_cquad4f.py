"""
Tests for the CQUAD4F structural shell element.

Element level: symmetry, rigid-body null space, mass, element frame.
Benchmark level: two single-element cases against the MSC.Nastran
reference (displacements from the .f06, element stiffness matrix from
the DMIG entries of the .pch), and a simply supported plate against the
analytical Kirchhoff solution.
"""

import os

import numpy as np
import pytest
from scipy.sparse.linalg import eigsh

from pycafe.build_matrices.assembly import assemble_KM
from pycafe.build_matrices.element_cquad4f import (
    DRILLING_DOF,
    MEMBRANE_DOF,
    PLATE_DOF,
    element_frame,
    element_matrices_cquad4f,
)

# reference model properties (mm, N, MPa, t/mm^3)
E, NU, T, RHO = 200e3, 0.3, 0.1, 7.8e-9

SQUARE = np.array([[0., 0., 0.], [2., 0., 0.], [2., 2., 0.], [0., 2., 0.]])
SKEW = np.array([[0., 0., 0.], [2., 0., 0.], [1.5, 2., 0.], [0., 2., 0.]])

# MSC.Nastran .f06, node 4 loaded with Fz = 1, nodes 1-2-3 clamped
NASTRAN_SQUARE = np.array([0.050742708, 0.047409551, 0.047409551])
NASTRAN_SKEW = np.array([0.03734316, 0.03541586, 0.04646765])

DATA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "examples", "data")
PCH_SQUARE = os.path.join(DATA, "nastran_cquad4_square.pch")


def corner_response(K, node=3):
    dof = np.arange(6 * node, 6 * node + 6)
    f = np.zeros(6)
    f[2] = 1.0
    u = np.linalg.solve(K[np.ix_(dof, dof)], f)
    return np.array([u[2], u[3], u[4]])


def rigid_body_modes(x_e):
    modes = []
    for d in range(3):
        u = np.zeros((4, 6))
        u[:, d] = 1.0
        modes.append(u.ravel())
    x0 = x_e.mean(axis=0)
    for axis in range(3):
        theta = np.zeros(3)
        theta[axis] = 1.0
        u = np.zeros((4, 6))
        u[:, 0:3] = np.cross(theta, x_e - x0)
        u[:, 3:6] = theta
        modes.append(u.ravel())
    return modes


def read_dmig(path, name="KAAX", n_nodes=4, dofs=6):
    """Read a DMIG matrix (large field) from a Nastran .pch file."""
    K = np.zeros((n_nodes * dofs, n_nodes * dofs))
    col = None
    with open(path) as fh:
        for line in fh:
            if line.startswith("DMIG*") and name in line:
                f = line[8:].split()
                col = (int(f[1]) - 1) * dofs + int(f[2]) - 1
            elif line.startswith("*") and col is not None:
                gi, ci = int(line[8:24]), int(line[24:40])
                K[(gi - 1) * dofs + ci - 1, col] = float(line[40:56])
            elif line.startswith("DMIG "):
                col = None
    return K + np.triu(K, 1).T          # only the upper triangle is stored


class TestElement:

    def test_shapes_and_symmetry(self):
        K, M = element_matrices_cquad4f(SKEW, T, RHO, E, NU)
        assert K.shape == (24, 24)
        assert M.shape == (24, 24)
        assert np.allclose(K, K.T, atol=1e-9 * abs(K).max())
        assert np.allclose(M, M.T, atol=1e-18)

    def test_bad_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            element_matrices_cquad4f(np.zeros((3, 3)), T, RHO, E, NU)

    @pytest.mark.parametrize("x_e", [
        SQUARE,
        SKEW,
        np.array([[0., 0., 0.], [3., 0., 0.], [3., 1., 0.], [0., 1., 0.]]),
        np.array([[0., 0., 0.], [2., 0., 0.], [2.5, 2., 0.], [0.5, 2., 0.]]),
    ])
    def test_rigid_body_modes(self, x_e):
        """
        Translations and out-of-plane rotations are in the null space of K
        for any quadrilateral. The rotation about the normal is excluded:
        the drilling penalty is assembled as in the Nastran element
        matrix (drilling-drilling and drilling-membrane blocks only), so
        it leaves a residual of order 1e-4 there.
        """
        K, _ = element_matrices_cquad4f(x_e, T, RHO, E, NU)
        scale = abs(K).max()
        for i, u in enumerate(rigid_body_modes(x_e)[:5]):
            assert np.abs(K @ u).max() < 1e-10 * scale, f"rigid mode {i}"

    def test_mass_total(self):
        _, M = element_matrices_cquad4f(SQUARE, T, RHO, E, NU)
        for d in range(3):
            u = np.zeros((4, 6))
            u[:, d] = 1.0
            u = u.ravel()
            assert np.isclose(u @ M @ u, RHO * T * 4.0, rtol=1e-12)

    def test_non_structural_mass(self):
        _, M0 = element_matrices_cquad4f(SQUARE, T, RHO, E, NU)
        _, M1 = element_matrices_cquad4f(SQUARE, T, RHO, E, NU, nsm=1e-9)
        u = np.zeros((4, 6))
        u[:, 2] = 1.0
        u = u.ravel()
        assert np.isclose(u @ M1 @ u - u @ M0 @ u, 1e-9 * 4.0, rtol=1e-12)

    def test_no_rotary_inertia(self):
        _, M = element_matrices_cquad4f(SQUARE, T, RHO, E, NU)
        for n in range(4):
            for d in range(3, 6):
                assert np.allclose(M[6 * n + d, :], 0.0, atol=1e-18)

    def test_frame_square_is_identity(self):
        T_e0, x0 = element_frame(SQUARE)
        assert np.allclose(T_e0, np.eye(3), atol=1e-12)
        assert np.allclose(x0, [1.0, 1.0, 0.0], atol=1e-12)

    def test_frame_orthonormal_and_origin_on_diagonals(self):
        T_e0, x0 = element_frame(SKEW)
        assert np.allclose(T_e0 @ T_e0.T, np.eye(3), atol=1e-12)
        for p, q in ((SKEW[0], SKEW[2]), (SKEW[1], SKEW[3])):
            d = q - p
            assert np.linalg.norm(np.cross(d, x0 - p)) < 1e-12

    def test_frame_invariance(self):
        """Rotating the element in space leaves the eigenvalues of K, M."""
        K0, M0 = element_matrices_cquad4f(SKEW, T, RHO, E, NU)
        a, b = 0.7, -0.4
        Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0],
                       [0, 0, 1]])
        Rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)],
                       [0, np.sin(b), np.cos(b)]])
        Q = Rx @ Rz
        K1, M1 = element_matrices_cquad4f(
            SKEW @ Q.T + np.array([0.3, -0.2, 1.1]), T, RHO, E, NU)
        assert np.allclose(np.sort(np.linalg.eigvalsh(K0)),
                           np.sort(np.linalg.eigvalsh(K1)),
                           rtol=1e-8, atol=1e-8 * abs(K0).max())
        assert np.allclose(np.sort(np.linalg.eigvalsh(M0)),
                           np.sort(np.linalg.eigvalsh(M1)), atol=1e-18)

    def test_thin_limit_is_thickness_independent(self):
        """
        With rbf_mindlin=False the element is a thin-plate element: the
        plate block scales exactly with t^3. Switching the option on
        introduces the shear compliance and breaks that scaling.
        """
        Ka, _ = element_matrices_cquad4f(SQUARE, 0.1, RHO, E, NU)
        Kb, _ = element_matrices_cquad4f(SQUARE, 0.2, RHO, E, NU)
        Pa = Ka[np.ix_(PLATE_DOF, PLATE_DOF)]
        Pb = Kb[np.ix_(PLATE_DOF, PLATE_DOF)]
        assert np.allclose(Pb, 8.0 * Pa, rtol=1e-10)

        Kc, _ = element_matrices_cquad4f(SQUARE, 0.2, RHO, E, NU,
                                         rbf_mindlin=True)
        Pc = Kc[np.ix_(PLATE_DOF, PLATE_DOF)]
        assert not np.allclose(Pc, 8.0 * Pa, rtol=1e-3)
        assert np.abs(Pc).max() < np.abs(Pb).max()      # shear softens


class TestSingleElementBenchmark:
    """Single-element test cases, MSC.Nastran reference."""

    def test_square_plate_vs_nastran(self):
        K, _ = element_matrices_cquad4f(SQUARE, T, RHO, E, NU, epsilon=0.04)
        u = corner_response(K)
        err = np.abs(u - NASTRAN_SQUARE) / NASTRAN_SQUARE
        assert err[0] < 0.01, f"T3 {u[0]} vs {NASTRAN_SQUARE[0]}"
        assert err.max() < 0.02

    def test_square_plate_preserves_diagonal_symmetry(self):
        """The model is symmetric about the 2-4 diagonal, so R1 = R2."""
        K, _ = element_matrices_cquad4f(SQUARE, T, RHO, E, NU)
        u = corner_response(K)
        assert np.isclose(u[1], u[2], rtol=1e-10)

    def test_skew_plate_vs_nastran(self):
        K, _ = element_matrices_cquad4f(SKEW, T, RHO, E, NU, epsilon=0.04)
        err = (np.abs(corner_response(K) - NASTRAN_SKEW) / NASTRAN_SKEW).mean()
        assert err < 0.10


class TestVsNastranStiffness:
    """Term-by-term comparison with the Nastran element stiffness matrix."""

    @pytest.fixture
    def K_nastran(self):
        if not os.path.exists(PCH_SQUARE):
            pytest.skip("Nastran .pch reference not available")
        return read_dmig(PCH_SQUARE)

    def test_reference_is_sane(self, K_nastran):
        assert np.allclose(K_nastran, K_nastran.T, atol=0)
        scale = abs(K_nastran).max()
        for u in rigid_body_modes(SQUARE):
            assert np.abs(K_nastran @ u).max() < 1e-9 * scale

    def test_drilling_block_is_exact(self, K_nastran):
        K, _ = element_matrices_cquad4f(SQUARE, T, RHO, E, NU)
        assert np.allclose(K[np.ix_(DRILLING_DOF, DRILLING_DOF)],
                           K_nastran[np.ix_(DRILLING_DOF, DRILLING_DOF)],
                           rtol=1e-8, atol=1e-9)
        assert np.allclose(K[np.ix_(DRILLING_DOF, MEMBRANE_DOF)],
                           K_nastran[np.ix_(DRILLING_DOF, MEMBRANE_DOF)],
                           rtol=1e-8, atol=1e-9)

    def test_membrane_block_within_2_percent(self, K_nastran):
        K, _ = element_matrices_cquad4f(SQUARE, T, RHO, E, NU)
        A = K[np.ix_(MEMBRANE_DOF, MEMBRANE_DOF)]
        B = K_nastran[np.ix_(MEMBRANE_DOF, MEMBRANE_DOF)]
        assert np.abs(A - B).max() / np.abs(B).max() < 0.02

    def test_full_matrix_within_2_percent(self, K_nastran):
        K, _ = element_matrices_cquad4f(SQUARE, T, RHO, E, NU, epsilon=0.04)
        err = np.abs(K - K_nastran).max() / np.abs(K_nastran).max()
        assert err < 0.02, f"max deviation {err*100:.2f} %"

    def test_plate_block_with_calibrated_epsilon(self, K_nastran):
        """With the calibrated epsilon the plate block matches to 0.2 %."""
        K, _ = element_matrices_cquad4f(SQUARE, T, RHO, E, NU, epsilon=0.0336)
        B = K[np.ix_(PLATE_DOF, PLATE_DOF)]
        BN = K_nastran[np.ix_(PLATE_DOF, PLATE_DOF)]
        assert np.abs(B - BN).max() / np.abs(BN).max() < 0.002


class TestSimplySupportedPlate:
    """Simply supported square plate against the Kirchhoff solution."""

    L, t, rho, e, nu = 1.0, 0.01, 7800.0, 210e9, 0.3

    def _mesh(self, n):
        v = np.linspace(0.0, self.L, n + 1)
        X, Y = np.meshgrid(v, v, indexing="ij")
        nodes = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])
        conn = [[i * (n + 1) + j, (i + 1) * (n + 1) + j,
                 (i + 1) * (n + 1) + j + 1, i * (n + 1) + j + 1]
                for i in range(n) for j in range(n)]
        return nodes, np.array(conn, dtype=int)

    def _f11(self, n):
        nodes, conn = self._mesh(n)
        K, M, _ = assemble_KM(nodes, conn, element_matrices_cquad4f,
                              dofs_per_node=6,
                              kernel_args=(self.t, self.rho, self.e, self.nu))
        nn = nodes.shape[0]
        fixed = set()
        for k in range(nn):
            xk, yk = nodes[k, :2]
            if (np.isclose(xk, 0) or np.isclose(xk, self.L)
                    or np.isclose(yk, 0) or np.isclose(yk, self.L)):
                fixed.update(6 * k + d for d in (0, 1, 2))
        free = np.array(sorted(set(range(6 * nn)) - fixed), dtype=int)
        vals, _ = eigsh(K[np.ix_(free, free)].tocsc(), k=1,
                        M=M[np.ix_(free, free)].tocsc(), sigma=0.0, which="LM")
        return np.sqrt(abs(vals[0])) / (2 * np.pi)

    def kirchhoff_f11(self):
        D = self.e * self.t**3 / (12 * (1 - self.nu**2))
        return np.pi * np.sqrt(D / (self.rho * self.t)) / self.L**2

    def test_fundamental_frequency(self):
        f_ex = self.kirchhoff_f11()
        assert abs(self._f11(12) - f_ex) / f_ex < 0.01

    def test_convergence(self):
        f_ex = self.kirchhoff_f11()
        errs = [abs(self._f11(n) - f_ex) / f_ex for n in (6, 12)]
        assert errs[1] < errs[0]
