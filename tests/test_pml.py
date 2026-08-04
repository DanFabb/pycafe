"""
Validation of the perfectly matched layer, in 2D and 3D.

The layer is checked three ways: the stretching factors against the
formulas they are defined by, the assembled operator against the
ordinary element matrices when the absorption is switched off, and a
duct terminated by the layer against the analytical travelling wave that
a non-reflecting termination must produce.
"""

import numpy as np
import pytest
import scipy.sparse.linalg as spla

from pycafe.build_matrices.assembly import assemble_KM
from pycafe.build_matrices.element_cquad4 import element_matrices_cquad4
from pycafe.build_matrices.element_hex8 import element_matrices_hex8
from pycafe.build_matrices.element_tetra4 import element_matrices_tetra4
from pycafe.build_matrices.pml import (
    CartesianPML,
    build_pml_operator,
)


C0 = 343.0
RHO = 1.204


# ---------------------------------------------------------------------------
# Meshes: a duct whose last stretch is the layer
# ---------------------------------------------------------------------------

def duct_2d(L_phys, L_pml, H, nx_phys, nx_pml, ny, quad8=False):
    x = np.concatenate([np.linspace(0.0, L_phys, nx_phys + 1),
                        np.linspace(L_phys, L_phys + L_pml, nx_pml + 1)[1:]])
    y = np.linspace(0.0, H, ny + 1)
    X, Y = np.meshgrid(x, y, indexing="ij")
    nodes = np.column_stack([X.ravel(), Y.ravel()])

    def nid(i, j):
        return i * (ny + 1) + j

    phys, layer = [], []
    for i in range(len(x) - 1):
        for j in range(ny):
            elem = [nid(i, j) + 1, nid(i + 1, j) + 1,
                    nid(i + 1, j + 1) + 1, nid(i, j + 1) + 1]
            (layer if x[i] >= L_phys - 1e-12 else phys).append(elem)

    return nodes, np.array(phys), np.array(layer), x, y


CORNERS = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
           (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
SPLIT = [(0, 1, 2, 6), (0, 1, 5, 6), (0, 3, 2, 6),
         (0, 3, 7, 6), (0, 4, 5, 6), (0, 4, 7, 6)]


def duct_3d(L_phys, L_pml, H, W, nx_phys, nx_pml, ny, nz, tetra=False):
    x = np.concatenate([np.linspace(0.0, L_phys, nx_phys + 1),
                        np.linspace(L_phys, L_phys + L_pml, nx_pml + 1)[1:]])
    y = np.linspace(0.0, H, ny + 1)
    z = np.linspace(0.0, W, nz + 1)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    nodes = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    def nid(i, j, k):
        return i * (ny + 1) * (nz + 1) + j * (nz + 1) + k

    phys, layer = [], []
    for i in range(len(x) - 1):
        for j in range(ny):
            for k in range(nz):
                v = [nid(i + a, j + b, k + c) for a, b, c in CORNERS]
                in_pml = x[i] >= L_phys - 1e-12
                if tetra:
                    for tet in SPLIT:
                        q = [v[m] for m in tet]
                        p = nodes[q]
                        if np.linalg.det(np.array([p[1] - p[0], p[2] - p[0],
                                                   p[3] - p[0]])) < 0.0:
                            q[2], q[3] = q[3], q[2]
                        (layer if in_pml else phys).append(
                            [m + 1 for m in q]
                        )
                else:
                    (layer if in_pml else phys).append([m + 1 for m in v])

    return nodes, np.array(phys), np.array(layer), x, y, z


def piston_load_2d(nodes, ny):
    """``int N dS`` on the x = 0 edge of the 2D duct."""
    n = nodes.shape[0]
    g = np.zeros(n)
    left = np.where(np.abs(nodes[:, 0]) < 1e-12)[0]
    left = left[np.argsort(nodes[left, 1])]
    for a, b in zip(left[:-1], left[1:]):
        length = nodes[b, 1] - nodes[a, 1]
        g[a] += length / 2.0
        g[b] += length / 2.0
    return g


def piston_load_3d(nodes, y, z, ny, nz):
    """``int N dS`` on the x = 0 face of the 3D duct."""
    n = nodes.shape[0]
    g = np.zeros(n)

    def nid(i, j, k):
        return i * (ny + 1) * (nz + 1) + j * (nz + 1) + k

    for j in range(ny):
        for k in range(nz):
            area = (y[j + 1] - y[j]) * (z[k + 1] - z[k])
            for m in (nid(0, j, k), nid(0, j + 1, k),
                      nid(0, j + 1, k + 1), nid(0, j, k + 1)):
                g[m] += area / 4.0
    return g


def travelling_wave_error(nodes, probe, p, k, v_piston=1.0):
    """
    Compare against ``p = rho c v exp(-i k x)`` and measure the ripple.

    The relative error mixes the reflection left by the layer with the
    numerical dispersion of the mesh; the standing-wave ratio isolates
    the reflection, since a purely dispersive error keeps the amplitude
    flat.
    """
    xs = nodes[probe, 0]
    exact = RHO * C0 * v_piston * np.exp(-1j * k * xs)
    rel = np.abs(p[probe] - exact) / np.abs(exact)
    amp = np.abs(p[probe])
    swr = (amp.max() - amp.min()) / (amp.max() + amp.min())
    return rel.max(), swr


# ---------------------------------------------------------------------------
# The absorption profile
# ---------------------------------------------------------------------------

class TestCartesianPML:

    def test_no_absorption_inside_the_physical_box(self):
        pml = CartesianPML(inner_min=[0.0, 0.0], inner_max=[1.0, 1.0],
                           thickness=[0.2, 0.2], sigma0=[50.0, 50.0])
        assert np.allclose(pml.sigma([[0.5, 0.5], [0.0, 1.0]]), 0.0)

    def test_profile_is_the_power_law(self):
        pml = CartesianPML(inner_min=[0.0, 0.0], inner_max=[1.0, 1.0],
                           thickness=[0.4, 0.4], sigma0=[10.0, 10.0],
                           order=2)
        # a quarter of the way into the layer along x only
        sigma = pml.sigma([1.1, 0.5])[0]
        assert sigma[0] == pytest.approx(10.0 * 0.25 ** 2)
        assert sigma[1] == 0.0
        # the outer face reaches sigma0
        assert pml.sigma([1.4, 0.5])[0][0] == pytest.approx(10.0)

    def test_order_one_is_linear(self):
        pml = CartesianPML(inner_min=[0.0, 0.0], inner_max=[1.0, 1.0],
                           thickness=[1.0, 0.0], sigma0=[4.0, 0.0], order=1)
        assert pml.sigma([1.5, 0.5])[0][0] == pytest.approx(2.0)

    def test_zero_thickness_axis_is_not_stretched(self):
        pml = CartesianPML(inner_min=[0.0, 0.0], inner_max=[1.0, 1.0],
                           thickness=[0.3, 0.0], sigma0=[10.0, 10.0])
        assert pml.sigma([1.3, 5.0])[0][1] == 0.0

    def test_beyond_the_outer_face_is_clipped(self):
        pml = CartesianPML(inner_min=[0.0, 0.0], inner_max=[1.0, 1.0],
                           thickness=[0.2, 0.2], sigma0=[7.0, 7.0])
        assert pml.sigma([100.0, 0.5])[0][0] == pytest.approx(7.0)

    @pytest.mark.parametrize("kwargs, message", [
        (dict(inner_min=[0.0], inner_max=[1.0], thickness=1.0, sigma0=1.0),
         "2 or 3 dimensions"),
        (dict(inner_min=[0.0, 0.0], inner_max=[0.0, 1.0], thickness=1.0,
              sigma0=1.0), "must exceed"),
        (dict(inner_min=[0.0, 0.0], inner_max=[1.0, 1.0], thickness=-1.0,
              sigma0=1.0), "Negative PML thickness"),
        (dict(inner_min=[0.0, 0.0], inner_max=[1.0, 1.0], thickness=1.0,
              sigma0=-1.0), "Negative sigma0"),
    ])
    def test_bad_input_is_rejected(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            CartesianPML(**kwargs)


# ---------------------------------------------------------------------------
# The stretching factors
# ---------------------------------------------------------------------------

class TestStretching:

    def _operator(self, sigma0, dim=2):
        if dim == 2:
            nodes, _, layer, _, _ = duct_2d(0.0, 1.0, 1.0, 0, 1, 1)
            elements = {"Quadrilateral 4": layer}
            pml = CartesianPML(inner_min=[-1e9, -1e9], inner_max=[0.0, 1e9],
                               thickness=[1.0, 0.0], sigma0=[sigma0, 0.0],
                               order=0)
        else:
            nodes, _, layer, _, _, _ = duct_3d(0.0, 1.0, 1.0, 1.0,
                                               0, 1, 1, 1)
            elements = {"Hexahedron 8": layer}
            pml = CartesianPML(inner_min=[-1e9, -1e9, -1e9],
                               inner_max=[0.0, 1e9, 1e9],
                               thickness=[1.0, 0.0, 0.0],
                               sigma0=[sigma0, 0.0, 0.0], order=0)
        return build_pml_operator(nodes, elements, C0, pml)

    def test_gamma_definition(self):
        op = self._operator(30.0)
        omega = 2 * np.pi * 400.0
        k = omega / C0
        gamma, lambdas, mass = op.stretching(omega)

        assert np.allclose(gamma, 1.0 - 1j * op.sigma_e / k)
        # Lambda_j = prod(gamma) / gamma_j^2
        prod = np.prod(gamma, axis=1)
        assert np.allclose(lambdas, prod[:, None] / gamma ** 2)
        assert np.allclose(mass, -(k ** 2) * prod)

    def test_two_dimensional_ratio(self):
        """In 2D, Lambda_x is gamma_y / gamma_x."""
        op = self._operator(30.0)
        gamma, lambdas, _ = op.stretching(2 * np.pi * 400.0)
        assert np.allclose(lambdas[:, 0], gamma[:, 1] / gamma[:, 0])
        assert np.allclose(lambdas[:, 1], gamma[:, 0] / gamma[:, 1])

    def test_three_dimensional_ratio(self):
        op = self._operator(30.0, dim=3)
        gamma, lambdas, _ = op.stretching(2 * np.pi * 400.0)
        assert np.allclose(
            lambdas[:, 0], gamma[:, 1] * gamma[:, 2] / gamma[:, 0]
        )

    def test_the_stretching_is_not_conjugated(self):
        """
        The factor must survive in the imaginary part: conjugating one
        of the two gradients would leave a real, non-absorbing operator.
        """
        op = self._operator(30.0)
        A = op.matrix(2 * np.pi * 400.0)
        assert np.abs(A.toarray().imag).max() > 1e-6

    def test_complex_symmetric_not_hermitian(self):
        op = self._operator(30.0)
        A = op.matrix(2 * np.pi * 400.0).toarray()
        assert np.allclose(A, A.T)
        assert not np.allclose(A, A.conj().T)

    def test_zero_frequency_is_refused(self):
        op = self._operator(30.0)
        with pytest.raises(ValueError, match="positive frequency"):
            op.matrix(0.0)


# ---------------------------------------------------------------------------
# Consistency: no absorption means the ordinary element matrices
# ---------------------------------------------------------------------------

class TestNoAbsorptionLimit:
    """
    With ``sigma = 0`` every gamma is 1, so the layer must reproduce
    ``K - omega^2 M`` of the standard kernels exactly. This is what ties
    the PML quadrature to the rest of the code.
    """

    OMEGA = 2 * np.pi * 300.0

    def _check(self, nodes, conn, key, kernel):
        pml = CartesianPML(
            inner_min=np.full(nodes.shape[1], -1e9),
            inner_max=np.full(nodes.shape[1], 1e9),
            thickness=1.0, sigma0=0.0,
        )
        op = build_pml_operator(nodes, {key: conn}, C0, pml)
        assert np.allclose(op.sigma_e, 0.0)

        K, M, _ = assemble_KM(nodes, conn - 1, kernel, C0)
        expected = (K - self.OMEGA ** 2 * M).toarray()
        got = op.matrix(self.OMEGA).toarray()

        assert np.allclose(got.imag, 0.0, atol=1e-12)
        assert np.allclose(got.real, expected, rtol=1e-10, atol=1e-12)

    def test_cquad4(self):
        nodes, phys, _, _, _ = duct_2d(1.0, 0.2, 0.3, 4, 1, 3)
        self._check(nodes, phys, "Quadrilateral 4", element_matrices_cquad4)

    def test_hexa8(self):
        nodes, phys, _, _, _, _ = duct_3d(1.0, 0.2, 0.3, 0.25, 3, 1, 2, 2)
        self._check(nodes, phys, "Hexahedron 8", element_matrices_hex8)

    def test_tetra4(self):
        nodes, phys, _, _, _, _ = duct_3d(1.0, 0.2, 0.3, 0.25, 3, 1, 2, 2,
                                          tetra=True)
        self._check(nodes, phys, "Tetrahedron 4", element_matrices_tetra4)


# ---------------------------------------------------------------------------
# The layer does its job: a duct that does not reflect
# ---------------------------------------------------------------------------

class TestDuctTermination:

    SIGMA0 = 80.0

    def _solve_2d(self, freq, nx_phys=100, nx_pml=40, sigma0=None):
        L_phys, L_pml, H = 1.0, 0.4, 0.2
        ny = 4
        nodes, phys, layer, _, y = duct_2d(L_phys, L_pml, H,
                                           nx_phys, nx_pml, ny)
        K, M, _ = assemble_KM(nodes, phys - 1, element_matrices_cquad4, C0)
        pml = CartesianPML(inner_min=[-1e9, -1e9], inner_max=[L_phys, 1e9],
                           thickness=[L_pml, 0.0],
                           sigma0=[self.SIGMA0 if sigma0 is None else sigma0,
                                   0.0],
                           order=2)
        op = build_pml_operator(nodes, {"Quadrilateral 4": layer}, C0, pml)

        omega = 2 * np.pi * freq
        A = (K - omega ** 2 * M).astype(complex) + op.matrix(omega)
        rhs = -1j * RHO * omega * (-1.0) * piston_load_2d(nodes, ny)
        p = spla.spsolve(A.tocsc(), rhs)

        probe = np.where(
            (np.abs(nodes[:, 1] - y[ny // 2]) < 1e-12)
            & (nodes[:, 0] <= L_phys + 1e-12)
        )[0]
        probe = probe[np.argsort(nodes[probe, 0])]
        return travelling_wave_error(nodes, probe, p, omega / C0)

    @pytest.mark.parametrize("freq", [200.0, 500.0])
    def test_2d_duct_is_non_reflecting(self, freq):
        rel, swr = self._solve_2d(freq)
        assert swr < 5e-3, f"standing wave ratio {swr:.2e} at {freq} Hz"
        assert rel < 0.02, f"error {rel*100:.2f}% at {freq} Hz"

    def test_absorption_off_leaves_a_standing_wave(self):
        """
        The counter-test: with sigma0 = 0 the layer is ordinary fluid,
        the duct end is rigid, and the field is a standing wave. Without
        this, the test above would also pass on a layer that does
        nothing but happens to sit at an anti-node.
        """
        _, swr_off = self._solve_2d(500.0, sigma0=0.0)
        _, swr_on = self._solve_2d(500.0)
        assert swr_off > 0.5
        assert swr_on < swr_off / 100.0

    def _solve_3d(self, freq, tetra):
        L_phys, L_pml, H, W = 1.0, 0.4, 0.15, 0.15
        nx_phys, nx_pml, ny, nz = 60, 24, 2, 2
        nodes, phys, layer, _, y, z = duct_3d(L_phys, L_pml, H, W, nx_phys,
                                              nx_pml, ny, nz, tetra=tetra)
        key = "Tetrahedron 4" if tetra else "Hexahedron 8"
        kernel = element_matrices_tetra4 if tetra else element_matrices_hex8

        K, M, _ = assemble_KM(nodes, phys - 1, kernel, C0)
        pml = CartesianPML(inner_min=[-1e9, -1e9, -1e9],
                           inner_max=[L_phys, 1e9, 1e9],
                           thickness=[L_pml, 0.0, 0.0],
                           sigma0=[self.SIGMA0, 0.0, 0.0], order=2)
        op = build_pml_operator(nodes, {key: layer}, C0, pml)

        omega = 2 * np.pi * freq
        A = (K - omega ** 2 * M).astype(complex) + op.matrix(omega)
        rhs = -1j * RHO * omega * (-1.0) * piston_load_3d(nodes, y, z, ny, nz)
        p = spla.spsolve(A.tocsc(), rhs)

        probe = np.where(
            (np.abs(nodes[:, 1] - y[1]) < 1e-12)
            & (np.abs(nodes[:, 2] - z[1]) < 1e-12)
            & (nodes[:, 0] <= L_phys + 1e-12)
        )[0]
        probe = probe[np.argsort(nodes[probe, 0])]
        return travelling_wave_error(nodes, probe, p, omega / C0)

    @pytest.mark.parametrize("tetra", [False, True], ids=["hexa8", "tetra4"])
    def test_3d_duct_is_non_reflecting(self, tetra):
        rel, swr = self._solve_3d(300.0, tetra)
        assert swr < 5e-3, f"standing wave ratio {swr:.2e}"
        assert rel < 0.02, f"error {rel*100:.2f}%"


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------

class TestBuildErrors:

    def test_unknown_element_type(self):
        nodes, _, layer, _, _ = duct_2d(0.0, 1.0, 1.0, 0, 1, 1)
        pml = CartesianPML(inner_min=[-1e9, -1e9], inner_max=[0.0, 1e9],
                           thickness=[1.0, 0.0], sigma0=[1.0, 0.0])
        with pytest.raises(ValueError, match="No PML quadrature"):
            build_pml_operator(nodes, {"Prism 6": layer}, C0, pml)

    def test_no_elements(self):
        nodes, _, _, _, _ = duct_2d(0.0, 1.0, 1.0, 0, 1, 1)
        pml = CartesianPML(inner_min=[-1e9, -1e9], inner_max=[0.0, 1e9],
                           thickness=[1.0, 0.0], sigma0=[1.0, 0.0])
        with pytest.raises(ValueError, match="No PML elements"):
            build_pml_operator(nodes, {}, C0, pml)

    def test_dimension_mismatch(self):
        nodes, _, layer, _, _, _ = duct_3d(0.0, 1.0, 1.0, 1.0, 0, 1, 1, 1)
        pml = CartesianPML(inner_min=[-1e9, -1e9], inner_max=[0.0, 1e9],
                           thickness=[1.0, 0.0], sigma0=[1.0, 0.0])
        with pytest.raises(ValueError, match="but the CartesianPML is"):
            build_pml_operator(nodes, {"Hexahedron 8": layer}, C0, pml)

    def test_mixed_dimensions(self):
        nodes, _, layer2d, _, _ = duct_2d(0.0, 1.0, 1.0, 0, 1, 1)
        nodes3, _, layer3d, _, _, _ = duct_3d(0.0, 1.0, 1.0, 1.0, 0, 1, 1, 1)
        pml = CartesianPML(inner_min=[-1e9, -1e9], inner_max=[0.0, 1e9],
                           thickness=[1.0, 0.0], sigma0=[1.0, 0.0])
        with pytest.raises(ValueError, match="mix"):
            build_pml_operator(nodes3, {"Quadrilateral 4": layer2d,
                                        "Hexahedron 8": layer3d}, C0, pml)

    def test_wrong_node_count(self):
        nodes, _, layer, _, _ = duct_2d(0.0, 1.0, 1.0, 0, 1, 1)
        pml = CartesianPML(inner_min=[-1e9, -1e9], inner_max=[0.0, 1e9],
                           thickness=[1.0, 0.0], sigma0=[1.0, 0.0])
        with pytest.raises(ValueError, match="8 nodes per element"):
            build_pml_operator(nodes, {"Quadrilateral 8": layer}, C0, pml)
