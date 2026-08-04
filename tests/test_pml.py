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
    CylindricalPML,
    PowerProfile,
    RadialPML,
    build_pml_operator,
    identify_layer_shape,
    pml_from_groups,
)


def cartesian(inner_min, inner_max, thickness, sigma0, order=2):
    """CartesianPML from per-axis thickness and sigma0, as the tests read."""
    thickness = np.broadcast_to(np.asarray(thickness, float),
                               (len(inner_min),))
    sigma0 = np.broadcast_to(np.asarray(sigma0, float), (len(inner_min),))
    return CartesianPML(
        inner_min=inner_min, inner_max=inner_max,
        profiles=tuple(PowerProfile(sigma0=s, thickness=t, order=order)
                       for s, t in zip(sigma0, thickness)),
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
        pml = cartesian(inner_min=[0.0, 0.0], inner_max=[1.0, 1.0],
                           thickness=[0.2, 0.2], sigma0=[50.0, 50.0])
        assert np.allclose(pml.sigma([[0.5, 0.5], [0.0, 1.0]]), 0.0)

    def test_profile_is_the_power_law(self):
        pml = cartesian(inner_min=[0.0, 0.0], inner_max=[1.0, 1.0],
                           thickness=[0.4, 0.4], sigma0=[10.0, 10.0],
                           order=2)
        # a quarter of the way into the layer along x only
        sigma = pml.sigma([1.1, 0.5])[0]
        assert sigma[0] == pytest.approx(10.0 * 0.25 ** 2)
        assert sigma[1] == 0.0
        # the outer face reaches sigma0
        assert pml.sigma([1.4, 0.5])[0][0] == pytest.approx(10.0)

    def test_order_one_is_linear(self):
        pml = cartesian(inner_min=[0.0, 0.0], inner_max=[1.0, 1.0],
                           thickness=[1.0, 0.0], sigma0=[4.0, 0.0], order=1)
        assert pml.sigma([1.5, 0.5])[0][0] == pytest.approx(2.0)

    def test_zero_thickness_axis_is_not_stretched(self):
        pml = cartesian(inner_min=[0.0, 0.0], inner_max=[1.0, 1.0],
                           thickness=[0.3, 0.0], sigma0=[10.0, 10.0])
        assert pml.sigma([1.3, 5.0])[0][1] == 0.0

    def test_beyond_the_outer_face_is_clipped(self):
        pml = cartesian(inner_min=[0.0, 0.0], inner_max=[1.0, 1.0],
                           thickness=[0.2, 0.2], sigma0=[7.0, 7.0])
        assert pml.sigma([100.0, 0.5])[0][0] == pytest.approx(7.0)

    @pytest.mark.parametrize("kwargs, message", [
        (dict(inner_min=[0.0], inner_max=[1.0], thickness=1.0, sigma0=1.0),
         "2 or 3 dimensions"),
        (dict(inner_min=[0.0, 0.0], inner_max=[0.0, 1.0], thickness=1.0,
              sigma0=1.0), "must exceed"),
        (dict(inner_min=[0.0, 0.0], inner_max=[1.0, 1.0], thickness=-1.0,
              sigma0=1.0), "Negative thickness"),
        (dict(inner_min=[0.0, 0.0], inner_max=[1.0, 1.0], thickness=1.0,
              sigma0=-1.0), "Negative sigma0"),
    ])
    def test_bad_input_is_rejected(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            cartesian(**kwargs)

    def test_one_profile_per_axis(self):
        with pytest.raises(ValueError, match="2 axes but 1 profiles"):
            CartesianPML(inner_min=[0.0, 0.0], inner_max=[1.0, 1.0],
                         profiles=(PowerProfile(1.0, 1.0),))

    def test_sigma0_from_a_target_reflection(self):
        """
        ``exp(-2 sigma0 L / (n+1))`` is the round trip, so inverting it
        must give back the reflection asked for.
        """
        profile = PowerProfile.for_reflection(0.5, 1e-4, order=2)
        round_trip = np.exp(-2.0 * profile.sigma0 * profile.thickness
                            / (profile.order + 1))
        assert round_trip == pytest.approx(1e-4)

    def test_profile_integral(self):
        profile = PowerProfile(sigma0=6.0, thickness=0.5, order=2)
        # int_0^L sigma0 (d/L)^n dd = sigma0 L / (n+1)
        assert profile.integral(0.5) == pytest.approx(6.0 * 0.5 / 3.0)
        assert profile.integral(0.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# The stretching factors
# ---------------------------------------------------------------------------

class TestStretching:

    def _operator(self, sigma0, dim=2):
        if dim == 2:
            nodes, _, layer, _, _ = duct_2d(0.0, 1.0, 1.0, 0, 1, 1)
            elements = {"Quadrilateral 4": layer}
            pml = cartesian(inner_min=[-1e9, -1e9], inner_max=[0.0, 1e9],
                               thickness=[1.0, 0.0], sigma0=[sigma0, 0.0],
                               order=0)
        else:
            nodes, _, layer, _, _, _ = duct_3d(0.0, 1.0, 1.0, 1.0,
                                               0, 1, 1, 1)
            elements = {"Hexahedron 8": layer}
            pml = cartesian(inner_min=[-1e9, -1e9, -1e9],
                               inner_max=[0.0, 1e9, 1e9],
                               thickness=[1.0, 0.0, 0.0],
                               sigma0=[sigma0, 0.0, 0.0], order=0)
        return build_pml_operator(nodes, elements, C0, pml)

    @staticmethod
    def _gamma(op, omega):
        return 1.0 - 1j * op.sigma_e / (omega / C0)

    def test_cartesian_coefficients(self):
        op = self._operator(30.0)
        omega = 2 * np.pi * 400.0
        k = omega / C0
        gamma = self._gamma(op, omega)
        lambdas, mu = op.coefficients(omega)

        prod = np.prod(gamma, axis=1)
        assert np.allclose(lambdas, prod[:, None] / gamma ** 2)
        assert np.allclose(mu, -(k ** 2) * prod)

    def test_two_dimensional_ratio(self):
        """In 2D, Lambda_x is gamma_y / gamma_x."""
        op = self._operator(30.0)
        omega = 2 * np.pi * 400.0
        gamma = self._gamma(op, omega)
        lambdas, _ = op.coefficients(omega)
        assert np.allclose(lambdas[:, 0], gamma[:, 1] / gamma[:, 0])
        assert np.allclose(lambdas[:, 1], gamma[:, 0] / gamma[:, 1])

    def test_three_dimensional_ratio(self):
        op = self._operator(30.0, dim=3)
        omega = 2 * np.pi * 400.0
        gamma = self._gamma(op, omega)
        lambdas, _ = op.coefficients(omega)
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
        pml = cartesian(
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
        pml = cartesian(inner_min=[-1e9, -1e9], inner_max=[L_phys, 1e9],
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
        pml = cartesian(inner_min=[-1e9, -1e9, -1e9],
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
        pml = cartesian(inner_min=[-1e9, -1e9], inner_max=[0.0, 1e9],
                           thickness=[1.0, 0.0], sigma0=[1.0, 0.0])
        with pytest.raises(ValueError, match="No PML quadrature"):
            build_pml_operator(nodes, {"Prism 6": layer}, C0, pml)

    def test_no_elements(self):
        nodes, _, _, _, _ = duct_2d(0.0, 1.0, 1.0, 0, 1, 1)
        pml = cartesian(inner_min=[-1e9, -1e9], inner_max=[0.0, 1e9],
                           thickness=[1.0, 0.0], sigma0=[1.0, 0.0])
        with pytest.raises(ValueError, match="No PML elements"):
            build_pml_operator(nodes, {}, C0, pml)

    def test_dimension_mismatch(self):
        nodes, _, layer, _, _, _ = duct_3d(0.0, 1.0, 1.0, 1.0, 0, 1, 1, 1)
        pml = cartesian(inner_min=[-1e9, -1e9], inner_max=[0.0, 1e9],
                           thickness=[1.0, 0.0], sigma0=[1.0, 0.0])
        with pytest.raises(ValueError, match="but CartesianPML is"):
            build_pml_operator(nodes, {"Hexahedron 8": layer}, C0, pml)

    def test_mixed_dimensions(self):
        nodes, _, layer2d, _, _ = duct_2d(0.0, 1.0, 1.0, 0, 1, 1)
        nodes3, _, layer3d, _, _, _ = duct_3d(0.0, 1.0, 1.0, 1.0, 0, 1, 1, 1)
        pml = cartesian(inner_min=[-1e9, -1e9], inner_max=[0.0, 1e9],
                           thickness=[1.0, 0.0], sigma0=[1.0, 0.0])
        with pytest.raises(ValueError, match="mix"):
            build_pml_operator(nodes3, {"Quadrilateral 4": layer2d,
                                        "Hexahedron 8": layer3d}, C0, pml)

    def test_wrong_node_count(self):
        nodes, _, layer, _, _ = duct_2d(0.0, 1.0, 1.0, 0, 1, 1)
        pml = cartesian(inner_min=[-1e9, -1e9], inner_max=[0.0, 1e9],
                           thickness=[1.0, 0.0], sigma0=[1.0, 0.0])
        with pytest.raises(ValueError, match="8 nodes per element"):
            build_pml_operator(nodes, {"Quadrilateral 8": layer}, C0, pml)


# ---------------------------------------------------------------------------
# Curved layers
# ---------------------------------------------------------------------------

def shell_mesh(r_inner, r_interface, r_outer, n_r_fluid, n_r_pml, n_theta,
               n_z=None):
    """
    A structured shell, hexahedral, between two radii.

    2D when ``n_z`` is None (an annulus of quads), otherwise a tube of
    bricks along z. The elements are split into fluid and layer at
    ``r_interface``.
    """
    radii = np.concatenate([
        np.linspace(r_inner, r_interface, n_r_fluid + 1),
        np.linspace(r_interface, r_outer, n_r_pml + 1)[1:],
    ])
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)

    if n_z is None:
        pts = [(r * np.cos(t), r * np.sin(t)) for r in radii for t in theta]
        nodes = np.array(pts)

        def nid(i, j):
            return i * n_theta + (j % n_theta)

        fluid, layer = [], []
        for i in range(len(radii) - 1):
            for j in range(n_theta):
                elem = [nid(i, j) + 1, nid(i + 1, j) + 1,
                        nid(i + 1, j + 1) + 1, nid(i, j + 1) + 1]
                (layer if radii[i] >= r_interface - 1e-12
                 else fluid).append(elem)
        return nodes, np.array(fluid), np.array(layer), "Quadrilateral 4"

    z = np.linspace(0.0, 1.0, n_z + 1)
    pts = [(r * np.cos(t), r * np.sin(t), zz)
           for r in radii for t in theta for zz in z]
    nodes = np.array(pts)

    def nid(i, j, k):
        return (i * n_theta + (j % n_theta)) * (n_z + 1) + k

    fluid, layer = [], []
    for i in range(len(radii) - 1):
        for j in range(n_theta):
            for k in range(n_z):
                elem = [nid(i, j, k) + 1, nid(i + 1, j, k) + 1,
                        nid(i + 1, j + 1, k) + 1, nid(i, j + 1, k) + 1,
                        nid(i, j, k + 1) + 1, nid(i + 1, j, k + 1) + 1,
                        nid(i + 1, j + 1, k + 1) + 1, nid(i, j + 1, k + 1) + 1]
                (layer if radii[i] >= r_interface - 1e-12
                 else fluid).append(elem)
    return nodes, np.array(fluid), np.array(layer), "Hexahedron 8"


class TestCurvedLayers:

    OMEGA = 2 * np.pi * 300.0

    def test_radial_2d_reduces_to_the_ordinary_matrices(self):
        nodes, fluid, _, key = shell_mesh(0.3, 0.8, 1.0, 4, 2, 24)
        layer = RadialPML(center=[0.0, 0.0], inner_radius=0.3,
                          profile=PowerProfile(sigma0=0.0, thickness=0.5))
        op = build_pml_operator(nodes, {key: fluid}, C0, layer)

        K, M, _ = assemble_KM(nodes, fluid - 1, element_matrices_cquad4, C0)
        expected = (K - self.OMEGA ** 2 * M).toarray()
        got = op.matrix(self.OMEGA).toarray()
        assert np.allclose(got.imag, 0.0, atol=1e-12)
        assert np.allclose(got.real, expected, rtol=1e-9, atol=1e-12)

    def test_cylindrical_reduces_to_the_ordinary_matrices(self):
        nodes, fluid, _, key = shell_mesh(0.3, 0.8, 1.0, 3, 2, 16, n_z=2)
        layer = CylindricalPML(center=[0.0, 0.0, 0.0], axis=[0.0, 0.0, 1.0],
                               inner_radius=0.3,
                               profile=PowerProfile(sigma0=0.0, thickness=0.5))
        op = build_pml_operator(nodes, {key: fluid}, C0, layer)

        K, M, _ = assemble_KM(nodes, fluid - 1, element_matrices_hex8, C0)
        expected = (K - self.OMEGA ** 2 * M).toarray()
        got = op.matrix(self.OMEGA).toarray()
        assert np.allclose(got.imag, 0.0, atol=1e-12)
        assert np.allclose(got.real, expected, rtol=1e-9, atol=1e-12)

    def test_radial_3d_reduces_to_the_ordinary_matrices(self):
        nodes, fluid, _, key = shell_mesh(0.3, 0.8, 1.0, 3, 2, 16, n_z=2)
        layer = RadialPML(center=[0.0, 0.0, 0.5], inner_radius=0.3,
                          profile=PowerProfile(sigma0=0.0, thickness=0.5))
        op = build_pml_operator(nodes, {key: fluid}, C0, layer)

        K, M, _ = assemble_KM(nodes, fluid - 1, element_matrices_hex8, C0)
        expected = (K - self.OMEGA ** 2 * M).toarray()
        got = op.matrix(self.OMEGA).toarray()
        assert np.allclose(got.real, expected, rtol=1e-9, atol=1e-12)

    def test_curved_layers_absorb(self):
        nodes, _, layer_conn, key = shell_mesh(0.3, 0.8, 1.0, 4, 4, 24)
        layer = RadialPML(
            center=[0.0, 0.0], inner_radius=0.8,
            profile=PowerProfile.for_reflection(0.2, 1e-2),
        )
        op = build_pml_operator(nodes, {key: layer_conn}, C0, layer)
        A = op.matrix(self.OMEGA).toarray()
        assert np.abs(A.imag).max() > 1e-6
        assert np.allclose(A, A.T)              # complex symmetric

    def test_an_element_on_the_axis_is_refused(self):
        """
        A shell layer has no radial direction on its own axis, so a
        silently zero frame would give a silently wrong operator.
        """
        layer = CylindricalPML(center=[0.0, 0.0, 0.0], axis=[0.0, 0.0, 1.0],
                               inner_radius=0.5,
                               profile=PowerProfile(1.0, 1.0))
        with pytest.raises(ValueError, match="sit on the axis"):
            layer.frames(np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))

        radial = RadialPML(center=[0.0, 0.0, 0.0], inner_radius=0.5,
                           profile=PowerProfile(1.0, 1.0))
        with pytest.raises(ValueError, match="sit on the centre"):
            radial.frames(np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))

    def test_frames_are_orthonormal(self):
        centroids = np.array([[1.0, 0.0, 0.0], [0.3, -0.7, 2.0],
                              [0.0, 0.4, 1.0], [-1.0, 2.0, -3.0]])
        for layer in (
            RadialPML(center=[0.0, 0.0, 0.0], inner_radius=0.5,
                      profile=PowerProfile(1.0, 1.0)),
            CylindricalPML(center=[0.0, 0.0, 0.0], axis=[0.0, 0.0, 1.0],
                           inner_radius=0.5, profile=PowerProfile(1.0, 1.0)),
        ):
            frames = layer.frames(centroids)
            for frame in frames:
                assert np.allclose(frame @ frame.T, np.eye(3), atol=1e-12)

    def test_radial_needs_a_positive_radius(self):
        with pytest.raises(ValueError, match="Non-positive inner radius"):
            RadialPML(center=[0.0, 0.0], inner_radius=0.0,
                      profile=PowerProfile(1.0, 1.0))

    def test_cylinder_axis_cannot_be_zero(self):
        with pytest.raises(ValueError, match="cannot be the zero vector"):
            CylindricalPML(center=[0.0, 0.0, 0.0], axis=[0.0, 0.0, 0.0],
                           inner_radius=1.0, profile=PowerProfile(1.0, 1.0))


# ---------------------------------------------------------------------------
# Reading the layer off the mesh
# ---------------------------------------------------------------------------

def as_groups(nodes, fluid, layer, key, extra=None):
    """The subset of load_mesh_with_groups that the PML factory reads."""
    groups = {
        "fluid": {"dim": 3, "tag": 1, "elements": {key: fluid},
                  "nodes": np.unique(fluid)},
        "pml": {"dim": 3, "tag": 2, "elements": {key: layer},
                "nodes": np.unique(layer)},
    }
    if extra:
        groups.update(extra)
    return groups


class TestShapeDetection:

    def test_box_is_recognised(self):
        nodes, fluid, layer, x, y, z = duct_3d(1.0, 0.4, 0.3, 0.25,
                                               4, 2, 2, 2)
        shape, params = identify_layer_shape(
            nodes, np.unique(fluid) - 1, np.unique(layer) - 1
        )
        assert shape.kind == "box"
        assert params["inner_max"][0] == pytest.approx(1.0)
        assert params["thickness"][0] == pytest.approx(0.4)

    def test_cylindrical_is_recognised(self):
        nodes, fluid, layer, key = shell_mesh(0.3, 0.8, 1.0, 3, 2, 24, n_z=2)
        shape, params = identify_layer_shape(
            nodes, np.unique(fluid) - 1, np.unique(layer) - 1
        )
        assert shape.kind == "cylindrical"
        assert params["inner_radius"] == pytest.approx(0.8, rel=1e-6)
        assert params["thickness"] == pytest.approx(0.2, rel=1e-6)
        assert np.allclose(np.abs(params["axis"]), [0.0, 0.0, 1.0])

    def test_circular_is_recognised_in_2d(self):
        nodes, fluid, layer, key = shell_mesh(0.3, 0.8, 1.0, 3, 2, 24)
        shape, params = identify_layer_shape(
            nodes, np.unique(fluid) - 1, np.unique(layer) - 1
        )
        assert shape.kind == "spherical"       # the 2D radial layer
        assert params["inner_radius"] == pytest.approx(0.8, rel=1e-6)
        assert np.allclose(params["center"], 0.0, atol=1e-9)

    def test_too_few_shared_nodes(self):
        nodes, fluid, layer, key = shell_mesh(0.3, 0.8, 1.0, 3, 2, 24)
        with pytest.raises(ValueError, match="too\\s+few to tell"):
            identify_layer_shape(nodes, np.unique(fluid) - 1,
                                 np.array([0]))


class TestFromGroups:

    def test_no_pml_group_means_no_layer(self):
        nodes, fluid, layer, key = shell_mesh(0.3, 0.8, 1.0, 3, 2, 16, n_z=2)
        groups = {"fluid": {"dim": 3, "tag": 1, "elements": {key: fluid},
                            "nodes": np.unique(fluid)}}
        assert pml_from_groups(nodes, groups, C0) is None

    def test_cylindrical_layer_from_groups(self):
        nodes, fluid, layer, key = shell_mesh(0.3, 0.8, 1.0, 3, 6, 24, n_z=2)
        op = pml_from_groups(nodes, as_groups(nodes, fluid, layer, key), C0,
                             target_reflection=1e-2, verbose=False)
        assert isinstance(op.layer, CylindricalPML)
        assert op.layer.inner_radius == pytest.approx(0.8, rel=1e-6)
        assert op.n_elements == layer.shape[0]

    def test_box_layer_from_groups(self):
        nodes, fluid, layer, x, y, z = duct_3d(1.0, 0.4, 0.3, 0.25,
                                               4, 6, 2, 2)
        op = pml_from_groups(nodes, as_groups(nodes, fluid, layer,
                                              "Hexahedron 8"), C0,
                             target_reflection=1e-2, verbose=False)
        assert isinstance(op.layer, CartesianPML)
        # only the axis the layer actually extends along is stretched
        stretched = [p.thickness > 0.0 for p in op.layer.profiles]
        assert stretched == [True, False, False]

    def test_forced_shape(self):
        nodes, fluid, layer, key = shell_mesh(0.3, 0.8, 1.0, 3, 6, 24, n_z=2)
        op = pml_from_groups(nodes, as_groups(nodes, fluid, layer, key), C0,
                             shape="spherical", target_reflection=1e-2,
                             verbose=False)
        assert isinstance(op.layer, RadialPML)

    def test_unknown_shape(self):
        nodes, fluid, layer, key = shell_mesh(0.3, 0.8, 1.0, 3, 2, 16, n_z=2)
        with pytest.raises(ValueError, match="Unknown layer shape"):
            pml_from_groups(nodes, as_groups(nodes, fluid, layer, key), C0,
                            shape="conical", verbose=False)

    def test_overlapping_groups_are_refused(self):
        nodes, fluid, layer, key = shell_mesh(0.3, 0.8, 1.0, 3, 2, 16, n_z=2)
        both = np.vstack([layer, fluid[:1]])
        with pytest.raises(ValueError, match="share"):
            pml_from_groups(nodes, as_groups(nodes, fluid, both, key), C0,
                            verbose=False)

    def test_a_thin_layer_warns(self):
        nodes, fluid, layer, key = shell_mesh(0.3, 0.8, 1.0, 3, 2, 16, n_z=2)
        with pytest.warns(RuntimeWarning, match="element\\(s\\) thick"):
            pml_from_groups(nodes, as_groups(nodes, fluid, layer, key), C0,
                            verbose=False)

    def test_target_reflection_sets_sigma0(self):
        nodes, fluid, layer, key = shell_mesh(0.3, 0.8, 1.0, 3, 6, 24, n_z=2)
        groups = as_groups(nodes, fluid, layer, key)
        strong = pml_from_groups(nodes, groups, C0, target_reflection=1e-4,
                                 verbose=False)
        weak = pml_from_groups(nodes, groups, C0, target_reflection=1e-1,
                               verbose=False)
        assert strong.layer.profile.sigma0 > weak.layer.profile.sigma0


# ---------------------------------------------------------------------------
# End to end: naming the group is the whole switch
# ---------------------------------------------------------------------------

def duct_msh(path, L_phys, L_pml, H, W, with_pml=True):
    """A gmsh duct whose last stretch is a separate 'pml' group."""
    gmsh = pytest.importorskip("gmsh")
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("duct")
        occ = gmsh.model.occ
        fluid = occ.addBox(0, 0, 0, L_phys, H, W)
        layer = occ.addBox(L_phys, 0, 0, L_pml, H, W)
        occ.fragment([(3, fluid)], [(3, layer)])
        occ.synchronize()

        volumes = gmsh.model.getEntities(3)
        fl = [t for _, t in volumes
              if occ.getCenterOfMass(3, t)[0] < L_phys - 1e-9]
        pm = [t for _, t in volumes if t not in fl]
        if with_pml:
            gmsh.model.addPhysicalGroup(3, fl, name="fluid")
            gmsh.model.addPhysicalGroup(3, pm, name="pml")
        else:
            gmsh.model.addPhysicalGroup(3, fl + pm, name="fluid")

        inlet = [t for _, t in gmsh.model.getEntities(2)
                 if abs(occ.getCenterOfMass(2, t)[0]) < 1e-9]
        gmsh.model.addPhysicalGroup(2, inlet, name="inlet")

        gmsh.option.setNumber("Mesh.MeshSizeMax", 0.035)
        gmsh.option.setNumber("Mesh.RecombineAll", 0)
        gmsh.model.mesh.generate(3)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.write(str(path))
    finally:
        gmsh.finalize()
    return path


class TestFromTheMesh:
    """
    The requirement: a mesh with a group called ``pml`` is absorbing,
    one without it is not, and nothing else has to be said.
    """

    L_PHYS, L_PML, H, W = 0.8, 0.35, 0.12, 0.12
    FREQ = 500.0

    def _run(self, path, target=1e-1):
        import pycafe
        from pycafe.create_geom.visualize_mesh import load_mesh_with_groups
        from pycafe.solver.solver_helmholtz_1 import (
            solve_helmholtz_frequency_sweep,
        )
        from pycafe.boundary_condition.acoustic_bc import AcousticBC

        nodes, elements, boundaries, groups = load_mesh_with_groups(str(path))
        system = pycafe.prepare_acoustic_system(
            nodes=nodes, elements=elements, boundaries=boundaries,
            rho=RHO, c0=C0,
            bc=AcousticBC().add_velocity("inlet", -1.0),
            groups=groups,
            pml={"target_reflection": target, "verbose": False},
        )
        freqs = np.array([self.FREQ])
        P_red = solve_helmholtz_frequency_sweep(
            K_red=system["K_red"], M_red=system["M_red"],
            C_red=system["C_red"], frequencies=freqs,
            pressure_nodes_red=system["pressure_nodes_red"],
            pressure_values=system["pressure_values"],
            nodes=nodes, idx_free=system["idx_free"], rho=RHO,
            boundaries=boundaries, elements=elements, groups=groups,
            boundary_dim=system["boundary_dim"],
            velocity_operator=system["velocity_red_op"],
            impedance_operator=system["C_red_op"],
            source_operator=system["source_red_op"],
            pml_operator=system["pml_red_op"],
        )
        p = np.zeros(nodes.shape[0], dtype=complex)
        p[system["idx_free"]] = P_red[:, 0]

        # probe along the axis of the physical stretch
        axis = np.where(
            (np.abs(nodes[:, 1] - self.H / 2) < 0.02)
            & (np.abs(nodes[:, 2] - self.W / 2) < 0.02)
            & (nodes[:, 0] > 0.1) & (nodes[:, 0] < self.L_PHYS - 0.05)
        )[0]
        amp = np.abs(p[axis])
        swr = (amp.max() - amp.min()) / (amp.max() + amp.min())
        return system, swr

    def test_a_pml_group_absorbs(self, tmp_path):
        path = duct_msh(tmp_path / "with.msh", self.L_PHYS, self.L_PML,
                        self.H, self.W, with_pml=True)
        system, swr = self._run(path)
        assert system["pml_op"] is not None
        assert isinstance(system["pml_op"].layer, CartesianPML)
        assert swr < 0.15

    def test_asking_for_too_little_reflection_backfires(self, tmp_path):
        """
        The layer is thin and the mesh is tetrahedral, so the profile is
        a coarse, irregular staircase. Past a point, raising sigma0 makes
        the wave scatter off the staircase rather than be absorbed by it,
        and the reflection **grows**. The behaviour is documented in
        :meth:`PowerProfile.for_reflection`; this pins it down so that a
        change in the profile handling shows up.
        """
        path = duct_msh(tmp_path / "greedy.msh", self.L_PHYS, self.L_PML,
                        self.H, self.W, with_pml=True)
        _, swr_mild = self._run(path, target=1e-1)
        _, swr_greedy = self._run(path, target=1e-3)
        assert swr_greedy > swr_mild

    def test_without_the_group_the_duct_rings(self, tmp_path):
        path = duct_msh(tmp_path / "without.msh", self.L_PHYS, self.L_PML,
                        self.H, self.W, with_pml=False)
        system, swr = self._run(path)
        assert system["pml_op"] is None
        assert swr > 0.5

    def test_pml_false_ignores_the_group(self, tmp_path):
        import pycafe
        from pycafe.create_geom.visualize_mesh import load_mesh_with_groups
        from pycafe.boundary_condition.acoustic_bc import AcousticBC

        path = duct_msh(tmp_path / "off.msh", self.L_PHYS, self.L_PML,
                        self.H, self.W, with_pml=True)
        nodes, elements, boundaries, groups = load_mesh_with_groups(str(path))
        system = pycafe.prepare_acoustic_system(
            nodes=nodes, elements=elements, boundaries=boundaries,
            rho=RHO, c0=C0, bc=AcousticBC().add_velocity("inlet", -1.0),
            groups=groups, pml=False,
        )
        assert system["pml_op"] is None
