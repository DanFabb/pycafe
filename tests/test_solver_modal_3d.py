"""
Validation of the CHEXA8 acoustic element against analytical solutions.

For a 3D rectangular cavity with rigid walls (all Neumann BCs), the
exact natural frequencies are:

    f_lmn = (c0 / 2) * sqrt((l/Lx)^2 + (m/Ly)^2 + (n/Lz)^2)

with l, m, n = 0, 1, 2, ... (not all zero). The (0,0,0) uniform mode
is filtered by the solver.

Includes an end-to-end test on the coupling_geometry box+plate mesh
(generated with gmsh, loaded through the pyCAFE mesh reader and
assembled through the registry dispatcher).
"""

import pathlib
import sys

import numpy as np
import pytest

from pycafe.build_matrices.assembly import assemble_KM
from pycafe.build_matrices.element_hex8 import element_matrices_hex8
from pycafe.solver.solver_modale import solve_modal_acoustic_reduced


# Helper: structured HEXA8 mesh (no gmsh dependency)

def make_box_mesh_hex8(Lx, Ly, Lz, nx, ny, nz):
    """
    Structured box mesh on [0,Lx]x[0,Ly]x[0,Lz] with nx*ny*nz HEXA8.

    Returns (nodes, elements) with 1-based Gmsh-ordered connectivity,
    mirroring make_rect_mesh_cquad4 in test_solver_modal.py.
    """
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
                n = [
                    nid(i, j, k), nid(i + 1, j, k),
                    nid(i + 1, j + 1, k), nid(i, j + 1, k),
                    nid(i, j, k + 1), nid(i + 1, j, k + 1),
                    nid(i + 1, j + 1, k + 1), nid(i, j + 1, k + 1),
                ]
                conn.append([m + 1 for m in n])  # 1-based

    elements = {"Hexahedron 8": np.array(conn, dtype=int)}
    return nodes, elements


def analytical_box_freqs(Lx, Ly, Lz, c0, n_modes=6):
    """First n_modes rigid-cavity frequencies f_lmn, duplicates kept."""
    freqs = []
    for l in range(8):
        for m in range(8):
            for n in range(8):
                if l == m == n == 0:
                    continue
                f = (c0 / 2.0) * np.sqrt(
                    (l / Lx) ** 2 + (m / Ly) ** 2 + (n / Lz) ** 2
                )
                freqs.append(f)
    return np.sort(freqs)[:n_modes]


def build_KM_hex(nodes, elements, c0):
    conn0 = elements["Hexahedron 8"] - 1
    K, M, _ = assemble_KM(nodes, conn0, element_matrices_hex8, c0)
    return K, M


# Analytical validation

class TestModal3DAnalyticalValidation:
    """
    Cavity: 1.0 x 0.7 x 0.5 m (all different, no degenerate modes early).
    First frequency: f_100 = c0/(2*Lx) = 171.5 Hz.
    """

    LX, LY, LZ = 1.0, 0.7, 0.5
    C0 = 343.0

    def _run_modal(self, nx, ny, nz, n_modes=4):
        nodes, elements = make_box_mesh_hex8(
            self.LX, self.LY, self.LZ, nx, ny, nz
        )
        K, M = build_KM_hex(nodes, elements, self.C0)
        return solve_modal_acoustic_reduced(K, M, num_modes=n_modes)

    def test_first_frequency_moderate_mesh(self):
        freqs, _ = self._run_modal(10, 7, 5)
        f_exact = self.C0 / (2.0 * self.LX)
        rel = abs(freqs[0] - f_exact) / f_exact
        assert rel < 0.01, (
            f"FEM={freqs[0]:.3f} Hz, exact={f_exact:.3f} Hz, err={rel*100:.2f}%"
        )

    def test_first_four_frequencies(self):
        freqs_fem, _ = self._run_modal(12, 9, 7, n_modes=4)
        freqs_exact = analytical_box_freqs(
            self.LX, self.LY, self.LZ, self.C0, n_modes=4
        )
        for k, (f_fem, f_ex) in enumerate(zip(freqs_fem, freqs_exact)):
            rel = abs(f_fem - f_ex) / f_ex
            assert rel < 0.01, (
                f"Mode {k+1}: FEM={f_fem:.3f} Hz, exact={f_ex:.3f} Hz, "
                f"err={rel*100:.3f}%"
            )

    def test_modes_M_orthogonal(self):
        nodes, elements = make_box_mesh_hex8(1.0, 1.0, 1.0, 4, 4, 4)
        K, M = build_KM_hex(nodes, elements, self.C0)
        _, modes = solve_modal_acoustic_reduced(K, M, num_modes=3)
        Md = M.toarray()
        for i in range(3):
            for j in range(3):
                if i != j:
                    assert abs(modes[:, i] @ Md @ modes[:, j]) < 1e-10


# Convergence

class TestModal3DConvergence:

    C0 = 1.0

    def _first_freq(self, n):
        nodes, elements = make_box_mesh_hex8(1.0, 1.0, 1.0, n, n, n)
        K, M = build_KM_hex(nodes, elements, self.C0)
        freqs, _ = solve_modal_acoustic_reduced(K, M, num_modes=1)
        return freqs[0]

    def test_monotonic_and_rate(self):
        """Error decreases monotonically, ~O(h^2) (ratio > 2 per doubling)."""
        f_exact = 0.5  # c0/(2*Lx)
        errs = [abs(self._first_freq(n) - f_exact) / f_exact
                for n in (2, 4, 8)]
        assert errs[0] > errs[1] > errs[2]
        assert errs[0] / errs[1] > 2.0
        assert errs[1] / errs[2] > 2.0


# End-to-end on the coupling geometry (gmsh required)

class TestBoxPlateMeshEndToEnd:
    """
    Generate the coupling_geometry box+plate mesh with gmsh, load it
    through pyCAFE and verify: registry detects CHEXA8, cavity modes
    match the analytical rigid-box solution.
    """

    def test_box_plate_modal(self, tmp_path):
        pytest.importorskip("gmsh")
        repo_root = pathlib.Path(__file__).parent.parent
        sys.path.insert(0, str(repo_root / "coupling_geometry"))
        from make_box_plate_mesh import make_box_plate_mesh

        Lx, Ly, Lz = 0.8, 0.6, 0.5
        msh = make_box_plate_mesh(
            Lx=Lx, Ly=Ly, Lz=Lz, nx=10, ny=8, nz=8,
            output_path=tmp_path / "box_plate_test.msh",
        )

        from pycafe.create_geom.visualize_mesh import load_mesh_and_elements
        from pycafe.build_matrices.assembly_dispatcher import build_KM_acoustic

        nodes, elements, boundaries = load_mesh_and_elements(str(msh))

        c0 = 343.0
        K, M, _, elem_type = build_KM_acoustic(nodes, elements, c0)
        assert elem_type == "CHEXA8"

        freqs_fem, _ = solve_modal_acoustic_reduced(K, M, num_modes=3)
        freqs_exact = analytical_box_freqs(Lx, Ly, Lz, c0, n_modes=3)

        for f_fem, f_ex in zip(freqs_fem, freqs_exact):
            rel = abs(f_fem - f_ex) / f_ex
            assert rel < 0.01, (
                f"FEM={f_fem:.3f} Hz, exact={f_ex:.3f} Hz, err={rel*100:.2f}%"
            )


# Discrete dispersion law (validation notebook: hexa8_cavity_validation.ipynb)

class TestDiscreteDispersion:
    """
    On a structured mesh of a rectangular cavity the trilinear element with
    consistent mass has a known discrete dispersion relation:

        (omega_h h / c)^2 = 6 (1 - cos(k h)) / (2 + cos(k h))

    which, being separable, predicts every cavity frequency exactly. This
    pins the element down completely: shape functions, quadrature, Jacobian
    and assembly must all be right for the computed frequencies to land on
    the predicted ones.

    Cavity of the validation notebook: 0.414 x 0.314 x 0.360 m.
    """

    LX, LY, LZ = 0.414, 0.314, 0.360
    C0 = 343.0

    def dispersion_frequency(self, idx, n):
        L = np.array([self.LX, self.LY, self.LZ])
        h = L / n
        k = np.array(idx, dtype=float) * np.pi / L
        kh = k * h
        S = np.where(kh > 0,
                     6 * (1 - np.cos(kh)) / ((2 + np.cos(kh)) * h**2), 0.0)
        return self.C0 * np.sqrt(S.sum()) / (2 * np.pi)

    def test_frequencies_follow_dispersion_law(self):
        n = 8
        nodes, elements = make_box_mesh_hex8(self.LX, self.LY, self.LZ,
                                             n, n, n)
        K, M = build_KM_hex(nodes, elements, self.C0)
        freqs, modes = solve_modal_acoustic_reduced(K, M, num_modes=8)

        # identify each mode by correlation with the analytical shapes
        cand = [(l, m, o) for l in range(4) for m in range(4) for o in range(4)
                if (l, m, o) != (0, 0, 0)]
        shapes = np.array([
            np.cos(l * np.pi * nodes[:, 0] / self.LX)
            * np.cos(m * np.pi * nodes[:, 1] / self.LY)
            * np.cos(o * np.pi * nodes[:, 2] / self.LZ)
            for l, m, o in cand])

        for i, f_fem in enumerate(freqs):
            v = modes[:, i]
            mac = (shapes @ v) ** 2 / ((shapes * shapes).sum(1) * (v @ v))
            j = int(mac.argmax())
            assert mac[j] > 0.999, f"mode {i+1} not identified (MAC {mac[j]})"
            f_pred = self.dispersion_frequency(cand[j], n)
            assert abs(f_fem - f_pred) / f_pred < 1e-9, (
                f"mode {i+1} {cand[j]}: FEM={f_fem:.6f} Hz, "
                f"dispersion law={f_pred:.6f} Hz"
            )

    def test_mass_matrix_integrates_the_volume(self):
        nodes, elements = make_box_mesh_hex8(self.LX, self.LY, self.LZ, 4, 4, 4)
        _, M = build_KM_hex(nodes, elements, self.C0)
        one = np.ones(nodes.shape[0])
        volume = one @ (M @ one) * self.C0**2
        assert np.isclose(volume, self.LX * self.LY * self.LZ, rtol=1e-12)
