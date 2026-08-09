"""
Validation tests for the modal acoustic solver against analytical solutions.

For a 2D rectangular cavity with rigid walls (all Neumann BCs), the exact
natural frequencies are:

    f_mn = (c0 / 2) * sqrt((m / Lx)^2 + (n / Ly)^2)

where m, n = 0, 1, 2, ... (not both zero).

The (0,0) mode corresponds to the trivial uniform-pressure mode (zero frequency)
which is filtered out by the solver. The first physical modes are (1,0) and (0,1).

Tests verify:
- Solver returns positive frequencies
- Frequencies match analytical values within tolerance depending on mesh density
- Mesh refinement monotonically reduces the frequency error (convergence)
- Modal shapes are orthogonal with respect to the mass matrix

Reviewers addressed: R1, R3, R4, R5, R6 (analytical benchmark, convergence study)
"""

import numpy as np
import pytest

from pycafe.build_matrices.assembly_cquad4 import build_KM_cquad4
from pycafe.solver.solver_modale import solve_modal_acoustic_reduced


# Helper: structured CQUAD4 mesh

def make_rect_mesh_cquad4(Lx, Ly, nx, ny):
    """Structured rectangular CQUAD4 mesh on [0,Lx]×[0,Ly]."""
    x = np.linspace(0.0, Lx, nx + 1)
    y = np.linspace(0.0, Ly, ny + 1)
    X, Y = np.meshgrid(x, y, indexing="ij")
    nodes = np.column_stack([X.ravel(), Y.ravel()])

    conn = []
    for ie in range(nx):
        for je in range(ny):
            n1 = ie * (ny + 1) + je
            n2 = (ie + 1) * (ny + 1) + je
            n3 = (ie + 1) * (ny + 1) + je + 1
            n4 = ie * (ny + 1) + je + 1
            conn.append([n1 + 1, n2 + 1, n3 + 1, n4 + 1])  # 1-based

    elements = {"Quadrilateral 4": np.array(conn, dtype=int)}
    return nodes, elements


def analytical_rigid_cavity_freqs(Lx, Ly, c0, n_modes=6):
    """
    Compute analytical natural frequencies of a 2D rigid rectangular cavity.

    f_mn = (c0/2) * sqrt((m/Lx)^2 + (n/Ly)^2)  for (m,n) != (0,0)

    Returns the first n_modes frequencies in ascending order [Hz].
    """
    freqs = []
    # Limit search to sufficient range
    for m in range(20):
        for n in range(20):
            if m == 0 and n == 0:
                continue
            f = (c0 / 2.0) * np.sqrt((m / Lx) ** 2 + (n / Ly) ** 2)
            freqs.append(f)
    freqs = np.sort(freqs)  # keep duplicates (degenerate modes count separately)
    return freqs[:n_modes]


# Tests: basic solver correctness

class TestModalSolverBasic:

    def test_returns_positive_frequencies(self):
        """Solver must return strictly positive natural frequencies."""
        nodes, elements = make_rect_mesh_cquad4(1.0, 1.0, 4, 4)
        c0 = 343.0
        K, M, _ = build_KM_cquad4(nodes, elements, c0)
        freqs, _ = solve_modal_acoustic_reduced(K, M, num_modes=3)
        assert np.all(freqs > 0), f"Non-positive frequencies found: {freqs}"

    def test_frequencies_are_sorted(self):
        """Returned frequencies must be in ascending order."""
        nodes, elements = make_rect_mesh_cquad4(1.0, 0.5, 4, 4)
        c0 = 343.0
        K, M, _ = build_KM_cquad4(nodes, elements, c0)
        freqs, _ = solve_modal_acoustic_reduced(K, M, num_modes=5)
        assert np.all(freqs[:-1] <= freqs[1:]), "Frequencies not sorted"

    def test_mode_shapes_shape(self):
        """Mode shapes array has shape (n_free, num_modes)."""
        nodes, elements = make_rect_mesh_cquad4(1.0, 1.0, 4, 4)
        c0 = 343.0
        K, M, _ = build_KM_cquad4(nodes, elements, c0)
        n_modes = 4
        freqs, modes = solve_modal_acoustic_reduced(K, M, num_modes=n_modes)
        assert modes.shape[1] == n_modes
        assert modes.shape[0] == K.shape[0]

    def test_modes_orthogonal_wrt_mass(self):
        """
        Mode shapes must be M-orthogonal:
            phi_i^T @ M @ phi_j = 0  for i != j
        """
        nodes, elements = make_rect_mesh_cquad4(1.0, 1.0, 6, 6)
        c0 = 343.0
        K, M, _ = build_KM_cquad4(nodes, elements, c0)
        n_modes = 4
        _, modes = solve_modal_acoustic_reduced(K, M, num_modes=n_modes)

        M_dense = M.toarray()
        for i in range(n_modes):
            for j in range(n_modes):
                val = modes[:, i] @ M_dense @ modes[:, j]
                if i != j:
                    assert abs(val) < 1e-10, (
                        f"Modes {i} and {j} not M-orthogonal: "
                        f"phi_{i}^T M phi_{j} = {val:.3e}"
                    )


# Tests: analytical validation — first natural frequency

class TestModalAnalyticalValidation:
    """
    Compare FEM natural frequencies to the exact analytical solution
    for a 2D rigid rectangular cavity.

    Cavity: Lx=2.0 m, Ly=1.0 m (non-square to avoid degenerate modes)
    Fluid:  c0 = 343.0 m/s (air at 20°C)

    Analytical first frequency: f_10 = c0/(2*Lx) = 343/4 = 85.75 Hz
    Analytical second freq:      f_01 = c0/(2*Ly) = 343/2 = 171.5 Hz
    """

    LX = 2.0
    LY = 1.0
    C0 = 343.0

    def _run_modal(self, nx, ny, n_modes=4):
        nodes, elements = make_rect_mesh_cquad4(self.LX, self.LY, nx, ny)
        K, M, _ = build_KM_cquad4(nodes, elements, self.C0)
        freqs, modes = solve_modal_acoustic_reduced(K, M, num_modes=n_modes)
        return freqs, modes

    def test_first_frequency_moderate_mesh(self):
        """
        First natural frequency within 1% of analytical solution
        for a 10×5 element mesh.
        """
        freqs, _ = self._run_modal(nx=10, ny=5)
        f_exact = self.C0 / (2.0 * self.LX)  # 85.75 Hz
        rel_error = abs(freqs[0] - f_exact) / f_exact
        assert rel_error < 0.01, (
            f"First frequency error too large: FEM={freqs[0]:.4f} Hz, "
            f"Exact={f_exact:.4f} Hz, Rel.error={rel_error*100:.2f}%"
        )

    def test_second_frequency_moderate_mesh(self):
        """
        Second natural frequency (0,1 mode) within 1% of analytical solution.
        """
        freqs, _ = self._run_modal(nx=10, ny=10)
        f_exact = self.C0 / (2.0 * self.LY)  # 171.5 Hz
        rel_error = abs(freqs[1] - f_exact) / f_exact
        assert rel_error < 0.01, (
            f"Second frequency error too large: FEM={freqs[1]:.4f} Hz, "
            f"Exact={f_exact:.4f} Hz, Rel.error={rel_error*100:.2f}%"
        )

    def test_first_four_frequencies_fine_mesh(self):
        """
        First 4 natural frequencies within 1% of analytical on fine mesh (20×10).
        Note: degenerate modes (e.g. f_01 = f_20 for Lx=2*Ly) are both expected
        and both captured by the FEM — the analytical list preserves duplicates.
        """
        freqs_fem, _ = self._run_modal(nx=20, ny=10, n_modes=4)
        freqs_exact = analytical_rigid_cavity_freqs(self.LX, self.LY, self.C0, n_modes=4)

        for k, (f_fem, f_ex) in enumerate(zip(freqs_fem, freqs_exact)):
            rel_error = abs(f_fem - f_ex) / f_ex
            assert rel_error < 0.01, (
                f"Mode {k+1}: FEM={f_fem:.4f} Hz, Exact={f_ex:.4f} Hz, "
                f"Rel.error={rel_error*100:.3f}%"
            )


# Tests: mesh convergence study

class TestModalConvergence:
    """
    Verify that the FEM natural frequencies converge monotonically
    toward the analytical solution as the mesh is refined.

    CQUAD4 (bilinear elements) approximates frequencies from above.
    With uniform mesh refinement, the error should decrease monotonically.

    Reviewers addressed: R3 (convergence study), R4/R5 (h-refinement).
    """

    LX = 1.0
    LY = 1.0
    C0 = 1.0   # normalised speed for clean numbers

    def _fem_first_freq(self, nx, ny):
        nodes, elements = make_rect_mesh_cquad4(self.LX, self.LY, nx, ny)
        K, M, _ = build_KM_cquad4(nodes, elements, self.C0)
        freqs, _ = solve_modal_acoustic_reduced(K, M, num_modes=1)
        return freqs[0]

    def test_convergence_monotonic(self):
        """
        Frequency error decreases monotonically with mesh refinement.

        Mesh sizes: n×n for n = 2, 4, 8, 16.
        Expected behaviour: error(n=2) > error(n=4) > error(n=8) > error(n=16).
        """
        mesh_sizes = [2, 4, 8, 16]
        f_exact = self.C0 / (2.0 * self.LX)  # 0.5 Hz for c0=1, Lx=1

        errors = []
        for n in mesh_sizes:
            f_fem = self._fem_first_freq(n, n)
            rel_err = abs(f_fem - f_exact) / f_exact
            errors.append(rel_err)

        # Check strict monotonic decrease
        for i in range(len(errors) - 1):
            assert errors[i] > errors[i + 1], (
                f"Non-monotonic convergence at mesh size {mesh_sizes[i]}: "
                f"error={errors[i]:.6f} → {errors[i+1]:.6f}"
            )

    def test_convergence_fine_mesh_accuracy(self):
        """FEM first frequency within 0.15% of analytical on a 24×24 mesh."""
        f_fem = self._fem_first_freq(24, 24)
        f_exact = self.C0 / (2.0 * self.LX)
        rel_error = abs(f_fem - f_exact) / f_exact
        assert rel_error < 0.0015, (
            f"Fine mesh error too large: {rel_error*100:.4f}%"
        )

    def test_convergence_rate(self):
        """
        CQUAD4 elements exhibit approximately O(h^2) convergence in frequency error.

        For mesh doubling (h → h/2), error should decrease by factor ~4.
        We check factor > 2 (conservative) to account for asymptotic behaviour.
        """
        f_exact = self.C0 / (2.0 * self.LX)

        f4 = self._fem_first_freq(4, 4)
        f8 = self._fem_first_freq(8, 8)
        f16 = self._fem_first_freq(16, 16)

        e4 = abs(f4 - f_exact) / f_exact
        e8 = abs(f8 - f_exact) / f_exact
        e16 = abs(f16 - f_exact) / f_exact

        ratio_1 = e4 / e8
        ratio_2 = e8 / e16

        assert ratio_1 > 2.0, (
            f"Convergence rate too slow (4→8 mesh): error ratio={ratio_1:.2f} < 2"
        )
        assert ratio_2 > 2.0, (
            f"Convergence rate too slow (8→16 mesh): error ratio={ratio_2:.2f} < 2"
        )
