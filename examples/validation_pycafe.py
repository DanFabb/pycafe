"""
validation_pycafe.py
====================
Publication-oriented validation benchmark for pyCAFE on a 2D rectangular
acoustic cavity.

The script generates the figures used to assess:
- modal accuracy against analytical solutions for hard-wall and
  pressure-release cavities,
- h-refinement trends for CQUAD4 and CQUAD8,
- cross-validation against FEniCSx for modal and direct-frequency analyses.

Dependencies
------------
- pyCAFE
- gmsh
- numpy
- matplotlib
- mpi4py
- dolfinx
- ufl
- scipy
- slepc4py

Notes
-----
- This script writes figure files only. It does not generate manuscript text.
- The benchmark assumes a local environment where pyCAFE and FEniCSx are both
  installed and available in the same Python environment.
- The mesh is built with transfinite divisions obtained from the target h. The
  actual element spacing is therefore the closest structured value compatible
  with the rectangle dimensions.

Outputs
-------
- validation_histogram.png   : grouped bar chart, 10 modes (analytical / CQUAD4 / CQUAD8)
- validation_convergence.png : log-log h-convergence, both element types
- comparison_modeshapes.png  : pyCAFE vs FEniCSx mode-shape comparison
- comparison_direct_*_*.png  : direct-field comparisons for pressure / velocity / impedance BCs

Balanced mesh configuration used for the modal comparison figures:
  CQUAD4  h = 0.07 m  (~middle of the convergence study range)
  CQUAD8  h = 0.07 m  (higher order compensates the coarser mesh)

Run from the repo root:
    conda run -n pycafe1 python examples/validation_pycafe.py
"""

import pathlib
import tempfile

import gmsh
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

import pycafe
from pycafe.solver.solver_modale import solve_modal_acoustic_reduced
from pycafe.build_matrices.assembly_cquad8 import expand_mode_to_full
from pycafe.build_matrices.assembly_cquad8 import expand_to_full

# ── parameters ────────────────────────────────────────────────────────────────
Lx      = 1.0    # cavity length  [m]
Ly      = 0.5    # cavity height  [m]
c0      = 343.0  # speed of sound [m/s]
rho     = 1.204  # air density    [kg/m³]
N_MODES = 10     # modes for comparison

# Balanced config: h=0.07 sits in the middle of the convergence range.
# CQUAD8 already achieves <0.05 % mean error here; CQUAD4 shows the
# accuracy gap without being unreasonably slow.
H_BAL  = 0.07
H_VALS = [0.20, 0.14, 0.10, 0.07, 0.05, 0.035]   # convergence sweep

OUT_DIR = pathlib.Path(__file__).parent


# ── helpers ───────────────────────────────────────────────────────────────────

def analytical_freqs(c0, Lx, Ly, n):
    """Exact eigenfrequencies of a 2D rigid rectangular cavity."""
    modes = []
    for m in range(12):
        for k in range(12):
            if m == 0 and k == 0:
                continue
            f = 0.5 * c0 * np.sqrt((m / Lx) ** 2 + (k / Ly) ** 2)
            modes.append((f, m, k))
    modes.sort()
    return modes[:n]


def analytical_freqs_zero_pressure(c0, Lx, Ly, n):
    """Exact eigenfrequencies of a 2D pressure-release rectangular cavity (p=0 on all walls)."""
    modes = []
    for m in range(1, 15):
        for k in range(1, 15):
            f = 0.5 * c0 * np.sqrt((m / Lx) ** 2 + (k / Ly) ** 2)
            modes.append((f, m, k))
    modes.sort()
    return modes[:n]


def make_mesh(Lx, Ly, h, order, filepath):
    """Transfinite structured quad mesh — order 1 = CQUAD4, order 2 = CQUAD8."""
    try:
        if hasattr(gmsh, "isInitialized") and gmsh.isInitialized():
            gmsh.finalize()
    except Exception:
        pass
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.model.add("rect")
    p1 = gmsh.model.geo.addPoint(0,  0,  0, h)
    p2 = gmsh.model.geo.addPoint(Lx, 0,  0, h)
    p3 = gmsh.model.geo.addPoint(Lx, Ly, 0, h)
    p4 = gmsh.model.geo.addPoint(0,  Ly, 0, h)
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p1)
    cl   = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    surf = gmsh.model.geo.addPlaneSurface([cl])
    gmsh.model.addPhysicalGroup(1, [l1], name="bottom")
    gmsh.model.addPhysicalGroup(1, [l2], name="right")
    gmsh.model.addPhysicalGroup(1, [l3], name="top")
    gmsh.model.addPhysicalGroup(1, [l4], name="left")
    gmsh.model.addPhysicalGroup(2, [surf], name="domain")
    gmsh.model.geo.synchronize()
    nx = max(int(round(Lx / h)) + 1, 3)
    ny = max(int(round(Ly / h)) + 1, 3)
    for line in [l1, l3]:
        gmsh.model.mesh.setTransfiniteCurve(line, nx)
    for line in [l2, l4]:
        gmsh.model.mesh.setTransfiniteCurve(line, ny)
    gmsh.model.mesh.setTransfiniteSurface(surf)
    gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 1)
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.recombine()
    gmsh.model.mesh.setOrder(order)
    gmsh.write(str(filepath))
    gmsh.finalize()


def run_pycafe(filepath, rho, c0, n_modes):
    """All-Neumann modal analysis. Returns (freqs_hz, n_nodes, elapsed_s)."""
    import time
    nodes, elements, boundaries = pycafe.load_mesh(
        str(filepath), show_info=False, show_plot=False
    )
    bc = ([], [], 0.0, [], 0.0j, [], 0.0, None, 0.0)
    t0 = time.perf_counter()
    system = pycafe.prepare_acoustic_system(
        nodes=nodes, elements=elements, boundaries=boundaries,
        rho=rho, c0=c0, bc=bc, debug=False,
    )
    raw_f, _ = solve_modal_acoustic_reduced(
        system["K_red"], system["M_red"], num_modes=n_modes + 4
    )
    elapsed = time.perf_counter() - t0
    freqs = raw_f[raw_f > 1.0][:n_modes]
    return freqs, nodes.shape[0], elapsed


# ── analytical reference ───────────────────────────────────────────────────────
analytic = analytical_freqs(c0, Lx, Ly, N_MODES)
f_ref    = np.array([x[0] for x in analytic])
labels   = [f"({x[1]},{x[2]})" for x in analytic]   # plain text for tick labels

print(f"Cavity {Lx} m × {Ly} m  |  c0={c0} m/s  rho={rho} kg/m³")
print(f"\nAnalytical eigenfrequencies (first {N_MODES} modes):")
for f, m, k in analytic:
    print(f"  ({m},{k})  {f:.4f} Hz")

# ── balanced run — modal comparison + histogram data ──────────────────────────
print(f"\nBalanced mesh  h = {H_BAL} m  (CQUAD4 and CQUAD8)")

with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as tmp:
    msh4 = pathlib.Path(tmp.name)
with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as tmp:
    msh8 = pathlib.Path(tmp.name)

make_mesh(Lx, Ly, H_BAL, order=1, filepath=msh4)
make_mesh(Lx, Ly, H_BAL, order=2, filepath=msh8)

freqs_q4, n_nodes_q4, _ = run_pycafe(msh4, rho, c0, N_MODES)
freqs_q8, n_nodes_q8, _ = run_pycafe(msh8, rho, c0, N_MODES)

msh4.unlink(missing_ok=True)
msh8.unlink(missing_ok=True)

err_q4 = np.abs(freqs_q4 - f_ref) / f_ref * 100.0
err_q8 = np.abs(freqs_q8 - f_ref) / f_ref * 100.0

print(f"\n{'Mode':>8}  {'Analytical [Hz]':>16}  {'CQUAD4 [Hz]':>13}  "
      f"{'err%':>7}  {'CQUAD8 [Hz]':>13}  {'err%':>7}")
print("-" * 76)
for i in range(N_MODES):
    print(f"  {labels[i]:>6}  {f_ref[i]:>16.4f}  {freqs_q4[i]:>13.4f}  "
          f"{err_q4[i]:>6.4f}%  {freqs_q8[i]:>13.4f}  {err_q8[i]:>6.4f}%")
print(f"\n  CQUAD4 nodes={n_nodes_q4}  mean error={err_q4.mean():.4f} %")
print(f"  CQUAD8 nodes={n_nodes_q8}  mean error={err_q8.mean():.4f} %")

# ── histogram figure ───────────────────────────────────────────────────────────
# bar order: Analytical | CQUAD8 | CQUAD4  (CQUAD8 adjacent to analytical)
x     = np.arange(N_MODES)
width = 0.26

plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["DejaVu Serif", "Times New Roman", "Times"],
    "mathtext.fontset":  "dejavuserif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    1.2,
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
})

fig, axes = plt.subplots(2, 1, figsize=(13, 9),
                          gridspec_kw={"height_ratios": [2.2, 1]})

# upper panel — absolute frequencies
ax = axes[0]
ax.bar(x - width,  f_ref,     width, label="Analytical",
       color="#2c5f8a", alpha=0.92, edgecolor="white", linewidth=0.5)
ax.bar(x,          freqs_q8,  width, label="CQUAD8  (p = 2)",
       color="#b82020", alpha=0.92, edgecolor="white", linewidth=0.5)
ax.bar(x + width,  freqs_q4,  width, label="CQUAD4  (p = 1)",
       color="#e07b20", alpha=0.92, edgecolor="white", linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=13, fontweight="bold")
ax.set_ylabel("Eigenfrequency [Hz]", fontsize=14, fontweight="bold")
ax.set_title(
    "Eigenfrequency Comparison — 10 Modes\n"
    f"2D Rigid Rectangular Cavity  {Lx} m × {Ly} m,  Air",
    fontsize=15, fontweight="bold", pad=10,
)
ax.legend(fontsize=12, frameon=True, framealpha=0.9, edgecolor="0.7")
ax.grid(axis="y", linewidth=0.5, color="0.75")
ax.set_xlim(-0.6, N_MODES - 0.4)
ax.tick_params(axis="both", labelsize=12, which="both")

# lower panel — relative error [%]  (CQUAD8 left, CQUAD4 right — same order)
ax2 = axes[1]
ax2.bar(x - width / 2, err_q8, width, label="CQUAD8  (p = 2)",
        color="#b82020", alpha=0.92, edgecolor="white", linewidth=0.5)
ax2.bar(x + width / 2, err_q4, width, label="CQUAD4  (p = 1)",
        color="#e07b20", alpha=0.92, edgecolor="white", linewidth=0.5)
ax2.axhline(err_q8.mean(), color="#b82020", linestyle="--", linewidth=1.3,
            label=f"CQUAD8 mean = {err_q8.mean():.3f} %")
ax2.axhline(err_q4.mean(), color="#e07b20", linestyle="--", linewidth=1.3,
            label=f"CQUAD4 mean = {err_q4.mean():.3f} %")
ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=13, fontweight="bold")
ax2.set_xlabel("Mode (m, n)", fontsize=14, fontweight="bold")
ax2.set_ylabel("Relative Error [%]", fontsize=14, fontweight="bold")
ax2.legend(fontsize=11, ncol=2, frameon=True, framealpha=0.9, edgecolor="0.7")
ax2.grid(axis="y", linewidth=0.5, color="0.75")
ax2.set_xlim(-0.6, N_MODES - 0.4)
ax2.tick_params(axis="both", labelsize=12, which="both")

plt.tight_layout(h_pad=2.0)
hist_path = OUT_DIR / "validation_histogram.png"
plt.savefig(str(hist_path), dpi=200, bbox_inches="tight")
plt.close()

plt.rcParams.update(plt.rcParamsDefault)
print(f"\nHistogram saved → {hist_path}")

# ── h-convergence study ────────────────────────────────────────────────────────
print("\nRunning h-convergence study …")

conv_q4 = []
conv_q8 = []

for h in H_VALS:
    with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as tmp:
        msh = pathlib.Path(tmp.name)

    make_mesh(Lx, Ly, h, order=1, filepath=msh)
    f4, n4_, t4 = run_pycafe(msh, rho, c0, N_MODES)
    msh.unlink(missing_ok=True)
    e4 = np.abs(f4 - f_ref) / f_ref * 100.0
    conv_q4.append((h, n4_, e4.mean(), t4))

    make_mesh(Lx, Ly, h, order=2, filepath=msh)
    f8, n8_, t8 = run_pycafe(msh, rho, c0, N_MODES)
    msh.unlink(missing_ok=True)
    e8 = np.abs(f8 - f_ref) / f_ref * 100.0
    conv_q8.append((h, n8_, e8.mean(), t8))

    print(f"  h={h:.3f}  CQUAD4({n4_} nodes, {t4:.2f}s) {e4.mean():.4f}%  "
          f"CQUAD8({n8_} nodes, {t8:.2f}s) {e8.mean():.4f}%")

# ── convergence figure ─────────────────────────────────────────────────────────
h_arr  = np.array(H_VALS)
e4_arr = np.array([r[2] for r in conv_q4])
e8_arr = np.array([r[2] for r in conv_q8])
t4_arr = np.array([r[3] for r in conv_q4])
t8_arr = np.array([r[3] for r in conv_q8])

# paper-style rcParams (serif font, clean axes)
plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["DejaVu Serif", "Times New Roman", "Times"],
    "mathtext.fontset":  "dejavuserif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    1.2,
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
})

fig, ax = plt.subplots(figsize=(7, 5))

ax.loglog(h_arr, e4_arr, "s--", color="#1a6faf", markersize=8,
          linewidth=2.0, label="CQUAD4  (p = 1)", zorder=3)
ax.loglog(h_arr, e8_arr, "o-",  color="#b82020", markersize=8,
          linewidth=2.0, label="CQUAD8  (p = 2)", zorder=3)
ax.loglog(h_arr[[0, -1]],
          e4_arr[0] * (h_arr[[0, -1]] / h_arr[0]) ** 2,
          "--", color="#1a6faf", linewidth=1.0, alpha=0.55,
          label=r"$\mathcal{O}(h^{2})$ slope")
ax.loglog(h_arr[[0, -1]],
          e8_arr[0] * (h_arr[[0, -1]] / h_arr[0]) ** 4,
          "--", color="#b82020", linewidth=1.0, alpha=0.55,
          label=r"$\mathcal{O}(h^{4})$ slope")

# timing labels: CQUAD4 above, CQUAD8 below
for i in range(len(h_arr)):
    ax.annotate(f"{t4_arr[i]:.2f} s",
                xy=(h_arr[i], e4_arr[i]),
                xytext=(5, 5), textcoords="offset points",
                fontsize=8.5, fontweight="bold", color="#1a6faf")
    ax.annotate(f"{t8_arr[i]:.2f} s",
                xy=(h_arr[i], e8_arr[i]),
                xytext=(5, -13), textcoords="offset points",
                fontsize=8.5, fontweight="bold", color="#b82020")

ax.set_xlabel("Element size [m]", fontsize=14, fontweight="bold")
ax.set_ylabel("Mean relative error [%]", fontsize=14, fontweight="bold")
ax.set_title("hp-Refinement Study — CQUAD4 vs CQUAD8\n"
             "2D Rigid Rectangular Cavity, Hard-Wall BCs",
             fontsize=14, fontweight="bold", pad=10)
ax.legend(fontsize=11, frameon=True, framealpha=0.9, edgecolor="0.7")
ax.tick_params(axis="both", labelsize=12, which="both")
ax.grid(True, which="major", linewidth=0.5, color="0.75")
ax.grid(True, which="minor", linewidth=0.25, color="0.88")

plt.tight_layout()
conv_path = OUT_DIR / "validation_convergence.png"
plt.savefig(str(conv_path), dpi=200, bbox_inches="tight")
plt.close()

# reset rcParams so histogram is unaffected if figures are regenerated
plt.rcParams.update(plt.rcParamsDefault)

print(f"Convergence figure saved → {conv_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# PART II — pyCAFE vs FEniCSx
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART II — pyCAFE vs FEniCSx")
print("=" * 60)

import time as _time

from mpi4py import MPI
import dolfinx
from dolfinx import mesh as dmesh, fem
from dolfinx.fem import functionspace
from dolfinx.fem.petsc import (assemble_matrix as petsc_assemble_matrix,
                                assemble_vector as petsc_assemble_vector)
import ufl
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from slepc4py import SLEPc

from pycafe.solver.solver_helmholtz_1 import (
    solve_helmholtz_frequency_sweep,
    solve_helmholtz_single_frequency,
    build_normal_velocity_rhs,
)

# shared paper rcParams for Part II figures
_paper_rc = {
    "font.family":       "serif",
    "font.serif":        ["DejaVu Serif", "Times New Roman", "Times"],
    "mathtext.fontset":  "dejavuserif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    1.2,
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
}

_cmp_colors = {
    "analytical": "#4c566a",
    "fenics": "#0f8b8d",
    "pycafe": "#b07d2d",
    "metric_error": "#7a7f87",
    "metric_dof": "#b07d2d",
    "metric_solve": "#0f8b8d",
}

P_EXC = 1.0 + 0.0j
V_EXC = 1.0
Z_IMP = 1.0 + 0.0j
FIELD_PROBE = np.array([Lx / 2.0, Ly / 2.0])

# ── build meshes for Part II ──────────────────────────────────────────────────
# pyCAFE uses Q8 (serendipity, 8-node quad, Gmsh type 16).
# FEniCSx uses Q9 (full Lagrange 2, 9-node quad) — dolfinx.io.gmshio does not
# support type 16, so we use create_rectangle for FEniCSx.
# Both codes use the same element grid: Nx × Ny quads over [0,Lx] × [0,Ly],
# with identical corner and edge-midpoint coordinates.
Nx = max(int(round(Lx / H_BAL)), 4)
Ny = max(int(round(Ly / H_BAL)), 4)

# pyCAFE shared mesh — built once, reused for all direct-sweep cases
with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as _tmp:
    _msh_shared = pathlib.Path(_tmp.name)
make_mesh(Lx, Ly, H_BAL, order=2, filepath=_msh_shared)
nodes_shared, elements_shared, boundaries_shared = pycafe.load_mesh(
    str(_msh_shared), show_info=False, show_plot=False)
_msh_shared.unlink(missing_ok=True)

# FEniCSx Q9 mesh — same Nx × Ny grid
msh_f = dmesh.create_rectangle(
    MPI.COMM_WORLD, [[0.0, 0.0], [Lx, Ly]], [Nx, Ny],
    cell_type=dmesh.CellType.quadrilateral,
)
V = functionspace(msh_f, ("Lagrange", 2))
u_t = ufl.TrialFunction(V)
v_t = ufl.TestFunction(V)
n_dof_fenics = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
n_nodes_shared = nodes_shared.shape[0]
print(f"Mesh verification: both codes use {Nx}×{Ny} quads over "
      f"[0,{Lx}]×[0,{Ly}] m")
print(f"  pyCAFE CQUAD8 (Q8): {n_nodes_shared} nodes")
print(f"  FEniCSx Lagrange-2 (Q9): {n_dof_fenics} DOF  "
      f"(+{n_dof_fenics - n_nodes_shared} interior DOF vs Q8)")

# ── II-a  Modal comparison ─────────────────────────────────────────────────────
print("\nII-a  Modal …")

# pyCAFE CQUAD8 — timed assembly + solve separately
with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as tmp:
    msh_modal = pathlib.Path(tmp.name)
make_mesh(Lx, Ly, H_BAL, order=2, filepath=msh_modal)
nodes_m, elements_m, boundaries_m = pycafe.load_mesh(
    str(msh_modal), show_info=False, show_plot=False)
msh_modal.unlink(missing_ok=True)
bc_m = ([], [], 0.0, [], 0.0j, [], 0.0, None, 0.0)

t0 = _time.perf_counter()
sys_m = pycafe.prepare_acoustic_system(
    nodes=nodes_m, elements=elements_m, boundaries=boundaries_m,
    rho=rho, c0=c0, bc=bc_m, debug=False,
)
t_cafe_assemble = _time.perf_counter() - t0

t0 = _time.perf_counter()
raw_f, modes_red_cafe = solve_modal_acoustic_reduced(
    sys_m["K_red"], sys_m["M_red"], num_modes=N_MODES + 4)
t_cafe_solve = _time.perf_counter() - t0
freqs_cafe = raw_f[raw_f > 1.0][:N_MODES]
err_cafe   = np.abs(freqs_cafe - f_ref) / f_ref * 100.0

# FEniCSx — timed assembly + solve separately
a_K  = ufl.inner(ufl.grad(u_t), ufl.grad(v_t)) * ufl.dx
a_M  = (1.0 / c0**2) * ufl.inner(u_t, v_t) * ufl.dx

t0 = _time.perf_counter()
K_pm = petsc_assemble_matrix(fem.form(a_K)); K_pm.assemble()
M_pm = petsc_assemble_matrix(fem.form(a_M)); M_pm.assemble()
t_fenics_assemble = _time.perf_counter() - t0

t0  = _time.perf_counter()
eps = SLEPc.EPS().create(MPI.COMM_WORLD)
eps.setOperators(K_pm, M_pm)
eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
eps.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL)
eps.setDimensions(nev=N_MODES + 6)
eps.setFromOptions()
eps.solve()
t_fenics_solve = _time.perf_counter() - t0

xr, xi = K_pm.createVecs()
freqs_fenics = []
for i in range(eps.getConverged()):
    lam = float(np.real(eps.getEigenpair(i, xr, xi)))
    if lam > 1000.0:
        freqs_fenics.append(np.sqrt(lam) / (2 * np.pi))
    if len(freqs_fenics) == N_MODES:
        break
freqs_fenics = np.array(freqs_fenics)
err_fenics   = np.abs(freqs_fenics - f_ref) / f_ref * 100.0
modal_total_cafe = t_cafe_assemble + t_cafe_solve
modal_total_fenics = t_fenics_assemble + t_fenics_solve
dof_ratio = n_nodes_q8 / n_dof_fenics
solve_ratio = t_cafe_solve / t_fenics_solve
dof_reduction_pct = (1.0 - dof_ratio) * 100.0
solve_speedup = t_fenics_solve / t_cafe_solve if t_cafe_solve > 0.0 else np.nan
modal_err_gap = np.abs(err_cafe.mean() - err_fenics.mean())

print(f"  pyCAFE  assemble={t_cafe_assemble:.3f}s  solve={t_cafe_solve:.3f}s  "
      f"mean err={err_cafe.mean():.4f}%")
print(f"  FEniCSx assemble={t_fenics_assemble:.3f}s  solve={t_fenics_solve:.3f}s  "
      f"mean err={err_fenics.mean():.4f}%")
print(f"  Modal mean-error difference = {modal_err_gap:.4f} percentage points")
print(f"  pyCAFE uses {dof_reduction_pct:.1f}% fewer DOFs "
      f"({n_nodes_q8} vs {n_dof_fenics})")
print(f"  Eigensolve time ratio (FEniCSx / pyCAFE) = {solve_speedup:.2f} "
      f"({t_fenics_solve:.3f}s / {t_cafe_solve:.3f}s)")

# mode-shape data
fenics_mode_vecs = []
for i in range(eps.getConverged()):
    lam = float(np.real(eps.getEigenpair(i, xr, xi)))
    if lam > 1000.0:
        fenics_mode_vecs.append(xr.getArray().copy())
    if len(fenics_mode_vecs) == N_MODES:
        break

mode_shape_ids = [0, 3, 5, 6]
mode_shape_labels = [labels[i] for i in mode_shape_ids]

def _normalize_mode(mode):
    mode = np.real(np.asarray(mode, dtype=float)).copy()
    amp = np.max(np.abs(mode))
    if amp > 0.0:
        mode /= amp
    idx_ref = int(np.argmax(np.abs(mode)))
    if mode[idx_ref] < 0.0:
        mode *= -1.0
    return mode

def _align_mode_sign_by_location(mode_ref, coords_ref, mode_cmp, coords_cmp):
    """Align the global sign of two modal fields using a common physical point."""
    idx_ref = int(np.argmax(np.abs(mode_ref)))
    ref_xy = coords_ref[idx_ref]
    d2 = np.sum((coords_cmp - ref_xy) ** 2, axis=1)
    idx_cmp = int(np.argmin(d2))
    if mode_ref[idx_ref] * mode_cmp[idx_cmp] < 0.0:
        return -mode_cmp
    return mode_cmp

coords_cafe = nodes_m[:, :2]
coords_fenics = V.tabulate_dof_coordinates()[:, :2]
tri_cafe = mtri.Triangulation(coords_cafe[:, 0], coords_cafe[:, 1])
tri_fenics = mtri.Triangulation(coords_fenics[:, 0], coords_fenics[:, 1])

mode_shapes_cafe = []
mode_shapes_fenics = []
for midx in mode_shape_ids:
    mode_full_cafe = expand_mode_to_full(
        modes_red_cafe[:, midx],
        sys_m["idx_free"],
        sys_m["p0_nodes"],
        nodes_m.shape[0],
    )
    cafe_norm = _normalize_mode(mode_full_cafe)
    fenics_norm = _normalize_mode(fenics_mode_vecs[midx])
    fenics_norm = _align_mode_sign_by_location(
        cafe_norm, coords_cafe, fenics_norm, coords_fenics
    )
    mode_shapes_cafe.append(cafe_norm)
    mode_shapes_fenics.append(fenics_norm)

plt.rcParams.update(_paper_rc)
fig, axes = plt.subplots(len(mode_shape_ids), 2, figsize=(10, 12), constrained_layout=True)
if len(mode_shape_ids) == 1:
    axes = np.array([axes])

last_mappable = None
for row, midx in enumerate(mode_shape_ids):
    ax_l = axes[row, 0]
    ax_r = axes[row, 1]
    cafe_plot = ax_l.tricontourf(
        tri_cafe, mode_shapes_cafe[row], levels=25, cmap="RdBu_r", vmin=-1.0, vmax=1.0
    )
    ax_r.tricontourf(
        tri_fenics, mode_shapes_fenics[row], levels=25, cmap="RdBu_r", vmin=-1.0, vmax=1.0
    )
    last_mappable = cafe_plot
    ax_l.set_ylabel(f"Mode {midx + 1}\n{mode_shape_labels[row]}", fontsize=12, fontweight="bold")
    for ax_ in (ax_l, ax_r):
        ax_.set_aspect("equal")
        ax_.set_xlim(0.0, Lx)
        ax_.set_ylim(0.0, Ly)
        ax_.set_xticks([0.0, 0.5, 1.0])
        ax_.set_yticks([0.0, 0.25, 0.5])
        ax_.tick_params(labelsize=10)
    if row == 0:
        ax_l.set_title("pyCAFE CQUAD8", fontsize=13, fontweight="bold")
        ax_r.set_title("FEniCSx Q2", fontsize=13, fontweight="bold")
    if row == len(mode_shape_ids) - 1:
        ax_l.set_xlabel("x [m]", fontsize=12, fontweight="bold")
        ax_r.set_xlabel("x [m]", fontsize=12, fontweight="bold")
    ax_l.set_facecolor("#f8f6f1")
    ax_r.set_facecolor("#f8f6f1")

cbar = fig.colorbar(last_mappable, ax=axes.ravel().tolist(), shrink=0.95, pad=0.02)
cbar.set_ticks([-1.0, 1.0])
cbar.set_ticklabels(["-1", "1"])
cbar.ax.tick_params(labelsize=11)
mode_shape_path = OUT_DIR / "comparison_modeshapes.png"
plt.savefig(str(mode_shape_path), dpi=220, bbox_inches="tight")
plt.close()
plt.rcParams.update(plt.rcParamsDefault)
print(f"  Figure saved → {mode_shape_path}")

# Figure II-a: left = frequency histogram, right = normalized efficiency summary
plt.rcParams.update(_paper_rc)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

x_m = np.arange(N_MODES)
w   = 0.28
ax  = axes[0]
ax.bar(x_m - w,  f_ref,        w, label="Analytical",
       color=_cmp_colors["analytical"], alpha=0.92, edgecolor="white")
ax.bar(x_m,      freqs_fenics, w, label="FEniCSx Q2",
       color=_cmp_colors["fenics"], alpha=0.92, edgecolor="white")
ax.bar(x_m + w,  freqs_cafe,   w, label="pyCAFE CQUAD8",
       color=_cmp_colors["pycafe"], alpha=0.92, edgecolor="white")
ax.set_xticks(x_m); ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
ax.set_ylabel("Eigenfrequency [Hz]", fontsize=13, fontweight="bold")
ax.set_title("Modal Comparison — 4 Hard Walls\npyCAFE CQUAD8 vs FEniCSx Q2",
             fontsize=13, fontweight="bold", pad=8)
ax.legend(fontsize=11, frameon=True, framealpha=0.9, edgecolor="0.7")
ax.grid(axis="y", linewidth=0.5, color="0.75")
ax.tick_params(labelsize=11)

ax2   = axes[1]
metric_labels = ["Mean error", "DOF count", "Eigensolve time"]
metric_ratios = [err_cafe.mean() / err_fenics.mean(), dof_ratio, solve_ratio]
x_t = np.arange(len(metric_labels))
bars = ax2.bar(x_t, metric_ratios, width=0.58,
               color=[_cmp_colors["metric_error"],
                      _cmp_colors["metric_dof"],
                      _cmp_colors["metric_solve"]],
               alpha=0.92, edgecolor="white")
ax2.axhline(1.0, color="0.35", linestyle="--", linewidth=1.2,
            label="FEniCSx baseline = 1")
for idx, bar in enumerate(bars):
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.035,
        f"{metric_ratios[idx]:.3f}",
        ha="center", va="bottom", fontsize=10, fontweight="bold",
    )
ax2.set_xticks(x_t)
ax2.set_xticklabels(metric_labels, fontsize=12, fontweight="bold")
ax2.set_ylabel("pyCAFE / FEniCSx", fontsize=13, fontweight="bold")
ax2.set_ylim(0.0, max(1.25, max(metric_ratios) + 0.18))
ax2.set_title("Modal Comparison Summary — 4 Hard Walls\nNormalized error, DOF, and solve time",
              fontsize=13, fontweight="bold", pad=8)
ax2.legend(fontsize=11, frameon=True, framealpha=0.9, edgecolor="0.7")
ax2.grid(axis="y", linewidth=0.5, color="0.75")
ax2.tick_params(labelsize=11)
ax2.text(
    0.03, 0.97,
    f"pyCAFE: {n_nodes_q8} DOF, solve {t_cafe_solve:.3f}s, total {modal_total_cafe:.3f}s\n"
    f"FEniCSx: {n_dof_fenics} DOF, solve {t_fenics_solve:.3f}s, total {modal_total_fenics:.3f}s",
    transform=ax2.transAxes, va="top", ha="left", fontsize=9.8,
    bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "0.75"},
)

plt.tight_layout()
modal_path = OUT_DIR / "comparison_modal.png"
plt.savefig(str(modal_path), dpi=200, bbox_inches="tight")
plt.close()
plt.rcParams.update(plt.rcParamsDefault)
print(f"  Figure saved → {modal_path}")

# ── II-a (zero-pressure)  All walls p = 0 ────────────────────────────────────
print("\nII-a (zero-pressure)  All walls p = 0 …")

analytic_zp = analytical_freqs_zero_pressure(c0, Lx, Ly, N_MODES)
f_ref_zp    = np.array([x[0] for x in analytic_zp])
labels_zp   = [f"({x[1]},{x[2]})" for x in analytic_zp]

# pyCAFE: zero-pressure Dirichlet on all 4 walls, reuse same mesh
bc_m_zp = (["bottom", "right", "top", "left"], [], 0.0, [], 0.0j, [], 0.0, None, 0.0)
sys_m_zp = pycafe.prepare_acoustic_system(
    nodes=nodes_m, elements=elements_m, boundaries=boundaries_m,
    rho=rho, c0=c0, bc=bc_m_zp, debug=False,
)
t0 = _time.perf_counter()
raw_f_zp, modes_red_cafe_zp = solve_modal_acoustic_reduced(
    sys_m_zp["K_red"], sys_m_zp["M_red"], num_modes=N_MODES + 4)
t_cafe_solve_zp = _time.perf_counter() - t0
mask_zp       = raw_f_zp > 1.0
freqs_cafe_zp = raw_f_zp[mask_zp][:N_MODES]
modes_red_cafe_zp = modes_red_cafe_zp[:, mask_zp][:, :N_MODES]
err_cafe_zp   = np.abs(freqs_cafe_zp - f_ref_zp) / f_ref_zp * 100.0

# FEniCSx: reduce K_pm / M_pm to free (interior) DOFs, then scipy eigsh
def _pm_to_scipy(A):
    ai, aj, av = A.getValuesCSR()
    return sp.csr_matrix((av, aj, ai), shape=A.getSize())

K_sc_m = _pm_to_scipy(K_pm)
M_sc_m = _pm_to_scipy(M_pm)

_all_bdry_facets = dmesh.locate_entities_boundary(
    msh_f, msh_f.topology.dim - 1, lambda x: np.ones(x.shape[1], dtype=bool))
_bdry_dofs_zp = fem.locate_dofs_topological(V, msh_f.topology.dim - 1, _all_bdry_facets)
_free_dofs_zp = np.setdiff1d(np.arange(n_dof_fenics), _bdry_dofs_zp)

K_red_f_zp = K_sc_m[np.ix_(_free_dofs_zp, _free_dofs_zp)]
M_red_f_zp = M_sc_m[np.ix_(_free_dofs_zp, _free_dofs_zp)]

t0 = _time.perf_counter()
vals_zp_f, vecs_zp_f = spla.eigsh(
    K_red_f_zp, k=N_MODES + 4, M=M_red_f_zp, sigma=0.0, which="LM")
t_fenics_solve_zp = _time.perf_counter() - t0

sort_idx_zp  = np.argsort(np.real(vals_zp_f))
vals_zp_f    = np.real(vals_zp_f[sort_idx_zp])
vecs_zp_f    = vecs_zp_f[:, sort_idx_zp]
pos_mask_zp  = vals_zp_f > 0.0
vals_zp_f    = vals_zp_f[pos_mask_zp][:N_MODES]
vecs_zp_f    = vecs_zp_f[:, pos_mask_zp][:, :N_MODES]
freqs_fenics_zp = np.sqrt(vals_zp_f) / (2 * np.pi)
err_fenics_zp   = np.abs(freqs_fenics_zp - f_ref_zp) / f_ref_zp * 100.0

# expand FEniCSx reduced eigenvectors to full DOF space
def _expand_fenics_zp(vec_red):
    full = np.zeros(n_dof_fenics)
    full[_free_dofs_zp] = np.real(vec_red)
    return full

fenics_mode_vecs_zp = [_expand_fenics_zp(vecs_zp_f[:, i]) for i in range(N_MODES)]

dof_ratio_zp    = sys_m_zp["K_red"].shape[0] / len(_free_dofs_zp)
solve_ratio_zp  = t_cafe_solve_zp / t_fenics_solve_zp if t_fenics_solve_zp > 0 else np.nan
err_ratio_zp    = err_cafe_zp.mean() / err_fenics_zp.mean() if err_fenics_zp.mean() > 0 else np.nan

print(f"  pyCAFE  solve={t_cafe_solve_zp:.3f}s  mean err={err_cafe_zp.mean():.4f}%  "
      f"red. DOF={sys_m_zp['K_red'].shape[0]}")
print(f"  FEniCSx solve={t_fenics_solve_zp:.3f}s  mean err={err_fenics_zp.mean():.4f}%  "
      f"free DOF={len(_free_dofs_zp)}")

print(f"\n{'Mode':>8}  {'Analytical [Hz]':>16}  {'CQUAD8 [Hz]':>13}  "
      f"{'err%':>7}  {'FEniCSx [Hz]':>13}  {'err%':>7}")
print("-" * 74)
for i in range(N_MODES):
    print(f"  {labels_zp[i]:>6}  {f_ref_zp[i]:>16.4f}  {freqs_cafe_zp[i]:>13.4f}  "
          f"{err_cafe_zp[i]:>6.4f}%  {freqs_fenics_zp[i]:>13.4f}  {err_fenics_zp[i]:>6.4f}%")

# ── mode shapes — zero-pressure ───────────────────────────────────────────────
mode_shape_ids_zp = [0, 1, 3, 5]
mode_shapes_cafe_zp = []
mode_shapes_fenics_zp = []
for midx in mode_shape_ids_zp:
    mode_full = expand_mode_to_full(
        modes_red_cafe_zp[:, midx],
        sys_m_zp["idx_free"],
        sys_m_zp["p0_nodes"],
        nodes_m.shape[0],
    )
    cafe_n  = _normalize_mode(mode_full)
    fen_n   = _normalize_mode(fenics_mode_vecs_zp[midx])
    fen_n   = _align_mode_sign_by_location(cafe_n, coords_cafe, fen_n, coords_fenics)
    mode_shapes_cafe_zp.append(cafe_n)
    mode_shapes_fenics_zp.append(fen_n)

plt.rcParams.update(_paper_rc)
fig_ms, axes_ms = plt.subplots(
    len(mode_shape_ids_zp), 2, figsize=(10, 12), constrained_layout=True)
last_mp = None
for row, midx in enumerate(mode_shape_ids_zp):
    ax_l = axes_ms[row, 0]
    ax_r = axes_ms[row, 1]
    mp = ax_l.tricontourf(
        tri_cafe, mode_shapes_cafe_zp[row], levels=25, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    ax_r.tricontourf(
        tri_fenics, mode_shapes_fenics_zp[row], levels=25, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    last_mp = mp
    ax_l.set_ylabel(f"Mode {midx + 1}\n{labels_zp[midx]}", fontsize=12, fontweight="bold")
    for ax_ in (ax_l, ax_r):
        ax_.set_aspect("equal"); ax_.set_xlim(0.0, Lx); ax_.set_ylim(0.0, Ly)
        ax_.set_xticks([0.0, 0.5, 1.0]); ax_.set_yticks([0.0, 0.25, 0.5])
        ax_.tick_params(labelsize=10)
        ax_.set_facecolor("#f8f6f1")
    if row == 0:
        ax_l.set_title("pyCAFE CQUAD8", fontsize=13, fontweight="bold")
        ax_r.set_title("FEniCSx Q2",    fontsize=13, fontweight="bold")
    if row == len(mode_shape_ids_zp) - 1:
        ax_l.set_xlabel("x [m]", fontsize=12, fontweight="bold")
        ax_r.set_xlabel("x [m]", fontsize=12, fontweight="bold")

fig_ms.suptitle(
    r"Mode Shapes — 4 Pressure-Release Walls ($p=0$)  |  pyCAFE vs FEniCSx",
    fontsize=13, fontweight="bold", y=1.01)
cbar_ms = fig_ms.colorbar(last_mp, ax=axes_ms.ravel().tolist(), shrink=0.95, pad=0.02)
cbar_ms.set_ticks([-1.0, 1.0]); cbar_ms.set_ticklabels(["-1", "1"])
cbar_ms.ax.tick_params(labelsize=11)
ms_zp_path = OUT_DIR / "comparison_modeshapes_zero_pressure.png"
plt.savefig(str(ms_zp_path), dpi=220, bbox_inches="tight")
plt.close()
plt.rcParams.update(plt.rcParamsDefault)
print(f"  Figure saved → {ms_zp_path}")

# ── 2-panel summary figure — zero-pressure ────────────────────────────────────
plt.rcParams.update(_paper_rc)
fig_zp, axes_zp = plt.subplots(1, 2, figsize=(13, 5))

x_zp = np.arange(N_MODES)
w_zp = 0.28
ax_zp = axes_zp[0]
ax_zp.bar(x_zp - w_zp, f_ref_zp,       w_zp, label="Analytical",    color=_cmp_colors["analytical"], alpha=0.92, edgecolor="white")
ax_zp.bar(x_zp,        freqs_fenics_zp, w_zp, label="FEniCSx Q2",    color=_cmp_colors["fenics"],     alpha=0.92, edgecolor="white")
ax_zp.bar(x_zp + w_zp, freqs_cafe_zp,   w_zp, label="pyCAFE CQUAD8", color=_cmp_colors["pycafe"],     alpha=0.92, edgecolor="white")
ax_zp.set_xticks(x_zp); ax_zp.set_xticklabels(labels_zp, fontsize=11, fontweight="bold")
ax_zp.set_ylabel("Eigenfrequency [Hz]", fontsize=13, fontweight="bold")
ax_zp.set_title(r"Modal Comparison — 4 Pressure-Release Walls ($p=0$)"
                "\npyCAFE CQUAD8 vs FEniCSx Q2", fontsize=13, fontweight="bold", pad=8)
ax_zp.legend(fontsize=11, frameon=True, framealpha=0.9, edgecolor="0.7")
ax_zp.grid(axis="y", linewidth=0.5, color="0.75")
ax_zp.tick_params(labelsize=11)

ax2_zp = axes_zp[1]
metric_labels_zp = ["Mean error", "DOF count", "Eigensolve time"]
metric_ratios_zp = [err_ratio_zp, dof_ratio_zp, solve_ratio_zp]
x_t_zp = np.arange(len(metric_labels_zp))
bars_zp = ax2_zp.bar(x_t_zp, metric_ratios_zp, width=0.58,
                     color=[_cmp_colors["metric_error"],
                            _cmp_colors["metric_dof"],
                            _cmp_colors["metric_solve"]],
                     alpha=0.92, edgecolor="white")
ax2_zp.axhline(1.0, color="0.35", linestyle="--", linewidth=1.2,
               label="FEniCSx baseline = 1")
for idx, bar in enumerate(bars_zp):
    ax2_zp.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.035,
        f"{metric_ratios_zp[idx]:.3f}",
        ha="center", va="bottom", fontsize=10, fontweight="bold",
    )
ax2_zp.set_xticks(x_t_zp)
ax2_zp.set_xticklabels(metric_labels_zp, fontsize=12, fontweight="bold")
ax2_zp.set_ylabel("pyCAFE / FEniCSx", fontsize=13, fontweight="bold")
ax2_zp.set_ylim(0.0, max(1.25, max(metric_ratios_zp) + 0.18))
ax2_zp.set_title(r"Modal Summary — 4 Pressure-Release Walls ($p=0$)"
                 "\nNormalized error, DOF, and solve time",
                 fontsize=13, fontweight="bold", pad=8)
ax2_zp.legend(fontsize=11, frameon=True, framealpha=0.9, edgecolor="0.7")
ax2_zp.grid(axis="y", linewidth=0.5, color="0.75")
ax2_zp.tick_params(labelsize=11)
ax2_zp.text(
    0.03, 0.97,
    f"pyCAFE: {sys_m_zp['K_red'].shape[0]} DOF, solve {t_cafe_solve_zp:.3f}s\n"
    f"FEniCSx: {len(_free_dofs_zp)} DOF, solve {t_fenics_solve_zp:.3f}s",
    transform=ax2_zp.transAxes, va="top", ha="left", fontsize=9.8,
    bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "0.75"},
)
plt.tight_layout()
modal_zp_path = OUT_DIR / "comparison_modal_zero_pressure.png"
plt.savefig(str(modal_zp_path), dpi=200, bbox_inches="tight")
plt.close()
plt.rcParams.update(plt.rcParamsDefault)
print(f"  Figure saved → {modal_zp_path}")

# ── II-b  Direct Helmholtz sweep — multiple probes ────────────────────────────
print("\nII-b  Direct Helmholtz sweep …")

sweep_freqs = np.arange(300.0, 701.0, 1.0)
v_n         = 1.0   # normal velocity on left wall [m/s]

# three probe locations
probes = [
    ("right-wall centre",  np.array([Lx,       Ly / 2.0])),
    ("top-right corner",   np.array([Lx,       Ly      ])),
    ("cavity centre",      np.array([Lx / 2.0, Ly / 2.0])),
]
probe_colors = ["#355070", "#6d597a", "#bc6c25"]

# ── pyCAFE: assemble once, extract all probes (reuse shared mesh) ─────────────
nodes_s, elements_s, boundaries_s = nodes_shared, elements_shared, boundaries_shared

bc_sweep = ([], [], 0.0, [], 0.0j, ["left"], v_n, None, 0.0)
sys_s = pycafe.prepare_acoustic_system(
    nodes=nodes_s, elements=elements_s, boundaries=boundaries_s,
    rho=rho, c0=c0, bc=bc_sweep, debug=False,
)

t0_s = _time.perf_counter()
P_red = solve_helmholtz_frequency_sweep(
    K_red=sys_s["K_red"], M_red=sys_s["M_red"], C_red=sys_s["C_red"],
    frequencies=sweep_freqs,
    pressure_nodes_red=sys_s["pressure_nodes_red"],
    pressure_values=sys_s["pressure_values"],
    nodes=nodes_s,
    boundary_velocity_nodes=sys_s["bc_velocity"],
    idx_free=sys_s["idx_free"],
    rho=rho, v_n=sys_s["value_velocity_normal"],
    boundaries=boundaries_s, elements=elements_s,
)
t_cafe_sweep = _time.perf_counter() - t0_s

P_full_cafe = np.zeros((nodes_s.shape[0], len(sweep_freqs)), dtype=complex)
P_full_cafe[sys_s["idx_free"], :] = P_red

p_cafe = []
for _, pxy in probes:
    d2 = (nodes_s[:, 0] - pxy[0])**2 + (nodes_s[:, 1] - pxy[1])**2
    p_cafe.append(P_full_cafe[int(np.argmin(d2)), :])
print(f"  pyCAFE sweep done  ({t_cafe_sweep:.2f} s)")

# ── FEniCSx: assemble once, solve per frequency, extract all probes ───────────
def _left(x): return np.isclose(x[0], 0.0)
def _right(x): return np.isclose(x[0], Lx)
facet_dim   = msh_f.topology.dim - 1
left_facets = dmesh.locate_entities_boundary(msh_f, facet_dim, _left)
right_facets = dmesh.locate_entities_boundary(msh_f, facet_dim, _right)
all_facets = np.hstack([left_facets, right_facets])
all_tags = np.hstack([
    np.full(left_facets.shape, 1, dtype=np.int32),
    np.full(right_facets.shape, 2, dtype=np.int32),
])
srt = np.argsort(all_facets)
facet_tags = dmesh.meshtags(msh_f, facet_dim, all_facets[srt], all_tags[srt])
ds = ufl.Measure("ds", domain=msh_f, subdomain_data=facet_tags)

a_Ks  = ufl.inner(ufl.grad(u_t), ufl.grad(v_t)) * ufl.dx
a_Ms  = ufl.inner(u_t, v_t) * ufl.dx
L_sur = ufl.inner(fem.Constant(msh_f, dolfinx.default_scalar_type(1.0)), v_t) * ds(1)

K_sp_m = petsc_assemble_matrix(fem.form(a_Ks)); K_sp_m.assemble()
M_sp_m = petsc_assemble_matrix(fem.form(a_Ms)); M_sp_m.assemble()
b_sp_v = petsc_assemble_vector(fem.form(L_sur)); b_sp_v.assemble()

def _to_scipy(A):
    ai, aj, av = A.getValuesCSR()
    return sp.csr_matrix((av, aj, ai), shape=A.getSize())

K_sc = _to_scipy(K_sp_m).astype(complex)
M_sc = _to_scipy(M_sp_m).astype(complex)
b_sc = b_sp_v.getArray().astype(complex).copy()

coords_f = V.tabulate_dof_coordinates()[:, :2]
probe_dofs = []
for _, pxy in probes:
    d2_f = (coords_f[:, 0] - pxy[0])**2 + (coords_f[:, 1] - pxy[1])**2
    probe_dofs.append(int(np.argmin(d2_f)))

n_pr = len(probes)
p_fenics = [np.zeros(len(sweep_freqs), dtype=complex) for _ in range(n_pr)]
t0_s = _time.perf_counter()
for i_f, f in enumerate(sweep_freqs):
    omega  = 2 * np.pi * f
    A_sys  = (K_sc - (omega / c0)**2 * M_sc).tocsc()
    p_all  = spla.spsolve(A_sys, (1j * omega * rho * v_n) * b_sc)
    for j, dof in enumerate(probe_dofs):
        p_fenics[j][i_f] = p_all[dof]
t_fenics_sweep = _time.perf_counter() - t0_s
print(f"  FEniCSx sweep done ({t_fenics_sweep:.2f} s)")
probe_rel_diff = []
for j in range(n_pr):
    mag_c = np.abs(p_cafe[j])
    mag_f = np.abs(p_fenics[j])
    denom = max(np.linalg.norm(mag_f), 1e-12)
    probe_rel_diff.append(np.linalg.norm(mag_c - mag_f) / denom * 100.0)
mean_probe_rel_diff = float(np.mean(probe_rel_diff))
for j, (pname, _) in enumerate(probes):
    print(f"  FRF magnitude mismatch at {pname}: {probe_rel_diff[j]:.3f}%")

# Figure II-b: agreement-focused FRF overlay, timing moved to inset text
plt.rcParams.update(_paper_rc)
fig, ax = plt.subplots(figsize=(10, 6))

for j, (pname, pxy) in enumerate(probes):
    coord_str = f"({pxy[0]:.2f}, {pxy[1]:.2f}) m"
    ax.semilogy(sweep_freqs, np.abs(p_cafe[j]),
                color=probe_colors[j], linewidth=2.0,
                linestyle="-",
                label=f"pyCAFE — {pname}  {coord_str}")
    ax.semilogy(sweep_freqs, np.abs(p_fenics[j]),
                color=probe_colors[j], linewidth=1.6,
                linestyle="--",
                label=f"FEniCSx — {pname}  {coord_str}")

ax.set_xlabel("Frequency [Hz]", fontsize=13, fontweight="bold")
ax.set_ylabel("|p| [Pa]", fontsize=13, fontweight="bold")
ax.set_title(
    "Direct Helmholtz Sweep — Response Agreement\n"
    "pyCAFE CQUAD8 vs FEniCSx Q2",
    fontsize=13, fontweight="bold", pad=10,
)
ax.legend(fontsize=10, frameon=True, framealpha=0.9, edgecolor="0.7",
          ncol=2)
ax.grid(True, which="major", linewidth=0.5, color="0.75")
ax.grid(True, which="minor", linewidth=0.25, color="0.88")
ax.tick_params(labelsize=11)
ax.text(
    0.985, 0.985,
    f"Left-wall piston: $v_n$ = {v_n:.1f} m/s\n"
    f"Mean FRF mismatch = {mean_probe_rel_diff:.3f}%\n"
    f"pyCAFE: {t_cafe_sweep:.2f} s\n"
    f"FEniCSx: {t_fenics_sweep:.2f} s",
    transform=ax.transAxes, ha="right", va="top", fontsize=10,
    bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "0.75"},
)

plt.tight_layout()
direct_path = OUT_DIR / "comparison_direct.png"
plt.savefig(str(direct_path), dpi=200, bbox_inches="tight")
plt.close()
plt.rcParams.update(plt.rcParamsDefault)
print(f"  Figure saved → {direct_path}")

# ── II-c  Probe responses for different boundary-condition types ──────────────
print("\nII-c  Probe responses by boundary-condition type …")

left_dofs_f  = fem.locate_dofs_geometrical(V, _left)
right_dofs_f = fem.locate_dofs_geometrical(V, _right)
B_right_pm   = petsc_assemble_matrix(fem.form(ufl.inner(u_t, v_t) * ds(2))); B_right_pm.assemble()
B_right_sc   = _to_scipy(B_right_pm).astype(complex)

field_cases = [
    {
        "key": "velocity",
        "title": "Prescribed normal velocity on left wall",
        "bc": ([], [], 0.0, [], 0.0j, ["left"], V_EXC, None, 0.0),
        "fenics_pressure_dofs": np.array([], dtype=int),
        "fenics_pressure_value": 0.0j,
        "fenics_velocity_rhs": b_sc,
        "fenics_impedance_matrix": None,
        "description": r"Left wall: $v_n=1$ m/s; other walls rigid.",
    },
    {
        "key": "impedance",
        "title": "Pressure excitation with impedance termination",
        "bc": ([], ["left"], P_EXC, ["right"], Z_IMP, [], 0.0, None, 0.0),
        "fenics_pressure_dofs": left_dofs_f,
        "fenics_pressure_value": P_EXC,
        "fenics_velocity_rhs": None,
        "fenics_impedance_matrix": B_right_sc,
        "description": r"Left wall: $p=1$ Pa; right wall: $Z=1$; top/bottom rigid.",
    },
]

def _solve_fenics_direct_field(
    omega,
    pressure_dofs,
    pressure_value,
    velocity_rhs,
    impedance_matrix,
):
    # Build full dynamic stiffness matrix (LIL for efficient column access)
    A_sys = (K_sc - (omega / c0) ** 2 * M_sc).tolil().astype(complex)
    if impedance_matrix is not None:
        A_sys = A_sys + (1j * omega / (Z_IMP * c0)) * impedance_matrix.tolil()

    rhs = np.zeros(A_sys.shape[0], dtype=complex)
    if velocity_rhs is not None:
        rhs += (1j * omega * rho * V_EXC) * velocity_rhs

    pressure_dofs = np.asarray(pressure_dofs, dtype=int)
    if pressure_dofs.size == 0:
        return spla.spsolve(A_sys.tocsc(), rhs)

    # ── FEniCSx-style symmetric elimination ──────────────────────────────────
    # Mirrors apply_lifting + zeroRowsColumns + set_bc in dolfinx.fem.petsc.
    p_vals = np.full(pressure_dofs.shape, pressure_value, dtype=complex)

    # Step 1 — apply_lifting: rhs -= A[:, d] * g_d  then zero column d
    for d, gd in zip(pressure_dofs, p_vals):
        col = np.array(A_sys.getcol(d).todense(), dtype=complex).ravel()
        rhs -= col * gd
        A_sys[:, d] = 0

    # Step 2 — zeroRows + unit diagonal
    for d in pressure_dofs:
        A_sys[d, :] = 0
        A_sys[d, d] = 1.0

    # Step 3 — set_bc on RHS
    for d, gd in zip(pressure_dofs, p_vals):
        rhs[d] = gd

    return spla.spsolve(A_sys.tocsc(), rhs)

def _save_probe_comparison(case_key, case_title, quantity_key, quantity_label, vals_cafe, vals_fenics):
    plt.rcParams.update(_paper_rc)
    fig, ax = plt.subplots(figsize=(8.6, 4.8), constrained_layout=True)
    if quantity_key == "abs":
        ax.semilogy(sweep_freqs, vals_cafe, color=_cmp_colors["pycafe"], linewidth=2.0, label="pyCAFE CQUAD8")
        ax.semilogy(sweep_freqs, vals_fenics, color=_cmp_colors["fenics"], linewidth=1.8, linestyle="--", label="FEniCSx Q2")
    else:
        ax.plot(sweep_freqs, vals_cafe, color=_cmp_colors["pycafe"], linewidth=2.0, label="pyCAFE CQUAD8")
        ax.plot(sweep_freqs, vals_fenics, color=_cmp_colors["fenics"], linewidth=1.8, linestyle="--", label="FEniCSx Q2")
        if quantity_key == "phase":
            ax.set_ylim(-np.pi, np.pi)
            ax.set_yticks([-np.pi, 0.0, np.pi])
            ax.set_yticklabels([r"$-\pi$", "0", r"$\pi$"])
    ax.set_xlabel("Frequency [Hz]", fontsize=12, fontweight="bold")
    ax.set_ylabel(quantity_label, fontsize=12, fontweight="bold")
    ax.set_title(
        f"{case_title}\nProbe at ({FIELD_PROBE[0]:.2f}, {FIELD_PROBE[1]:.2f}) m",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10, frameon=True, framealpha=0.9, edgecolor="0.7")
    ax.grid(True, which="major", linewidth=0.5, color="0.75")
    ax.grid(True, which="minor", linewidth=0.25, color="0.88")
    ax.tick_params(labelsize=10)
    out_path = OUT_DIR / f"comparison_direct_{case_key}_{quantity_key}.png"
    plt.savefig(str(out_path), dpi=220, bbox_inches="tight")
    plt.close()
    plt.rcParams.update(plt.rcParamsDefault)
    print(f"  Figure saved → {out_path}")

for case in field_cases:
    # reuse shared mesh — same nodes for pyCAFE and FEniCSx
    nodes_case, elements_case, boundaries_case = (
        nodes_shared, elements_shared, boundaries_shared)

    sys_case = pycafe.prepare_acoustic_system(
        nodes=nodes_case,
        elements=elements_case,
        boundaries=boundaries_case,
        rho=rho,
        c0=c0,
        bc=case["bc"],
        debug=False,
    )

    probe_idx_cafe = int(np.argmin(np.sum((nodes_case[:, :2] - FIELD_PROBE) ** 2, axis=1)))
    probe_idx_fenics = int(np.argmin(np.sum((coords_f - FIELD_PROBE) ** 2, axis=1)))
    resp_cafe = np.zeros(len(sweep_freqs), dtype=complex)
    resp_fenics = np.zeros(len(sweep_freqs), dtype=complex)

    for i_f, f in enumerate(sweep_freqs):
        omega_f = 2 * np.pi * f
        if case["bc"][5] and case["bc"][6] != 0.0:
            vel_nodes_red, vel_vals = build_normal_velocity_rhs(
                nodes=nodes_case,
                boundary_velocity_nodes=sys_case["bc_velocity"],
                idx_free=sys_case["idx_free"],
                rho=rho,
                omega=omega_f,
                v_n=sys_case["value_velocity_normal"],
                boundaries=boundaries_case,
                elements=elements_case,
            )
        else:
            vel_nodes_red, vel_vals = None, None

        p_case_red = solve_helmholtz_single_frequency(
            sys_case["K_red"],
            sys_case["M_red"],
            sys_case["C_red"],
            omega_f,
            sys_case["pressure_nodes_red"],
            sys_case["pressure_values"],
            vel_nodes_red,
            vel_vals,
        )
        p_case_cafe = expand_to_full(
            p_case_red, sys_case["idx_free"], sys_case["p0_nodes"], nodes_case.shape[0]
        )
        resp_cafe[i_f] = p_case_cafe[probe_idx_cafe]

        p_case_fenics = _solve_fenics_direct_field(
            omega_f,
            case["fenics_pressure_dofs"],
            case["fenics_pressure_value"],
            case["fenics_velocity_rhs"],
            case["fenics_impedance_matrix"],
        )
        resp_fenics[i_f] = p_case_fenics[probe_idx_fenics]

    quantities = {
        "abs": ("$|p|$ [Pa]", np.abs(resp_cafe), np.abs(resp_fenics)),
        "real": (r"$\Re(p)$ [Pa]", np.real(resp_cafe), np.real(resp_fenics)),
        "phase": ("Phase [rad]", np.angle(resp_cafe), np.angle(resp_fenics)),
    }
    for q_key, (q_label, q_cafe, q_fenics) in quantities.items():
        _save_probe_comparison(case["key"], case["title"], q_key, q_label, q_cafe, q_fenics)

print("\nAll done.")
