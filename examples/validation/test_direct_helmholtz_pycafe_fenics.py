"""
Direct Helmholtz solver: pyCAFE (CQUAD8) vs FEniCSx (Q2) — 4 cases.

Geometry : 1 m × 0.5 m rectangular cavity, air (c₀=343 m/s, ρ=1.204 kg/m³).
Frequency : 50–800 Hz, 5 Hz steps.
Probe    : see each case.

Cases
-----
1  Piston v_n=1 m/s left / all rigid
   Analytical: p(Lx) = −i·ρ·c₀·v_n / sin(k·Lx)

2  Prescribed pressure p₀=1 Pa left / all rigid
   Analytical: p(Lx) = p₀ / cos(k·Lx)

3  Piston v_n=1 m/s left / pressure-release right (p=0)
   Analytical: p(Lx/2) = i·ρ·c₀·v_n · sin(k(Lx−x)) / cos(k·Lx)

4  Piston v_n=1 m/s left / anechoic right (Z=ρ·c₀)
   Analytical: |p| = ρ·c₀·v_n  (flat, no standing waves)
"""

import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

# ── pyCAFE ────────────────────────────────────────────────────────────
import pycafe
from pycafe.solver.solver_helmholtz_1 import solve_helmholtz_frequency_sweep

# ── FEniCSx ───────────────────────────────────────────────────────────
from mpi4py import MPI
import dolfinx
from dolfinx import mesh as dmesh, fem
from dolfinx.fem import functionspace
from dolfinx.fem.petsc import (
    assemble_matrix as petsc_assemble_matrix,
    assemble_vector as petsc_assemble_vector,
)
import ufl
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# ═════════════════════════════════════════════════════════════════════
# PARAMETERS
# ═════════════════════════════════════════════════════════════════════
MESH_FILE = os.path.join(HERE, "rectangle_CQUAD8.msh")
Lx, Ly = 1.0, 0.5
c0, rho = 343.0, 1.204
v_n = 1.0        # piston velocity toward +x (into the fluid)
# The BCs prescribe the normal velocity along the OUTWARD normal of the
# fluid; at the left wall that normal points toward -x, hence -v_n below.
p0 = 1.0
freqs = np.arange(50.0, 800.0 + 1.0, 5.0)

RESONANCES_RIGID  = c0 / (2*Lx) * np.arange(1, 12)        # sin(kLx)=0
RESONANCES_OPEN   = c0 / (4*Lx) * (2*np.arange(0,12)+1)   # cos(kLx)=0

def mask_away(freqs, singularities, half_bw=6.0):
    m = np.ones(len(freqs), dtype=bool)
    for f_s in singularities:
        m &= np.abs(freqs - f_s) > half_bw
    return m

def closest_node(coords, xy):
    d2 = (coords[:, 0] - xy[0])**2 + (coords[:, 1] - xy[1])**2
    return int(np.argmin(d2))


# ═════════════════════════════════════════════════════════════════════
# LOAD MESH (pyCAFE)
# ═════════════════════════════════════════════════════════════════════
fluid = pycafe.load_fluid("air")
nodes, elements, boundaries = pycafe.load_mesh(MESH_FILE)
N_nodes = nodes.shape[0]
print(f"pyCAFE mesh : {N_nodes} nodes")


# ═════════════════════════════════════════════════════════════════════
# BUILD FEniCSx MESH & ASSEMBLE BASE MATRICES ONCE
# ═════════════════════════════════════════════════════════════════════
Nx, Ny = 64, 32
msh = dmesh.create_rectangle(
    MPI.COMM_WORLD,
    [[0.0, 0.0], [Lx, Ly]],
    [Nx, Ny],
    cell_type=dmesh.CellType.quadrilateral,
)
V = functionspace(msh, ("Lagrange", 2))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
facet_dim = msh.topology.dim - 1

def tag_boundary(msh, facet_dim, pred, tag_id):
    facets = dmesh.locate_entities_boundary(msh, facet_dim, pred)
    markers = np.full(facets.shape, tag_id, dtype=np.int32)
    sort = np.argsort(facets)
    return facets[sort], markers[sort]

left_f,  left_m  = tag_boundary(msh, facet_dim, lambda x: np.isclose(x[0], 0.0),  1)
right_f, right_m = tag_boundary(msh, facet_dim, lambda x: np.isclose(x[0], Lx),   2)

# Combined facet tags (both left=1, right=2)
all_f = np.concatenate([left_f,  right_f])
all_m = np.concatenate([left_m,  right_m])
sort  = np.argsort(all_f)
facet_tags = dmesh.meshtags(msh, facet_dim, all_f[sort], all_m[sort])

ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_tags)

# K, M (real, assembled once)
K_form = fem.form(ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx)
M_form = fem.form(ufl.inner(u, v) * ufl.dx)
K_petsc = petsc_assemble_matrix(K_form); K_petsc.assemble()
M_petsc = petsc_assemble_matrix(M_form); M_petsc.assemble()

# Surface load vectors (∫ N ds on left and right)
b_left_form  = fem.form(ufl.inner(fem.Constant(msh, dolfinx.default_scalar_type(1.0)), v) * ds(1))
b_left_petsc = petsc_assemble_vector(b_left_form); b_left_petsc.assemble()

# Impedance matrix on right (∫ N_i N_j ds on right — assembled once for Z=ρc₀)
C_right_form  = fem.form(ufl.inner(u, v) * ds(2))
C_right_petsc = petsc_assemble_matrix(C_right_form); C_right_petsc.assemble()

def petsc_to_scipy(A):
    ai, aj, av = A.getValuesCSR()
    return sp.csr_matrix((av, aj, ai), shape=A.getSize())

K_sp       = petsc_to_scipy(K_petsc).astype(complex)
M_sp       = petsc_to_scipy(M_petsc).astype(complex)
C_right_sp = petsc_to_scipy(C_right_petsc).astype(complex)
b_left_sp  = b_left_petsc.getArray().astype(complex).copy()

n_dof = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
coords_f = V.tabulate_dof_coordinates()[:, :2]
print(f"FEniCSx mesh: {n_dof} DOF")

# DOF indices on left and right boundaries
def get_boundary_dofs(V, pred):
    facets = dmesh.locate_entities_boundary(V.mesh, facet_dim, pred)
    return fem.locate_dofs_topological(V, facet_dim, facets)

left_dofs  = get_boundary_dofs(V, lambda x: np.isclose(x[0], 0.0))
right_dofs = get_boundary_dofs(V, lambda x: np.isclose(x[0], Lx))
all_dofs   = np.arange(n_dof)


def fenics_sweep(frequencies, *, rhs_func, dirichlet_dofs=None, dirichlet_val=0.0,
                 Z_right=None, probe_xy):
    """
    Generic FEniCSx frequency sweep.
    rhs_func(omega) → complex vector of size n_dof
    dirichlet_dofs  → DOFs where p is imposed (essential BC)
    Z_right         → normalized impedance on right wall (None = rigid)
    """
    probe_dof = closest_node(coords_f, probe_xy)
    p_probe = np.zeros(len(frequencies), dtype=complex)

    for i, f in enumerate(frequencies):
        omega = 2 * np.pi * f
        k2    = (omega / c0) ** 2

        # Assemble system
        if Z_right is not None:
            admittance_f = 1.0 / (Z_right * rho * c0)
            A = K_sp - k2 * M_sp + 1j * omega * rho * admittance_f * C_right_sp
        else:
            A = K_sp - k2 * M_sp

        rhs = rhs_func(omega)

        # Apply Dirichlet BCs by row/col zeroing + penalty diagonal
        if dirichlet_dofs is not None and len(dirichlet_dofs) > 0:
            free_dofs = np.setdiff1d(all_dofs, dirichlet_dofs)
            A_ff = A[np.ix_(free_dofs, free_dofs)]
            A_fd = A[np.ix_(free_dofs, dirichlet_dofs)]
            p_dir = np.full(len(dirichlet_dofs), dirichlet_val, dtype=complex)
            rhs_f = rhs[free_dofs] - A_fd @ p_dir

            p_free = spla.spsolve(A_ff.tocsc(), rhs_f)
            sol = np.zeros(n_dof, dtype=complex)
            sol[dirichlet_dofs] = dirichlet_val
            sol[free_dofs] = p_free
        else:
            sol = spla.spsolve(A.tocsc(), rhs)

        p_probe[i] = sol[probe_dof]

    return p_probe, probe_dof


def pycafe_sweep(bc, probe_xy):
    """Run pyCAFE and return p at probe for all frequencies."""
    system = pycafe.prepare_acoustic_system(
        nodes=nodes, elements=elements, boundaries=boundaries,
        rho=rho, c0=c0, bc=bc, debug=False,
    )
    P_red = solve_helmholtz_frequency_sweep(
        K_red=system["K_red"], M_red=system["M_red"], C_red=system["C_red"],
        frequencies=freqs,
        pressure_nodes_red=system["pressure_nodes_red"],
        pressure_values=system["pressure_values"],
        nodes=nodes,
        boundary_velocity_nodes=system["bc_velocity"],
        idx_free=system["idx_free"],
        rho=rho, v_n=system["value_velocity_normal"],
        boundaries=boundaries, elements=elements,
    )
    P_full = np.zeros((N_nodes, len(freqs)), dtype=complex)
    P_full[system["idx_free"], :] = P_red
    probe_node = closest_node(nodes, probe_xy)
    return P_full[probe_node, :], probe_node


# ═════════════════════════════════════════════════════════════════════
# CASE 1 — Piston left / all rigid
# ═════════════════════════════════════════════════════════════════════
print("\n── Case 1: piston left / all rigid ──")
probe1 = np.array([Lx, Ly / 2.0])

bc1 = ([], [], 0.0, [], 0.0+0j, ["left"], -v_n, None, 0.0)
t0 = time.perf_counter()
p_cafe1, pnode1 = pycafe_sweep(bc1, probe1)
t_cafe1 = time.perf_counter() - t0
print(f"  pyCAFE  probe node {pnode1}  xy={nodes[pnode1,:2]}  t={t_cafe1:.2f}s")

t0 = time.perf_counter()
p_fen1, fdof1 = fenics_sweep(
    freqs,
    rhs_func=lambda omega: (1j * omega * rho * v_n) * b_left_sp,
    probe_xy=probe1,
)
t_fen1 = time.perf_counter() - t0
print(f"  FEniCSx probe DOF {fdof1}  xy={coords_f[fdof1]}  t={t_fen1:.2f}s")

k1  = 2 * np.pi * freqs / c0
p_ana1  = -1j * rho * c0 * v_n / np.sin(k1 * Lx)
mask1   = mask_away(freqs, RESONANCES_RIGID)
err_c1  = np.abs(p_cafe1[mask1] - p_ana1[mask1])  / np.abs(p_ana1[mask1]) * 100
err_f1  = np.abs(p_fen1[mask1]  - p_ana1[mask1])  / np.abs(p_ana1[mask1]) * 100
cross1  = np.abs(p_cafe1[mask1] - p_fen1[mask1])  / np.abs(p_fen1[mask1]) * 100
print(f"  vs analytical  pyCAFE {np.median(err_c1):.4f}%  FEniCSx {np.median(err_f1):.4f}%")
print(f"  cross-error pyCAFE vs FEniCSx  median {np.median(cross1):.4f}%  max {np.max(cross1):.4f}%")


# ═════════════════════════════════════════════════════════════════════
# CASE 2 — Prescribed pressure p₀=1 Pa left / all rigid
# ═════════════════════════════════════════════════════════════════════
print("\n── Case 2: prescribed p₀=1 Pa left / all rigid ──")
probe2 = np.array([Lx, Ly / 2.0])

bc2 = ([], ["left"], p0, [], 0.0+0j, [], 0.0, None, 0.0)
t0 = time.perf_counter()
p_cafe2, pnode2 = pycafe_sweep(bc2, probe2)
t_cafe2 = time.perf_counter() - t0
print(f"  pyCAFE  probe node {pnode2}  xy={nodes[pnode2,:2]}  t={t_cafe2:.2f}s")

t0 = time.perf_counter()
p_fen2, fdof2 = fenics_sweep(
    freqs,
    rhs_func=lambda omega: np.zeros(n_dof, dtype=complex),
    dirichlet_dofs=left_dofs,
    dirichlet_val=p0,
    probe_xy=probe2,
)
t_fen2 = time.perf_counter() - t0
print(f"  FEniCSx probe DOF {fdof2}  xy={coords_f[fdof2]}  t={t_fen2:.2f}s")

k2  = 2 * np.pi * freqs / c0
p_ana2  = p0 / np.cos(k2 * Lx)
f_res2  = c0 / (4 * Lx) * (2 * np.arange(0, 12) + 1)
mask2   = mask_away(freqs, f_res2)
err_c2  = np.abs(p_cafe2[mask2] - p_ana2[mask2]) / np.abs(p_ana2[mask2]) * 100
err_f2  = np.abs(p_fen2[mask2]  - p_ana2[mask2]) / np.abs(p_ana2[mask2]) * 100
cross2  = np.abs(p_cafe2[mask2] - p_fen2[mask2]) / (np.abs(p_fen2[mask2]) + 1e-12) * 100
print(f"  vs analytical  pyCAFE {np.median(err_c2):.4f}%  FEniCSx {np.median(err_f2):.4f}%")
print(f"  cross-error pyCAFE vs FEniCSx  median {np.median(cross2):.4f}%  max {np.max(cross2):.4f}%")


# ═════════════════════════════════════════════════════════════════════
# CASE 3 — Piston left / pressure-release right (p=0)
# ═════════════════════════════════════════════════════════════════════
print("\n── Case 3: piston left / pressure-release right ──")
probe3 = np.array([Lx / 2.0, Ly / 2.0])

bc3 = (["right"], [], 0.0, [], 0.0+0j, ["left"], -v_n, None, 0.0)
t0 = time.perf_counter()
p_cafe3, pnode3 = pycafe_sweep(bc3, probe3)
t_cafe3 = time.perf_counter() - t0
print(f"  pyCAFE  probe node {pnode3}  xy={nodes[pnode3,:2]}  t={t_cafe3:.2f}s")

t0 = time.perf_counter()
p_fen3, fdof3 = fenics_sweep(
    freqs,
    rhs_func=lambda omega: (1j * omega * rho * v_n) * b_left_sp,
    dirichlet_dofs=right_dofs,
    dirichlet_val=0.0,
    probe_xy=probe3,
)
t_fen3 = time.perf_counter() - t0
print(f"  FEniCSx probe DOF {fdof3}  xy={coords_f[fdof3]}  t={t_fen3:.2f}s")

k3     = 2 * np.pi * freqs / c0
x_p3   = nodes[pnode3, 0]
p_ana3 = 1j * rho * c0 * v_n * np.sin(k3 * (Lx - x_p3)) / np.cos(k3 * Lx)
f_res3 = f_res2  # same poles
mask3  = mask_away(freqs, f_res3)
err_c3 = np.abs(p_cafe3[mask3] - p_ana3[mask3]) / (np.abs(p_ana3[mask3]) + 1e-12) * 100
err_f3 = np.abs(p_fen3[mask3]  - p_ana3[mask3]) / (np.abs(p_ana3[mask3]) + 1e-12) * 100
cross3 = np.abs(p_cafe3[mask3] - p_fen3[mask3])  / (np.abs(p_fen3[mask3])  + 1e-12) * 100
print(f"  vs analytical  pyCAFE {np.median(err_c3):.4f}%  FEniCSx {np.median(err_f3):.4f}%")
print(f"  cross-error pyCAFE vs FEniCSx  median {np.median(cross3):.4f}%  max {np.max(cross3):.4f}%")


# ═════════════════════════════════════════════════════════════════════
# CASE 4 — Piston left / anechoic right (Z = ρ·c₀, i.e. Z_norm = 1)
# ═════════════════════════════════════════════════════════════════════
print("\n── Case 4: piston left / anechoic right (Z=1) ──")
probe4 = np.array([Lx / 2.0, Ly / 2.0])

right_nodes_0b = np.array(boundaries["right"], dtype=int) - 1
sort_r = np.argsort(nodes[right_nodes_0b, 1])
right_nodes_1b_sorted = right_nodes_0b[sort_r] + 1

bc4 = ([], [], 0.0, right_nodes_1b_sorted, 1.0+0j, ["left"], -v_n, None, 0.0)
t0 = time.perf_counter()
p_cafe4, pnode4 = pycafe_sweep(bc4, probe4)
t_cafe4 = time.perf_counter() - t0
print(f"  pyCAFE  probe node {pnode4}  xy={nodes[pnode4,:2]}  t={t_cafe4:.2f}s")

# FEniCSx: add iω/Z_spec * C_right to the system; Z_norm=1 → Z_spec = ρ·c₀
t0 = time.perf_counter()
p_fen4, fdof4 = fenics_sweep(
    freqs,
    rhs_func=lambda omega: (1j * omega * rho * v_n) * b_left_sp,
    Z_right=1.0,
    probe_xy=probe4,
)
t_fen4 = time.perf_counter() - t0
print(f"  FEniCSx probe DOF {fdof4}  xy={coords_f[fdof4]}  t={t_fen4:.2f}s")

p_ana4_flat = rho * c0 * v_n   # 413 Pa flat
err_c4  = np.abs(np.abs(p_cafe4) - p_ana4_flat) / p_ana4_flat * 100
err_f4  = np.abs(np.abs(p_fen4)  - p_ana4_flat) / p_ana4_flat * 100
cross4  = np.abs(p_cafe4 - p_fen4) / (np.abs(p_fen4) + 1e-12) * 100
print(f"  vs flat analytical  pyCAFE {np.median(err_c4):.4f}%  FEniCSx {np.median(err_f4):.4f}%")
print(f"  cross-error pyCAFE vs FEniCSx  median {np.median(cross4):.4f}%  max {np.max(cross4):.4f}%")


# ═════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ═════════════════════════════════════════════════════════════════════
print("\n" + "="*72)
print("  CROSS-VALIDATION  pyCAFE vs FEniCSx  (median relative error)")
print("="*72)
print(f"  Case 1  piston/rigid          {np.median(cross1):.4f}%  max {np.max(cross1):.4f}%")
print(f"  Case 2  prescribed p/rigid    {np.median(cross2):.4f}%  max {np.max(cross2):.4f}%")
print(f"  Case 3  piston/p=0 right      {np.median(cross3):.4f}%  max {np.max(cross3):.4f}%")
print(f"  Case 4  piston/anechoic       {np.median(cross4):.4f}%  max {np.max(cross4):.4f}%")
print("="*72)
print(f"  Timing (sweep 151 freqs):")
print(f"    Case 1: pyCAFE {t_cafe1:.2f}s  FEniCSx {t_fen1:.2f}s")
print(f"    Case 2: pyCAFE {t_cafe2:.2f}s  FEniCSx {t_fen2:.2f}s")
print(f"    Case 3: pyCAFE {t_cafe3:.2f}s  FEniCSx {t_fen3:.2f}s")
print(f"    Case 4: pyCAFE {t_cafe4:.2f}s  FEniCSx {t_fen4:.2f}s")
print("="*72)


# ═════════════════════════════════════════════════════════════════════
# PLOTS  (4 rows × 3 cols: |p|, Re(p), phase(p))
# ═════════════════════════════════════════════════════════════════════
titles = [
    "Case 1 — Piston left / rigid  [probe: (Lx, Ly/2)]",
    "Case 2 — Prescribed p=1 Pa left / rigid  [probe: (Lx, Ly/2)]",
    "Case 3 — Piston left / p=0 right  [probe: (Lx/2, Ly/2)]",
    "Case 4 — Piston left / anechoic right  [probe: (Lx/2, Ly/2)]",
]
cases = [
    (p_cafe1, p_fen1, p_ana1),
    (p_cafe2, p_fen2, p_ana2),
    (p_cafe3, p_fen3, p_ana3),
    (p_cafe4, p_fen4, p_ana4_flat * np.ones(len(freqs))),   # flat analytical for case 4
]

fig, axes = plt.subplots(4, 3, figsize=(18, 20))

for row, (p_c, p_f, p_a) in enumerate(cases):
    ax_mod   = axes[row, 0]
    ax_real  = axes[row, 1]
    ax_phase = axes[row, 2]
    title    = titles[row]

    # ── |p| ──
    ax_mod.semilogy(freqs, np.abs(p_c), "b-",  lw=1.6, label="pyCAFE (CQUAD8)")
    ax_mod.semilogy(freqs, np.abs(p_f), "r--", lw=1.4, label="FEniCSx (Q2)")
    if p_a is not None:
        ax_mod.semilogy(freqs, np.abs(p_a), "k:", lw=1.2, label="Analytical 1D")
    ax_mod.set_xlabel("Frequency [Hz]")
    ax_mod.set_ylabel("|p| [Pa]")
    ax_mod.set_title(f"{title}\n|p|", fontsize=8)
    ax_mod.grid(True, which="both", alpha=0.4)
    ax_mod.legend(fontsize=7)

    # ── Re(p) ──
    ax_real.plot(freqs, np.real(p_c), "b-",  lw=1.6, label="pyCAFE (CQUAD8)")
    ax_real.plot(freqs, np.real(p_f), "r--", lw=1.4, label="FEniCSx (Q2)")
    if p_a is not None:
        ax_real.plot(freqs, np.real(p_a), "k:", lw=1.2, label="Analytical 1D")
    ax_real.set_xlabel("Frequency [Hz]")
    ax_real.set_ylabel("Re(p) [Pa]")
    ax_real.set_title(f"{title}\nRe(p)", fontsize=8)
    ax_real.grid(True, alpha=0.4)
    ax_real.legend(fontsize=7)

    # ── phase(p) in degrees ──
    ax_phase.plot(freqs, np.degrees(np.angle(p_c)), "b-",  lw=1.6, label="pyCAFE (CQUAD8)")
    ax_phase.plot(freqs, np.degrees(np.angle(p_f)), "r--", lw=1.4, label="FEniCSx (Q2)")
    if p_a is not None:
        ax_phase.plot(freqs, np.degrees(np.angle(p_a)), "k:", lw=1.2, label="Analytical 1D")
    ax_phase.set_xlabel("Frequency [Hz]")
    ax_phase.set_ylabel("phase(p) [deg]")
    ax_phase.set_title(f"{title}\nphase(p)", fontsize=8)
    ax_phase.set_yticks([-180, -90, 0, 90, 180])
    ax_phase.grid(True, alpha=0.4)
    ax_phase.legend(fontsize=7)

plt.tight_layout()
out_path = os.path.join(HERE, "test_direct_helmholtz_pycafe_fenics.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nFigure saved: {out_path}")
