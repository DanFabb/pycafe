"""
Five direct Helmholtz solver test cases — pyCAFE CQUAD8 vs 1D analytical.

Geometry : 1 m × 0.5 m rectangular cavity, air.
Mesh     : examples/validation/rectangle_CQUAD8.msh  (6337 DOF, CQUAD8)

Cases
-----
1. Piston (v_n=1 m/s) on left, rigid everywhere else
   Analytical:  p(Lx) = -i·ρ·c₀·v_n / sin(k·Lx)

2. Prescribed pressure p₀=1 Pa on left, rigid everywhere else
   Analytical:  p(Lx) = p₀ / cos(k·Lx)

3. Piston on left, pressure-release (p=0) on right
   Analytical:  p(Lx/2) = i·ρ·c₀·v_n · sin(k·Lx/2) / cos(k·Lx)

4. Piston on left, anechoic impedance Z=1 on right (no resonances)
   Analytical:  |p| = ρ·c₀·v_n  (flat, no standing wave)

5. Symmetry: piston on left, rigid everywhere — probe at two y-heights
   Expected:    identical pressure (y-independence of uniform piston)
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Allow running from the project root or from examples/validation/
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

import pycafe
from pycafe.solver.solver_helmholtz_1 import solve_helmholtz_frequency_sweep

# ─────────────────────────────────────────────────────────
MESH = os.path.join(HERE, "rectangle_CQUAD8.msh")
Lx, Ly = 1.0, 0.5
c0 = 343.0
rho = 1.204
v_n = 1.0        # piston velocity toward +x (into the fluid)
# The BCs prescribe the normal velocity along the OUTWARD normal of the
# fluid; at the left wall that normal points toward -x, hence -v_n below.
p0 = 1.0   # prescribed pressure amplitude

freqs = np.arange(50.0, 800.0 + 1.0, 5.0)
# Exclude points within 5 Hz of resonances (singularities)
f_res_rigid = c0 / (2 * Lx) * np.arange(1, 10)      # n·c0/(2Lx): rigid-rigid
f_res_open  = c0 / (4 * Lx) * (2 * np.arange(1, 10) - 1)  # (2n-1)·c0/(4Lx): rigid-open

fluid = pycafe.load_fluid("air")
nodes, elements, boundaries = pycafe.load_mesh(MESH)

print(f"Mesh: {nodes.shape[0]} nodes  |  freqs: {freqs[0]}–{freqs[-1]} Hz  ({len(freqs)} steps)")


def closest_node(nodes, xy):
    d2 = (nodes[:, 0] - xy[0])**2 + (nodes[:, 1] - xy[1])**2
    return int(np.argmin(d2))


def mask_resonances(freqs, f_sing, half_bw=6.0):
    """Return boolean mask: True where freq is far from all singularities."""
    mask = np.ones(len(freqs), dtype=bool)
    for f_s in f_sing:
        mask &= np.abs(freqs - f_s) > half_bw
    return mask


def run_sweep(bc, probe_xy, elements=None):
    """Prepare system and run frequency sweep; return (p_probe, system)."""
    system = pycafe.prepare_acoustic_system(
        nodes=nodes, elements=elements, boundaries=boundaries,
        rho=rho, c0=c0, bc=bc, debug=False,
    )
    P_red = solve_helmholtz_frequency_sweep(
        K_red=system["K_red"],
        M_red=system["M_red"],
        C_red=system["C_red"],
        frequencies=freqs,
        pressure_nodes_red=system["pressure_nodes_red"],
        pressure_values=system["pressure_values"],
        nodes=nodes,
        boundary_velocity_nodes=system["bc_velocity"],
        idx_free=system["idx_free"],
        rho=rho,
        v_n=system["value_velocity_normal"],
        boundaries=boundaries,
        elements=elements,
    )
    N_nodes = nodes.shape[0]
    P_full = np.zeros((N_nodes, len(freqs)), dtype=complex)
    P_full[system["idx_free"], :] = P_red

    probe_node = closest_node(nodes, probe_xy)
    return P_full[probe_node, :], system, probe_node


results = {}  # store (freqs, p_num, p_ana, mask, title)

# ═════════════════════════════════════════════════════════
# Case 1: Piston left, all rigid
# Analytical: p(Lx, ·) = -i·ρ·c₀·v_n / sin(k·Lx)
# ═════════════════════════════════════════════════════════
print("\n── Case 1: Piston left, rigid everywhere ──")
bc1 = ([], [], 0.0,
       [], 0.0 + 0j,
       ["left"], -v_n,
       None, 0.0)
probe1 = np.array([Lx, Ly / 2.0])
p_num1, sys1, node1 = run_sweep(bc1, probe1, elements=elements)
print(f"  Probe node {node1}  xy={nodes[node1, :2]}")

k1 = 2 * np.pi * freqs / c0
p_ana1 = -1j * rho * c0 * v_n / np.sin(k1 * Lx)
mask1 = mask_resonances(freqs, f_res_rigid)
err1 = np.abs(p_num1[mask1] - p_ana1[mask1]) / np.abs(p_ana1[mask1]) * 100
print(f"  Median rel. error: {np.median(err1):.4f} %   Max: {np.max(err1):.4f} %")
results["Case 1: Piston left / Rigid"] = (freqs, p_num1, p_ana1, mask1,
                                           "Case 1 — Piston left, rigid everywhere\nProbe: (Lx, Ly/2)")

# ═════════════════════════════════════════════════════════
# Case 2: Prescribed pressure p0 on left, rigid everywhere else
# Analytical: p(Lx) = p₀ / cos(k·Lx)
# ═════════════════════════════════════════════════════════
print("\n── Case 2: Prescribed pressure left, rigid everywhere ──")
bc2 = ([], ["left"], p0,
       [], 0.0 + 0j,
       [], 0.0,
       None, 0.0)
probe2 = np.array([Lx, Ly / 2.0])
p_num2, sys2, node2 = run_sweep(bc2, probe2, elements=elements)
print(f"  Probe node {node2}  xy={nodes[node2, :2]}")

# Singularities at cos(kLx)=0 → k·Lx = π/2 + nπ → f = c0/(4Lx)·(2n+1)
f_res_cos = c0 / (4 * Lx) * (2 * np.arange(0, 10) + 1)
k2 = 2 * np.pi * freqs / c0
p_ana2 = p0 / np.cos(k2 * Lx)
mask2 = mask_resonances(freqs, f_res_cos)
err2 = np.abs(p_num2[mask2] - p_ana2[mask2]) / np.abs(p_ana2[mask2]) * 100
print(f"  Median rel. error: {np.median(err2):.4f} %   Max: {np.max(err2):.4f} %")
results["Case 2: Prescribed p left / Rigid"] = (freqs, p_num2, p_ana2, mask2,
    "Case 2 — Prescribed pressure p₀=1 Pa left, rigid everywhere\nProbe: (Lx, Ly/2)")

# ═════════════════════════════════════════════════════════
# Case 3: Piston left, pressure-release right (p=0)
# Analytical: p(Lx/2) = i·ρ·c₀·v_n · sin(k·Lx/2) / cos(k·Lx)
# Wait: p(x) = i·ρ·c₀·v_n · sin(k(Lx-x)) / cos(kLx)  → at x=Lx/2
# ═════════════════════════════════════════════════════════
print("\n── Case 3: Piston left, pressure-release right ──")
bc3 = (["right"], [], 0.0,
       [], 0.0 + 0j,
       ["left"], -v_n,
       None, 0.0)
probe3 = np.array([Lx / 2.0, Ly / 2.0])
p_num3, sys3, node3 = run_sweep(bc3, probe3, elements=elements)
print(f"  Probe node {node3}  xy={nodes[node3, :2]}")

k3 = 2 * np.pi * freqs / c0
x_probe3 = nodes[node3, 0]
p_ana3 = 1j * rho * c0 * v_n * np.sin(k3 * (Lx - x_probe3)) / np.cos(k3 * Lx)
# Singularities: cos(kLx)=0 → same as f_res_cos
mask3 = mask_resonances(freqs, f_res_cos)
err3 = np.abs(p_num3[mask3] - p_ana3[mask3]) / (np.abs(p_ana3[mask3]) + 1e-12) * 100
print(f"  Median rel. error: {np.median(err3):.4f} %   Max: {np.max(err3):.4f} %")
results["Case 3: Piston left / p=0 right"] = (freqs, p_num3, p_ana3, mask3,
    "Case 3 — Piston left, pressure-release right\nProbe: (Lx/2, Ly/2)")

# ═════════════════════════════════════════════════════════
# Case 4: Piston left, anechoic (Z=1, perfectly absorbing) right
# Analytical: |p(x)| = ρ·c₀·v_n  (flat, no standing waves)
# ═════════════════════════════════════════════════════════
print("\n── Case 4: Piston left, anechoic right (Z=1) ──")
# boundary_nodes_impedance: pass 1-based sorted right-wall nodes
right_nodes_0based = np.array(boundaries["right"], dtype=int) - 1
y_coords = nodes[right_nodes_0based, 1]
sort_idx = np.argsort(y_coords)
right_nodes_1based_sorted = right_nodes_0based[sort_idx] + 1

bc4 = ([], [], 0.0,
       right_nodes_1based_sorted, 1.0 + 0j,
       ["left"], -v_n,
       None, 0.0)
probe4 = np.array([Lx / 2.0, Ly / 2.0])
p_num4, sys4, node4 = run_sweep(bc4, probe4, elements=elements)
print(f"  Probe node {node4}  xy={nodes[node4, :2]}")

p_ana4_abs = rho * c0 * v_n  # 413 Pa, flat
# No analytical singularities for anechoic — compare |p| to flat level
err4 = np.abs(np.abs(p_num4) - p_ana4_abs) / p_ana4_abs * 100
print(f"  Median |p| error vs flat: {np.median(err4):.4f} %   Max: {np.max(err4):.4f} %")
results["Case 4: Piston left / Anechoic right"] = (freqs, p_num4, None, None,
    f"Case 4 — Piston left, anechoic right (Z=ρc₀)\nProbe: (Lx/2, Ly/2)   Analytical |p| = {p_ana4_abs:.1f} Pa")

# ═════════════════════════════════════════════════════════
# Case 5: Symmetry — two probes at different y, same x
# Expected: identical pressure (1D field, y-independent)
# ═════════════════════════════════════════════════════════
print("\n── Case 5: Symmetry check ──")
bc5 = bc1  # same as Case 1
probe5a = np.array([Lx * 0.6, 0.10])
probe5b = np.array([Lx * 0.6, 0.40])

# re-run for two probes
system5 = pycafe.prepare_acoustic_system(
    nodes=nodes, elements=elements, boundaries=boundaries,
    rho=rho, c0=c0, bc=bc5, debug=False,
)
P_red5 = solve_helmholtz_frequency_sweep(
    K_red=system5["K_red"],
    M_red=system5["M_red"],
    C_red=system5["C_red"],
    frequencies=freqs,
    pressure_nodes_red=system5["pressure_nodes_red"],
    pressure_values=system5["pressure_values"],
    nodes=nodes,
    boundary_velocity_nodes=system5["bc_velocity"],
    idx_free=system5["idx_free"],
    rho=rho,
    v_n=system5["value_velocity_normal"],
    boundaries=boundaries,
    elements=elements,
)
P_full5 = np.zeros((nodes.shape[0], len(freqs)), dtype=complex)
P_full5[system5["idx_free"], :] = P_red5

node5a = closest_node(nodes, probe5a)
node5b = closest_node(nodes, probe5b)
p5a = P_full5[node5a, :]
p5b = P_full5[node5b, :]
sym_err = np.abs(p5a - p5b) / (np.abs(p5a) + 1e-12) * 100
print(f"  Probe A: node {node5a}  xy={nodes[node5a, :2]}")
print(f"  Probe B: node {node5b}  xy={nodes[node5b, :2]}")
print(f"  Symmetry error — median: {np.median(sym_err):.4f} %   max: {np.max(sym_err):.4f} %")

# ═════════════════════════════════════════════════════════
# PLOTS
# ═════════════════════════════════════════════════════════
fig, axes = plt.subplots(3, 2, figsize=(14, 14))
axes = axes.ravel()

plot_order = [
    "Case 1: Piston left / Rigid",
    "Case 2: Prescribed p left / Rigid",
    "Case 3: Piston left / p=0 right",
    "Case 4: Piston left / Anechoic right",
]

for ax_idx, key in enumerate(plot_order):
    ax = axes[ax_idx]
    freqs_p, p_num, p_ana, mask, title = results[key]

    ax.semilogy(freqs_p, np.abs(p_num), "b-", lw=1.6, label="pyCAFE CQUAD8")
    if p_ana is not None:
        ax.semilogy(freqs_p, np.abs(p_ana), "r--", lw=1.4, label="Analytical 1D")

    if key == "Case 4: Piston left / Anechoic right":
        ax.axhline(rho * c0 * v_n, color="r", ls="--", lw=1.4, label=f"Analytical |p|={rho*c0*v_n:.0f} Pa")

    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("|p| [Pa]")
    ax.set_title(title, fontsize=9)
    ax.grid(True, which="both", alpha=0.4)
    ax.legend(fontsize=8)

# Case 5 — relative error between two probes
ax5 = axes[4]
ax5.semilogy(freqs, np.abs(p5a), "b-", lw=1.4, label=f"Probe A  y={nodes[node5a,1]:.3f} m")
ax5.semilogy(freqs, np.abs(p5b), "r--", lw=1.4, label=f"Probe B  y={nodes[node5b,1]:.3f} m")
ax5.set_xlabel("Frequency [Hz]")
ax5.set_ylabel("|p| [Pa]")
ax5.set_title("Case 5 — Symmetry: two probes at same x, different y\n(should overlap exactly)")
ax5.grid(True, which="both", alpha=0.4)
ax5.legend(fontsize=8)

# Case 5 — sym error
ax6 = axes[5]
mask_sym = mask_resonances(freqs, f_res_rigid)
ax6.plot(freqs[mask_sym], sym_err[mask_sym], "k-", lw=1.2)
ax6.set_xlabel("Frequency [Hz]")
ax6.set_ylabel("Symmetry error [%]")
ax6.set_title("Case 5 — |pA − pB| / |pA|  (away from resonances)")
ax6.grid(True, alpha=0.4)

plt.tight_layout()
out_path = os.path.join(HERE, "test_direct_helmholtz_cases.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nFigure saved: {out_path}")

# ═════════════════════════════════════════════════════════
# SUMMARY TABLE
# ═════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  SUMMARY  —  median relative error vs analytical (away from poles)")
print("="*65)
print(f"  Case 1  piston / rigid           {np.median(err1):.4f} %  max {np.max(err1):.4f} %")
print(f"  Case 2  prescribed p / rigid     {np.median(err2):.4f} %  max {np.max(err2):.4f} %")
print(f"  Case 3  piston / p=0 right       {np.median(err3):.4f} %  max {np.max(err3):.4f} %")
print(f"  Case 4  piston / anechoic        {np.median(err4):.4f} %  max {np.max(err4):.4f} %  (|p| vs flat)")
print(f"  Case 5  symmetry error           {np.median(sym_err[mask_sym]):.4f} %  max {np.max(sym_err[mask_sym]):.4f} %")
print("="*65)
