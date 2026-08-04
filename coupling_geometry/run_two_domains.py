"""
End-to-end vibroacoustic preparation: from box_plate.msh to the two
separate domains, ready for the coupling matrices.

Computes and prints the uncoupled modes of both domains:
- acoustic cavity (rigid walls) vs analytical f_lmn;
- clamped plate vs analytical thin-plate estimate
  (lambda_11 ~ 35.99 for a clamped rectangular plate a x b via the
  Leissa coefficient for a/b = 0.8/0.6 -- printed for reference only).

Run after make_box_plate_mesh.py:

    python coupling_geometry/run_two_domains.py
"""

import pathlib
import sys

import numpy as np
from scipy.sparse.linalg import eigsh

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import pycafe  # noqa: E402
from pycafe.solver.solver_modale import solve_modal_acoustic_reduced  # noqa: E402

MESH = pathlib.Path(__file__).parent / "box_plate.msh"

# Geometria (deve combaciare con make_box_plate_mesh)
LX, LY, LZ = 0.8, 0.6, 0.5

# Fluido: aria
RHO0, C0 = 1.204, 343.0
# Piastra: acciaio 2 mm
T, RHO_S, E, NU = 0.002, 7800.0, 210e9, 0.3


def main():
    nodes, elements, boundaries, groups = pycafe.load_mesh_with_groups(str(MESH))

    system = pycafe.prepare_vibroacoustic_system(
        nodes=nodes, groups=groups,
        rho0=RHO0, c0=C0,
        t=T, rho_s=RHO_S, E=E, nu=NU,
    )

    dom = system["domains"]
    print("\n=== Due domini separati ===")
    for role in ("fluid", "structure"):
        d = dom[role]
        print(f"  {role:<10} group '{d['group']}': {d['elem_type']} "
              f"x {d['conn0'].shape[0]}")
    print(f"  interfaccia: {system['interface']['conn0'].shape[0]} facce, "
          f"{len(system['interface']['nodes0'])} nodi condivisi")
    print(f"  struttura: {system['structural']['K_red'].shape[0]} DOF liberi "
          f"({len(system['structural']['clamped_nodes0'])} nodi incastrati)")
    Kc = system["coupling"]["Kc"]
    print(f"  coupling: Kc {Kc.shape}, nnz {Kc.nnz} "
          "(risolto in run_coupled.py)")

    # ---- modi acustici (cavità rigida) ----
    K_a, M_a = system["acoustic"]["K"], system["acoustic"]["M"]
    freqs_a, _ = solve_modal_acoustic_reduced(K_a, M_a, num_modes=3)
    f_exact = sorted(
        (C0 / 2.0) * np.sqrt((l / LX) ** 2 + (m / LY) ** 2 + (n / LZ) ** 2)
        for l in range(4) for m in range(4) for n in range(4)
        if (l, m, n) != (0, 0, 0)
    )[:3]
    print("\n=== Modi acustici (cavità rigida) ===")
    for f_fem, f_ex in zip(freqs_a, f_exact):
        print(f"  FEM {f_fem:8.2f} Hz | analitico {f_ex:8.2f} Hz "
              f"| err {abs(f_fem-f_ex)/f_ex*100:.2f}%")

    # ---- modi strutturali (piastra incastrata) ----
    K_s, M_s = system["structural"]["K_red"], system["structural"]["M_red"]
    vals, _ = eigsh(K_s.tocsc(), k=3, M=M_s.tocsc(), sigma=0.0, which="LM")
    freqs_s = np.sqrt(np.abs(np.sort(vals))) / (2 * np.pi)
    D = E * T**3 / (12 * (1 - NU**2))
    # Leissa CCCC, a/b = 1.33: lambda^2 ~ 52.5
    f_ref = 52.5 / (2 * np.pi * LX**2) * np.sqrt(D / (RHO_S * T))
    print("\n=== Modi strutturali (piastra incastrata) ===")
    for f in freqs_s:
        print(f"  FEM {f:8.2f} Hz")
    print(f"  (rif. Leissa f11 ~ {f_ref:.2f} Hz)")

    print("\nSistema pronto. Per il problema accoppiato: "
          "python coupling_geometry/run_coupled.py")


if __name__ == "__main__":
    main()
