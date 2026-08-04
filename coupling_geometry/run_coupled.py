"""
End-to-end vibroacoustic workflow: box cavity closed by a flexible plate.

Runs the whole coupled chain on ``box_plate.msh``:

    mesh -> two domains -> coupling matrix Kc -> coupled modes
         -> coupled frequency response -> 3D movie of the cavity

and prints, next to the coupled results, the two uncoupled references
(plate in vacuo, cavity with rigid walls) so the effect of the coupling
is visible: the trapped air acts as a spring on the first plate mode,
which moves up in frequency.

Run after make_box_plate_mesh.py:

    python coupling_geometry/run_coupled.py

Requires an FFmpeg binary for the .mp4 movie; without one the movie is
written as an animated GIF instead.
"""

import pathlib
import sys

import numpy as np
from scipy.sparse.linalg import eigsh

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import pycafe  # noqa: E402
from pycafe.post_processing.post_processing_3d import (  # noqa: E402
    animate_pressure_3d,
    default_movie_extension,
)
from pycafe.solver.solver_modale import (  # noqa: E402
    solve_modal_acoustic_reduced,
)
from pycafe.solver.solver_vibroacoustic import (  # noqa: E402
    build_coupled_blocks,
    expand_pressure,
    solve_vibroacoustic_frequency_sweep,
    solve_vibroacoustic_modal,
    structural_displacement_field,
)

MESH = pathlib.Path(__file__).parent / "box_plate.msh"
OUT = pathlib.Path(__file__).parent

# Geometria (deve combaciare con make_box_plate_mesh)
LX, LY, LZ = 0.8, 0.6, 0.5

# Fluido: aria
RHO0, C0 = 1.204, 343.0
# Piastra: acciaio 2 mm
T, RHO_S, E, NU = 0.002, 7800.0, 210e9, 0.3
# Smorzamento strutturale (loss factor): senza di esso la risposta in
# risonanza e' limitata solo dalla discretizzazione della frequenza.
ETA_S = 0.02

N_MODES = 6
F_MIN, F_MAX, DF = 20.0, 120.0, 1.0


def centre_plate_dof(system, nodes):
    """Reduced index of the transverse DOF at the centre of the plate."""
    idx_s = system["structural"]["idx_free"]
    node_of_dof, component = idx_s // 6, idx_s % 6
    plate = np.unique(node_of_dof)
    centre = np.array([LX / 2.0, LY / 2.0, LZ])
    target = plate[np.argmin(np.linalg.norm(nodes[plate] - centre, axis=1))]
    return int(np.where((node_of_dof == target) & (component == 2))[0][0])


def main():
    nodes, elements, boundaries, groups = pycafe.load_mesh_with_groups(
        str(MESH)
    )

    # ------------------------------------------------------------------
    # 1. Due domini + accoppiamento
    # ------------------------------------------------------------------
    system = pycafe.prepare_vibroacoustic_system(
        nodes=nodes, groups=groups,
        rho0=RHO0, c0=C0,
        t=T, rho_s=RHO_S, E=E, nu=NU,
    )
    coupling = system["coupling"]
    interface = system["interface"]

    print("\n=== Domini e interfaccia ===")
    for role in ("fluid", "structure"):
        d = system["domains"][role]
        print(f"  {role:<10} '{d['group']}': {d['elem_type']} "
              f"x {d['conn0'].shape[0]}")
    print(f"  interfaccia: {interface['conn0'].shape[0]} facce, "
          f"normale media {np.round(interface['normals'].mean(axis=0), 6)} "
          "(uscente dal fluido)")
    print(f"  Kc: {coupling['Kc'].shape}, nnz {coupling['Kc'].nnz}")
    print(f"  area vettoriale int_S n dS = "
          f"{np.round(coupling['area_vector'], 6)} m^2 "
          f"(attesa {LX * LY:.4f} lungo z)")

    # ------------------------------------------------------------------
    # 2. Riferimenti disaccoppiati
    # ------------------------------------------------------------------
    f_cav, _ = solve_modal_acoustic_reduced(
        system["acoustic"]["K"], system["acoustic"]["M"], num_modes=3
    )
    K_s = system["structural"]["K_red"].tocsc()
    M_s = system["structural"]["M_red"].tocsc()
    vals, _ = eigsh(K_s, k=3, M=M_s, sigma=0.0, which="LM")
    f_plate = np.sqrt(np.abs(np.sort(vals))) / (2 * np.pi)

    print("\n=== Riferimenti disaccoppiati ===")
    print(f"  piastra in vacuo : {np.round(f_plate, 2)} Hz")
    print(f"  cavita' rigida   : {np.round(f_cav, 2)} Hz")

    # ------------------------------------------------------------------
    # 3. Modi accoppiati
    # ------------------------------------------------------------------
    blocks = build_coupled_blocks(system)
    freqs, _ = solve_vibroacoustic_modal(system, num_modes=N_MODES,
                                         blocks=blocks)

    print(f"\n=== Modi accoppiati ({blocks['n_s']} DOF struttura + "
          f"{blocks['n_a']} pressioni) ===")
    print(f"  {np.round(freqs, 2)} Hz")
    print(f"  il primo modo (~0 Hz) e' la pressione uniforme della "
          "cavita' chiusa")
    shift = (freqs[1] - f_plate[0]) / f_plate[0] * 100
    print(f"  1o modo flessibile: {freqs[1]:.2f} Hz contro "
          f"{f_plate[0]:.2f} Hz in vacuo  ({shift:+.1f}%: l'aria "
          "intrappolata irrigidisce la piastra)")

    # ------------------------------------------------------------------
    # 4. Risposta in frequenza accoppiata
    # ------------------------------------------------------------------
    dof = centre_plate_dof(system, nodes)
    F_s = np.zeros(blocks["n_s"])
    F_s[dof] = 1.0                      # 1 N normale al centro piastra

    frequencies = np.arange(F_MIN, F_MAX + DF, DF)
    damped = build_coupled_blocks(system, eta_s=ETA_S)
    result = solve_vibroacoustic_frequency_sweep(
        system, frequencies, F_s=F_s, blocks=damped, verbose=False,
    )

    w_centre = result["w"][dof]
    p_max = np.abs(result["p"]).max(axis=0)
    peak = int(np.argmax(np.abs(w_centre)))

    print(f"\n=== Risposta forzata (1 N al centro, eta_s = {ETA_S}) ===")
    print(f"  picco a {frequencies[peak]:.1f} Hz: "
          f"|w| = {abs(w_centre[peak]):.3e} m, "
          f"|p|max = {p_max[peak]:.2f} Pa")

    w_field = structural_displacement_field(
        result["w"][:, peak], result["idx_s"], nodes.shape[0]
    )
    print(f"  freccia massima piastra: "
          f"{np.abs(w_field[:, 2]).max():.3e} m")

    # ------------------------------------------------------------------
    # 5. Filmato 3D del campo di pressione accoppiato
    # ------------------------------------------------------------------
    p_full = expand_pressure(result["p"], result["idx_a"], nodes.shape[0])
    ext = default_movie_extension()

    sweep_movie = OUT / f"coupled_sweep{ext}"
    animate_pressure_3d(
        nodes, elements, p_full, frequencies,
        mode="sweep", filename=str(sweep_movie), fps=8,
    )

    time_movie = OUT / f"coupled_wave{ext}"
    animate_pressure_3d(
        nodes, elements, p_full, frequencies,
        mode="time", freq_id=peak, n_time_steps=36,
        clip={"axis": "y", "keep": "<"},
        filename=str(time_movie), fps=15,
    )

    print("\nFilmati scritti in coupling_geometry/.")


if __name__ == "__main__":
    main()
