import numpy as np
import matplotlib.pyplot as plt
from pycafe.build_matrices.assembly_cquad8 import expand_mode_to_full
from pycafe.post_processing.post_processing import _ask_yes_no

def run_modal_post_processing(
    nodes,
    modes_red,
    freqs,
    idx_free,
    p0_nodes,
    elements=None,
):
    """
    Interactive modal post-processing for acoustic modes.

    This function allows the user to interactively select and
    visualize acoustic mode shapes obtained from a modal analysis.
    The selected mode is expanded from the reduced system to the
    full mesh and then plotted.

    Parameters
    ----------
    nodes : ndarray of shape (N, 2) or (N, 3)
        Coordinates of the mesh nodes.
    modes_red : ndarray of shape (N_red, N_modes)
        Modal shapes computed on the reduced system.
        Each column corresponds to one acoustic mode.
    freqs : ndarray of shape (N_modes,)
        Natural frequencies associated with the acoustic modes [Hz].
    idx_free : ndarray
        Indices of the free degrees of freedom in the reduced system.
    p0_nodes : ndarray
        Indices of nodes where the pressure is constrained to zero.

    Returns
    -
    None

    Notes
    -
    This function is intended for interactive use. The user is
    prompted to select which mode to visualize via the terminal.
    The modal shape is reconstructed on the full mesh before
    visualization.

    See Also
    -
    expand_mode_to_full : Expand a reduced modal vector to the full mesh.
    plot_acoustic_mode : Plot an acoustic mode shape on the mesh.
    """

    num_modes = modes_red.shape[1]

    print("\n=== MODAL POST-PROCESSING ===")
    print("Available acoustic modes:")
    for i, f in enumerate(freqs):
        print(f"  [{i}]  f = {f:.3f} Hz")

    while True:
        ans = input(
            f"\nSelect mode index to plot [0–{num_modes-1}] "
            "(or press ENTER to exit): "
        ).strip()

        if ans == "":
            print("Exiting modal post-processing.")
            break

        try:
            k = int(ans)
        except ValueError:
            print("Invalid input. Please enter an integer.")
            continue

        if k < 0 or k >= num_modes:
            print("Index out of range.")
            continue

        # EXPAND MODE TO FULL DOMAIN
        
        mode_full = expand_mode_to_full(
            modes_red[:, k],
            idx_free,
            p0_nodes,
            nodes.shape[0]
        )

        # PLOT
        plot_acoustic_mode(
            nodes,
            mode_full,
            mode_id=k,
            freq=freqs[k]
        )

    # --- VTK export
    if _ask_yes_no("Export all mode shapes to VTK/VTU files (ParaView)?"):
        out_dir = input("Output directory [vtu_modes]: ").strip() or "vtu_modes"
        from pycafe.post_processing.export_vtk import export_modes_vtu
        export_modes_vtu(nodes, elements or {}, modes_red, freqs, idx_free, p0_nodes, out_dir)

def plot_acoustic_mode(nodes, mode_full, mode_id, freq):
    
    
    mode_plot = np.real(mode_full)
    mode_plot /= np.max(np.abs(mode_plot))

    plt.figure(figsize=(6, 5))
    sc = plt.scatter(
        nodes[:, 0],
        nodes[:, 1],
        c=mode_plot,
        cmap="seismic",
        s=20
    )
    plt.colorbar(sc, label="Normalized Mode Amplitude")
    plt.gca().set_aspect("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Acoustic mode {mode_id+1} – f = {freq:.2f} Hz")
    plt.grid(True)
    plt.show()