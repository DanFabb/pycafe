import numpy as np
import matplotlib.pyplot as plt
import os

def _ask_yes_no(question, default="y"):
    prompt = " [Y/n] " if default.lower() == "y" else " [y/N] "
    ans = input(question + prompt).strip().lower()
    if ans == "":
        return default.lower() == "y"
    return ans.startswith("y")
def create_pressure_video(
    nodes,
    p_full,
    frequencies,
    filename="pressure_field.avi",
    fps=2.5,
):
    import cv2

    tmp_dir = "_frames_tmp"
    os.makedirs(tmp_dir, exist_ok=True)

    frames = []

    # -----------------------------
    # CREATE FRAMES
    # -----------------------------
    for i, f in enumerate(frequencies):
        fig, ax = plt.subplots(figsize=(6, 5))

        sc = ax.scatter(
            nodes[:, 0],
            nodes[:, 1],
            c=np.abs(p_full[:, i]),
            cmap="rainbow",
            s=20
        )
        plt.colorbar(sc, ax=ax, label="pressure module [Pa]")
        ax.set_aspect("equal")
        ax.set_title(f"pressure module at f = {f:.1f} Hz")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True)

        fname = os.path.join(tmp_dir, f"frame_{i:04d}.png")
        plt.savefig(fname, dpi=150)
        plt.close(fig)

        frames.append(fname)

    if len(frames) == 0:
        raise RuntimeError("No frames generated, video aborted.")

    # -----------------------------
    # CREATE VIDEO
    # -----------------------------
    first = cv2.imread(frames[0])
    if first is None:
        raise RuntimeError("Failed to read first frame image.")

    height, width, _ = first.shape

# forza estensione corretta
    if not filename.lower().endswith(".mp4"):
        filename = filename + ".mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    if not video.isOpened():
        raise RuntimeError(
            "VideoWriter failed to open. "
            "Try a different codec or check OpenCV installation."
        )

    for fname in frames:
        img = cv2.imread(fname)
        if img is None:
            raise RuntimeError(f"Failed to read frame: {fname}")
        video.write(img)

    video.release()
    print(f"✅ Video successfully saved: {filename}")


def plot_pressure_field(nodes, p_full, freq_id, frequencies):
    plt.figure(figsize=(6, 5))
    sc = plt.scatter(
        nodes[:, 0],
        nodes[:, 1],
        c=np.abs(p_full[:, freq_id]),
        cmap="rainbow",
        s=20
    )
    plt.colorbar(sc, label="Pressure Amplitude [Pa]")
    plt.gca().set_aspect("equal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"pressure module at f = {frequencies[freq_id]:.1f} Hz")
    plt.grid(True)
    plt.show()

def plot_pressure_frequency_response(frequencies, p_point, label=None):
    """
    Plot pressure amplitude and phase versus frequency.
    """

    amplitude = np.abs(p_point)
    phase = np.angle(p_point)  # [rad]

    fig, ax1 = plt.subplots(figsize=(7, 5))

    # --- Amplitude (left axis)
    ax1.plot(
        frequencies,
        amplitude,
        "b-o",
        label="|p| (Amplitude)"
    )
    ax1.set_xlabel("Frequency [Hz]")
    ax1.set_ylabel("Pressure amplitude [Pa]", color="b")
    ax1.tick_params(axis="y", labelcolor="b")
    ax1.grid(True)

    # -------------------------------------------------
    # PHASE PLOT
    # -------------------------------------------------
    plt.figure(figsize=(7, 5))
    plt.plot(frequencies, phase, "r--s", linewidth=2)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Pressure phase [rad]")
    if label:
        plt.title(f"Pressure phase at {label}")
    else:
        plt.title("Pressure phase vs frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Title
    if label:
        ax1.set_title(f"Pressure frequency response at {label}")
    else:
        ax1.set_title("Pressure frequency response")

    # --- Combined legend
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    ax1.legend(lines_1, labels_1, loc="best")

    plt.tight_layout()
    plt.show()

def run_post_processing(nodes, p_full, frequencies, elements=None):
    """
    Interactive post-processing entry point.
    ALL user interaction happens here.
    """

    print("\n=== POST-PROCESSING MENU ===")

    # --- pressure field
    if _ask_yes_no("Plot pressure field at a specific frequency?"):
        fid = int(input(f"Frequency index [0-{len(frequencies)-1}]: "))
        plot_pressure_field(nodes, p_full, fid, frequencies)

    # --- frequency response
    if _ask_yes_no("Plot pressure frequency response at a point?"):
        x = float(input("x coordinate: "))
        y = float(input("y coordinate: "))
        p_point = weighted_pressure_at_point(
            p_full,
            nodes[:, :2],
            np.array([x, y])
        )
        # ----------------------------------------
        # OPTIONAL: SAVE PRESSURE TO FILE
        # ----------------------------------------
        if _ask_yes_no("Save pressure at this point to a .txt file?"):
            fname = input("Filename [pressure_point.txt]: ").strip()
            if fname == "":
                fname = "pressure_point.txt"

            data = np.column_stack([
                    frequencies,
                    np.real(p_point),
                    np.imag(p_point),
                    np.abs(p_point),
                    np.angle(p_point),
                ])

            header = (
                    "Frequency[Hz]    Re(p)[Pa]    Im(p)[Pa]    |p|[Pa]    Phase[rad]\n"
                    f"Point: x={x}, y={y}"
                )

            np.savetxt(
                    fname,
                    data,
                    header=header,
                    fmt="%.6e"
                )

            print(f"✅ Pressure data saved to '{fname}'")        
        plot_pressure_frequency_response(
            frequencies,
            p_point,
            label=f"Pressure at x={x}, y={y}"
        )

    # --- video
    if _ask_yes_no("Create pressure field video?"):
        fname = input("Video filename [pressure_field.avi]: ").strip()
        if fname == "":
            fname = "pressure_field.avi"
        create_pressure_video(nodes, p_full, frequencies, filename=fname)

    # --- VTK export
    if _ask_yes_no("Export pressure field to VTK/VTU files (ParaView)?"):
        out_dir = input("Output directory [vtu_pressure]: ").strip() or "vtu_pressure"
        from pycafe.post_processing.export_vtk import export_pressure_vtu
        export_pressure_vtu(nodes, elements or {}, p_full, frequencies, out_dir)

    print("=== END POST-PROCESSING ===\n")
import numpy as np

def weighted_pressure_at_point(
    p_full,
    node_xy,
    target_xy,
    num_closest=3,
):
    """
    Compute weighted pressure at a given geometric point.

    Parameters
    ----------
    p_full : ndarray (N_nodes, N_freq)
        Complex pressure field at all nodes for all frequencies.
    node_xy : ndarray (N_nodes, 2)
        Node coordinates [x, y].
    target_xy : array-like (2,)
        Target point coordinates [x, y].
    num_closest : int, optional
        Number of closest nodes used for weighting (default = 3).

    Returns
    -------
    p_weighted : ndarray (N_freq,)
        Weighted complex pressure as a function of frequency.
    """

    node_xy = np.asarray(node_xy, dtype=float)
    target_xy = np.asarray(target_xy, dtype=float)

    if node_xy.shape[1] != 2:
        raise ValueError("node_xy must be of shape (N_nodes, 2)")

    # distanza euclidea dai nodi
    distances = np.linalg.norm(node_xy - target_xy, axis=1)

    # nodi più vicini
    idx = np.argsort(distances)[:num_closest]

    # pesi inversamente proporzionali alla distanza
    w = 1.0 / (distances[idx] + np.finfo(float).eps)
    w /= np.sum(w)

    # pressione pesata (su tutte le frequenze)
    p_weighted = np.sum(p_full[idx, :] * w[:, None], axis=0)

    return p_weighted
