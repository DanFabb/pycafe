r"""
Visualization and animation of a 3D acoustic pressure field.

Everything here runs inside the Python environment with the packages
pyCAFE already requires (numpy and matplotlib): no OpenCV, no PyVista,
no external viewer such as ParaView. Movies are written through
matplotlib's own animation writers -- FFmpeg when the binary is
available, Pillow (animated GIF) otherwise -- and the animation object
is returned, so it can also be shown inline in a notebook with
``HTML(anim.to_jshtml())``.

Two kinds of movie are produced from a frequency sweep:

``mode="sweep"``
    One frame per frequency, showing :math:`|p|`: how the response map
    changes across the sweep.

``mode="time"``
    One frequency, frames over a period, showing
    :math:`\mathrm{Re}\{p\, e^{j\omega t}\}`: the wave actually moving.

The colour scale is held fixed across the frames of a movie, so that
frames are comparable. Only the outer skin of the mesh is rendered; a
cutting plane (``clip=``) exposes the interior.

Examples
--------
Static field, then the two movies::

    plot_pressure_3d(nodes, elements, p_full[:, 12], part="spl")

    animate_pressure_3d(nodes, elements, p_full, frequencies,
                        mode="sweep", filename="sweep.mp4", fps=8)

    animate_pressure_3d(nodes, elements, p_full, frequencies,
                        mode="time", freq_id=12,
                        clip={"axis": "y", "keep": "<"},
                        filename="wave.mp4", fps=15)

Notes
-----
``.mp4`` and ``.avi`` need an FFmpeg binary: a system-wide one is used
when it is on the ``PATH``, otherwise the one shipped by the
``imageio-ffmpeg`` wheel, a pyCAFE requirement. ``.gif`` always works,
Pillow being part of matplotlib. Use :func:`default_movie_extension` to
pick the best available container at run time.
"""

import os
import shutil

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, cm, colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Reference pressure of airborne acoustics, for the SPL scale.
P_REF = 2e-5


# ------------------------------------------------------------
#  SURFACE EXTRACTION
# ------------------------------------------------------------
# Corner faces of the volume elements, in the Gmsh node ordering.
# Only corner nodes are used: a second-order element is rendered
# through its corners, which is enough for a flat-shaded picture.
_VOLUME_FACES = {
    "Hexahedron": (8, [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
                       (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]),
    "Tetrahedron": (4, [(0, 1, 3), (0, 2, 1), (0, 3, 2), (1, 2, 3)]),
}

# Surface element types, rendered as they are.
_SURFACE_CORNERS = {
    "Quadrilateral": 4,
    "Quadrangle": 4,
    "Triangle": 3,
}


def _match(name, table):
    """Match a Gmsh element name against a table of name prefixes."""
    for key, value in table.items():
        if key in name:
            return value
    return None


def extract_surface_faces(elements):
    """
    Extract the faces to be rendered from a mesh connectivity dict.

    On a 3D mesh the outer skin of the volume elements is returned: a
    face is on the skin when it belongs to exactly one element. On a 2D
    mesh the surface elements themselves are returned.

    Parameters
    ----------
    elements : dict
        ``{gmsh_element_name: (n_elem, n_nodes) int array}``, **1-based**,
        as returned by the mesh loader.

    Returns
    -------
    faces : list of ndarray
        Each entry holds the 0-based node indices of one face (3 or 4
        corners).

    Raises
    ------
    ValueError
        If no renderable element type is found.
    """
    volume_faces = {}
    has_volume = False

    for name, conn in elements.items():
        spec = _match(name, _VOLUME_FACES)
        if spec is None:
            continue
        has_volume = True
        n_corner, face_defs = spec
        conn0 = np.asarray(conn, dtype=int)[:, :n_corner] - 1
        for face in face_defs:
            nodes_on_face = conn0[:, list(face)]
            for row in nodes_on_face:
                key = tuple(sorted(row.tolist()))
                # A face shared by two elements is interior: drop it.
                if key in volume_faces:
                    del volume_faces[key]
                else:
                    volume_faces[key] = row
    if has_volume:
        return list(volume_faces.values())

    # No volume elements: the mesh is a surface (or a 2D fluid).
    faces = []
    for name, conn in elements.items():
        n_corner = _match(name, _SURFACE_CORNERS)
        if n_corner is None:
            continue
        conn0 = np.asarray(conn, dtype=int)[:, :n_corner] - 1
        faces.extend(list(conn0))

    if not faces:
        raise ValueError(
            "No renderable element type found. Supported: "
            f"{sorted(_VOLUME_FACES) + sorted(_SURFACE_CORNERS)}; "
            f"mesh contains {sorted(elements)}."
        )
    return faces


def _face_values(faces, nodal_values):
    """Average the nodal field over each face (flat shading)."""
    return np.array([nodal_values[f].mean() for f in faces])


# ------------------------------------------------------------
#  FIGURE SET-UP
# ------------------------------------------------------------
def _setup_axes(nodes, faces, elev, azim, figsize):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection="3d")

    coords = np.asarray(nodes, dtype=float)
    if coords.shape[1] == 2:
        coords = np.column_stack([coords, np.zeros(len(coords))])

    polys = [coords[f] for f in faces]
    coll = Poly3DCollection(polys, edgecolor="k", linewidths=0.15)
    ax.add_collection3d(coll)

    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    # A flat mesh (a 2D fluid lying in z = 0) has a degenerate extent
    # along one axis: pad it so the limits stay non-singular.
    pad = 0.05 * np.max(hi - lo)
    flat = (hi - lo) < 1e-12
    lo = np.where(flat, lo - pad, lo)
    hi = np.where(flat, hi + pad, hi)
    span = np.maximum(hi - lo, 1e-12)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(span)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.view_init(elev=elev, azim=azim)

    return fig, ax, coll


def ensure_ffmpeg():
    """
    Make an FFmpeg binary visible to matplotlib, and report it.

    Looked up in this order, the first hit winning: what matplotlib is
    already configured with, a system-wide ``ffmpeg`` on the PATH, and
    the binary shipped by the ``imageio-ffmpeg`` wheel (a pyCAFE
    dependency, so the movie export works even without a system
    install). The path found is written into
    ``matplotlib.rcParams["animation.ffmpeg_path"]``.

    Returns
    -------
    path : str or None
        Path of the usable binary, or ``None`` when no FFmpeg is
        available; ``.gif`` export works regardless.
    """
    if animation.FFMpegWriter.isAvailable():
        return plt.rcParams["animation.ffmpeg_path"]

    candidates = [shutil.which("ffmpeg")]
    try:
        import imageio_ffmpeg

        candidates.append(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        # No wheel, or no binary inside it: fall through to None.
        pass

    for path in candidates:
        if path and os.path.exists(path):
            plt.rcParams["animation.ffmpeg_path"] = path
            if animation.FFMpegWriter.isAvailable():
                return path

    return None


def _writer_for(filename, fps):
    """
    Pick a matplotlib writer from the file extension.

    Video containers need FFmpeg (see :func:`ensure_ffmpeg`); ``.gif``
    is always available through Pillow, which ships with matplotlib.
    """
    ext = str(filename).lower().rsplit(".", 1)[-1]

    if ext == "gif":
        return animation.PillowWriter(fps=fps), filename

    if ext in ("mp4", "avi", "mov", "mkv"):
        if ensure_ffmpeg() is None:
            raise RuntimeError(
                f"No FFmpeg binary was found, so '{filename}' cannot be "
                "written. Install one inside the environment "
                "(`pip install imageio-ffmpeg`, a pyCAFE requirement) or "
                "system-wide (`brew install ffmpeg`, `conda install "
                "ffmpeg`), or ask for a '.gif' file, which needs no "
                "external codec."
            )
        # MJPEG is the codec an .avi container expects; the default
        # h264 belongs to .mp4/.mov.
        codec = "mjpeg" if ext == "avi" else "h264"
        return animation.FFMpegWriter(fps=fps, codec=codec), filename

    raise ValueError(
        f"Unsupported movie extension '.{ext}'. Use .mp4/.avi/.mov/.mkv "
        "(FFmpeg) or .gif (no external codec)."
    )


# ------------------------------------------------------------
#  STATIC PLOT
# ------------------------------------------------------------
def plot_pressure_3d(
    nodes,
    elements,
    p,
    *,
    part="abs",
    title=None,
    cmap=None,
    clim=None,
    elev=25,
    azim=-60,
    figsize=(7, 6),
    clip=None,
    faces=None,
    show=True,
):
    """
    Plot one 3D pressure field on the outer surface of the mesh.

    Parameters
    ----------
    nodes : ndarray (N, 3)
        Node coordinates.
    elements : dict
        Mesh connectivity (1-based), as returned by the mesh loader.
        Ignored when ``faces`` is given.
    p : ndarray (N,) complex
        Nodal pressure at one frequency.
    part : {"abs", "spl", "real", "imag"}, optional
        Quantity mapped to colour. ``"abs"`` and ``"spl"`` (level in dB
        re 20 uPa) use a sequential map, ``"real"``/``"imag"`` a
        symmetric diverging one.
    title : str, optional
    cmap : str or Colormap, optional
        Defaults to ``"rainbow"`` for ``abs``/``spl`` and ``"seismic"``
        otherwise.
    clim : (vmin, vmax), optional
        Colour limits; computed from the field when omitted (a 60 dB
        window below the peak for ``part="spl"``).
    elev, azim : float, optional
        View angles [deg].
    figsize : tuple, optional
    clip : dict, optional
        Cutting plane exposing the interior, passed to
        :func:`clip_elements`, e.g. ``{"axis": "y", "value": 0.25}``.
    faces : list of ndarray, optional
        Pre-extracted faces (see :func:`extract_surface_faces`), to avoid
        re-extracting them on repeated calls. Takes precedence over
        ``clip``.
    show : bool, optional
        Call ``plt.show()`` before returning.

    Returns
    -------
    fig, ax : matplotlib figure and 3D axes
    """
    faces = _resolve_faces(nodes, elements, faces, clip)

    values = _extract_part(np.asarray(p), part)
    cmap = _default_cmap(cmap, part)

    if clim is None:
        clim = _default_clim(values, part)
    norm = colors.Normalize(vmin=clim[0], vmax=clim[1])

    fig, ax, coll = _setup_axes(nodes, faces, elev, azim, figsize)
    coll.set_facecolor(cmap(norm(_face_values(faces, values))))

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    fig.colorbar(mappable, ax=ax, shrink=0.7, label=_label(part))
    ax.set_title(title or f"Pressure field ({part})")

    if show:
        plt.show()
    return fig, ax


_SEQUENTIAL_PARTS = ("abs", "spl")


def _extract_part(p, part):
    if part == "abs":
        return np.abs(p)
    if part == "spl":
        # Sound pressure level of the complex amplitude, i.e. of the
        # peak value; the floor keeps log10 finite at a nodal line.
        return 20.0 * np.log10(np.maximum(np.abs(p), 1e-30) / P_REF)
    if part == "real":
        return np.real(p)
    if part == "imag":
        return np.imag(p)
    raise ValueError(
        f"Unknown part '{part}'. Use 'abs', 'spl', 'real' or 'imag'."
    )


def _default_clim(values, part):
    if part == "spl":
        # A 60 dB window under the peak: without it the -600 dB of a
        # nodal line would flatten the whole scale.
        vmax = float(np.max(values)) if values.size else 0.0
        return (vmax - 60.0, vmax)
    if part == "abs":
        vmax = float(np.max(values)) if values.size else 1.0
        return (0.0, vmax if vmax > 0 else 1.0)
    vmax = float(np.max(np.abs(values))) if values.size else 1.0
    vmax = vmax if vmax > 0 else 1.0
    return (-vmax, vmax)


def _label(part):
    return {
        "abs": "|p| [Pa]",
        "spl": f"SPL [dB re {P_REF:g} Pa]",
        "real": "Re(p) [Pa]",
        "imag": "Im(p) [Pa]",
    }[part]


def _default_cmap(cmap, part):
    if cmap is not None:
        return plt.get_cmap(cmap)
    return plt.get_cmap("rainbow" if part in _SEQUENTIAL_PARTS else "seismic")


# ------------------------------------------------------------
#  CLIPPING
# ------------------------------------------------------------
def clip_elements(nodes, elements, axis="x", value=None, keep="<"):
    """
    Keep only the elements lying on one side of a plane.

    Rendering the skin of the *clipped* mesh exposes a cross section, so
    the interior of a 3D fluid becomes visible instead of only its outer
    surface.

    Parameters
    ----------
    nodes : ndarray (N, 3)
        Node coordinates.
    elements : dict
        Mesh connectivity (1-based), as returned by the mesh loader.
    axis : {"x", "y", "z"} or int, optional
        Normal of the cutting plane.
    value : float, optional
        Position of the plane along ``axis``; defaults to the mid-span
        of the mesh.
    keep : {"<", ">"}, optional
        Side that survives, judged on the element centroid.

    Returns
    -------
    clipped : dict
        Connectivity of the surviving elements, 1-based, same layout as
        the input.

    Raises
    ------
    ValueError
        On an unknown axis or side, or when the plane leaves no element.
    """
    axes = {"x": 0, "y": 1, "z": 2}
    k = axes.get(axis, axis) if not isinstance(axis, int) else axis
    if k not in (0, 1, 2):
        raise ValueError(f"Unknown axis '{axis}'. Use 'x', 'y', 'z' or 0/1/2.")
    if keep not in ("<", ">"):
        raise ValueError(f"Unknown side '{keep}'. Use '<' or '>'.")

    coords = np.asarray(nodes, dtype=float)
    if value is None:
        value = 0.5 * (coords[:, k].min() + coords[:, k].max())

    clipped = {}
    for name, conn in elements.items():
        conn = np.asarray(conn, dtype=int)
        centroid = coords[conn - 1, k].mean(axis=1)
        mask = centroid < value if keep == "<" else centroid > value
        if np.any(mask):
            clipped[name] = conn[mask]

    if not clipped:
        raise ValueError(
            f"The plane {axis} {keep} {value} leaves no element; check the "
            "position against the extent of the mesh "
            f"[{coords[:, k].min():.4g}, {coords[:, k].max():.4g}]."
        )
    return clipped


def _resolve_faces(nodes, elements, faces, clip):
    """Faces to render, applying the clipping plane when requested."""
    if faces is not None:
        return faces
    if clip is not None:
        elements = clip_elements(nodes, elements, **clip)
    return extract_surface_faces(elements)


# ------------------------------------------------------------
#  POINT RESPONSE
# ------------------------------------------------------------
def pressure_at_point_3d(p_full, nodes, target, num_closest=4):
    """
    Complex pressure at an arbitrary 3D point, over the whole sweep.

    Inverse-distance interpolation on the ``num_closest`` nearest nodes,
    the 3D counterpart of
    :func:`pycafe.post_processing.post_processing.weighted_pressure_at_point`.

    Parameters
    ----------
    p_full : ndarray (N, N_freq) complex
        Nodal pressure over the sweep.
    nodes : ndarray (N, 3)
        Node coordinates.
    target : array-like (3,)
        Point of interest.
    num_closest : int, optional
        Nodes used for the interpolation.

    Returns
    -------
    p_point : ndarray (N_freq,) complex
    """
    from .post_processing import weighted_pressure_at_point

    return weighted_pressure_at_point(
        p_full, nodes, target, num_closest=num_closest
    )


# ------------------------------------------------------------
#  ANIMATION
# ------------------------------------------------------------
def animate_pressure_3d(
    nodes,
    elements,
    p_full,
    frequencies,
    *,
    mode="sweep",
    part=None,
    freq_id=None,
    filename=None,
    fps=10,
    n_time_steps=36,
    n_periods=1,
    cmap=None,
    clim=None,
    elev=25,
    azim=-60,
    figsize=(7, 6),
    dpi=120,
    clip=None,
    faces=None,
    show=False,
):
    """
    Animate a 3D acoustic pressure field.

    Two kinds of movie are supported.

    ``mode="sweep"``
        One frame per frequency of the sweep, colouring the surface with
        ``|p|``. Shows how the response map evolves along the sweep.

    ``mode="time"``
        A single frequency, animated over one or more periods:
        ``Re{p e^{j omega t}}``, with the ``exp(+j omega t)`` convention
        used by the solver. This is the movie that shows the wave
        travelling / the mode breathing.

    The colour scale is held fixed across all frames, so frames are
    comparable.

    Parameters
    ----------
    nodes : ndarray (N, 3)
        Node coordinates.
    elements : dict
        Mesh connectivity (1-based), as returned by the mesh loader.
        Ignored when ``faces`` is given.
    p_full : ndarray (N, N_freq) complex
        Nodal pressure over the sweep, expanded to the full mesh.
    frequencies : array-like (N_freq,)
        Sweep frequencies [Hz].
    mode : {"sweep", "time"}, optional
    part : {"abs", "spl", "real", "imag"}, optional
        Quantity mapped to colour, overriding the default of the mode
        (``"abs"`` for a sweep, ``"real"`` for a time movie). Use
        ``"spl"`` for a sweep in dB re 20 uPa.
    freq_id : int, optional
        Frequency index animated in ``mode="time"``; defaults to the
        index of maximum global response.
    filename : str, optional
        Output movie. The extension picks the writer:
        ``.mp4``/``.mov``/``.mkv`` (FFmpeg, h264), ``.avi`` (FFmpeg,
        MJPEG) or ``.gif`` (Pillow, no external codec). When omitted
        nothing is written and only the animation object is returned.
    fps : int, optional
        Frames per second of the movie.
    n_time_steps : int, optional
        Frames per period in ``mode="time"``.
    n_periods : int, optional
        Number of periods animated in ``mode="time"``.
    cmap : str or Colormap, optional
    clim : (vmin, vmax), optional
        Colour limits; computed from the whole sweep when omitted.
    elev, azim : float, optional
        View angles [deg].
    figsize : tuple, optional
    dpi : int, optional
        Resolution of the written movie.
    clip : dict, optional
        Cutting plane exposing the interior, passed to
        :func:`clip_elements`, e.g. ``{"axis": "y", "value": 0.25}``.
    faces : list of ndarray, optional
        Pre-extracted faces (see :func:`extract_surface_faces`). Takes
        precedence over ``clip``.
    show : bool, optional
        Call ``plt.show()`` after building the animation.

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
        Kept alive by the caller; in a notebook it can be displayed with
        ``HTML(anim.to_jshtml())``.

    Raises
    ------
    ValueError
        On an unknown ``mode`` or an unsupported movie extension.
    RuntimeError
        When a video container is requested but FFmpeg is missing.

    See Also
    --------
    plot_pressure_3d : Single-frame version.
    extract_surface_faces : Outer skin of the mesh.
    clip_elements : Cutting plane exposing the interior.
    """
    p_full = np.asarray(p_full)
    frequencies = np.asarray(frequencies, dtype=float)

    if p_full.ndim != 2:
        raise ValueError("p_full must have shape (n_nodes, n_freq).")
    if p_full.shape[1] != frequencies.size:
        raise ValueError(
            f"p_full has {p_full.shape[1]} columns but "
            f"{frequencies.size} frequencies were given."
        )

    # Resolve the writer before any frame is built: a missing codec or a
    # wrong extension must fail immediately, not after the rendering.
    writer = None
    if filename is not None:
        writer, filename = _writer_for(filename, fps)

    faces = _resolve_faces(nodes, elements, faces, clip)

    # --- frame data, colour scale and titles, per mode ---
    if mode == "sweep":
        part = part or "abs"
        fields = [_extract_part(p_full[:, i], part)
                  for i in range(frequencies.size)]
        titles = [f"{_label(part)}  -  f = {f:.2f} Hz" for f in frequencies]
    elif mode == "time":
        part = part or "real"
        if freq_id is None:
            freq_id = int(np.argmax(np.max(np.abs(p_full), axis=0)))
        p_f = p_full[:, freq_id]
        n_frames = int(n_time_steps) * int(n_periods)
        phases = np.linspace(0.0, 2.0 * np.pi * n_periods, n_frames,
                             endpoint=False)
        # The instantaneous field: the harmonic amplitude rotated in the
        # complex plane, then read on the axis asked for by `part`.
        fields = [_extract_part(p_f * np.exp(1j * ph), part) for ph in phases]
        titles = [
            f"{_label(part)}  -  f = {frequencies[freq_id]:.2f} Hz  -  "
            f"phase = {np.degrees(ph) % 360.0:6.1f} deg"
            for ph in phases
        ]
    else:
        raise ValueError(f"Unknown mode '{mode}'. Use 'sweep' or 'time'.")

    cmap = _default_cmap(cmap, part)
    if clim is None:
        clim = _default_clim(np.concatenate(fields), part)
    norm = colors.Normalize(vmin=clim[0], vmax=clim[1])

    # Flat-shade once per frame, up front: the surface never moves, only
    # its colours do, so the animation itself is a colour swap.
    face_colors = [cmap(norm(_face_values(faces, v))) for v in fields]

    fig, ax, coll = _setup_axes(nodes, faces, elev, azim, figsize)
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    fig.colorbar(mappable, ax=ax, shrink=0.7, label=_label(part))

    def _draw(i):
        coll.set_facecolor(face_colors[i])
        ax.set_title(titles[i])
        return (coll,)

    _draw(0)

    anim = animation.FuncAnimation(
        fig,
        _draw,
        frames=len(face_colors),
        interval=1000.0 / max(fps, 1),
        blit=False,
        repeat=True,
    )

    if writer is not None:
        anim.save(filename, writer=writer, dpi=dpi)
        print(f"Movie saved: {filename}  "
              f"({len(face_colors)} frames, {fps} fps)")

    if show:
        plt.show()

    return anim


# ------------------------------------------------------------
#  INTERACTIVE MENU
# ------------------------------------------------------------
def default_movie_extension():
    """
    Extension used by the menus when the user does not give one.

    ``.mp4`` when an FFmpeg binary is reachable, ``.gif`` otherwise, so
    the interactive session never fails on a missing codec.
    """
    return ".mp4" if ensure_ffmpeg() is not None else ".gif"


def _ask_index(label, n):
    """Ask for an index in [0, n-1], repeating until it is valid."""
    while True:
        try:
            k = int(input(f"{label} [0-{n - 1}]: "))
        except ValueError:
            print("  Not an integer, try again.")
            continue
        if 0 <= k < n:
            return k
        print("  Index out of range, try again.")


def _ask_clip(nodes):
    """Ask for an optional cutting plane; ``None`` when declined."""
    from .post_processing import _ask_yes_no

    if not _ask_yes_no("Cut the model open to see the interior?", default="n"):
        return None

    axis = (input("  Cut normal [x/y/z, default x]: ").strip().lower() or "x")
    k = {"x": 0, "y": 1, "z": 2}.get(axis, 0)
    lo, hi = float(nodes[:, k].min()), float(nodes[:, k].max())
    mid = 0.5 * (lo + hi)
    raw = input(f"  Plane position along {axis} "
                f"[{lo:.4g} .. {hi:.4g}, default {mid:.4g}]: ").strip()
    value = float(raw) if raw else mid
    keep = input("  Side to keep [< or >, default <]: ").strip() or "<"
    return {"axis": axis, "value": value, "keep": keep}


def _ask_part(default="abs"):
    """Ask which quantity is mapped to colour."""
    raw = input(
        f"  Quantity [abs/spl/real/imag, default {default}]: "
    ).strip().lower()
    return raw or default


def _save_point_response(frequencies, p_point, point):
    """Write the point frequency response to a text file, on request."""
    from .post_processing import _ask_yes_no

    if not _ask_yes_no("Save this response to a .txt file?", default="n"):
        return

    fname = input("  Filename [pressure_point_3d.txt]: ").strip()
    fname = fname or "pressure_point_3d.txt"

    data = np.column_stack([
        frequencies,
        np.real(p_point),
        np.imag(p_point),
        np.abs(p_point),
        np.angle(p_point),
        20.0 * np.log10(np.maximum(np.abs(p_point), 1e-30) / P_REF),
    ])
    header = (
        "Frequency[Hz]  Re(p)[Pa]  Im(p)[Pa]  |p|[Pa]  Phase[rad]  SPL[dB]\n"
        f"Point: x={point[0]}, y={point[1]}, z={point[2]}"
    )
    np.savetxt(fname, data, header=header, fmt="%.6e")
    print(f"  Response saved: {fname}")


def run_post_processing_3d(nodes, elements, p_full, frequencies):
    """
    Interactive post-processing of a 3D frequency-response result.

    Terminal counterpart of
    :func:`pycafe.post_processing.post_processing.run_post_processing`
    for a 3D fluid. Offers, in order:

    1. the pressure field at one frequency (``abs``, ``spl``, ``real``
       or ``imag``), optionally cut open by a plane;
    2. the frequency response at an arbitrary (x, y, z) point, amplitude
       and phase, with an optional text export;
    3. the frequency-sweep movie, one frame per frequency;
    4. the time movie at one frequency, ``Re{p e^{j omega t}}``.

    Parameters
    ----------
    nodes : ndarray (N, 3)
        Node coordinates.
    elements : dict
        Mesh connectivity (1-based), as returned by the mesh loader.
    p_full : ndarray (N, N_freq) complex
        Nodal pressure over the sweep, expanded to the full mesh.
    frequencies : array-like (N_freq,)
        Sweep frequencies [Hz].
    """
    from .post_processing import (
        _ask_yes_no,
        plot_pressure_frequency_response,
    )

    nodes = np.asarray(nodes, dtype=float)
    frequencies = np.asarray(frequencies, dtype=float)
    n_freq = frequencies.size

    # The outer skin is extracted once and reused by every entry that
    # does not ask for its own cut.
    faces = extract_surface_faces(elements)
    ext = default_movie_extension()

    print("\n=== 3D POST-PROCESSING ===")
    if ext == ".gif":
        print("No FFmpeg found: movies default to .gif "
              "(`pip install imageio-ffmpeg` to get .mp4).")

    # --- 1. field at one frequency ---
    if _ask_yes_no("Plot the 3D pressure field at one frequency?"):
        fid = _ask_index("Frequency index", n_freq)
        part = _ask_part("abs")
        clip = _ask_clip(nodes)
        plot_pressure_3d(
            nodes, elements, p_full[:, fid],
            part=part,
            title=f"{_label(part)}  -  f = {frequencies[fid]:.2f} Hz",
            clip=clip,
            faces=None if clip else faces,
        )

    # --- 2. frequency response at a point ---
    if _ask_yes_no("Plot the frequency response at a point?"):
        x = float(input("  x coordinate: "))
        y = float(input("  y coordinate: "))
        z = float(input("  z coordinate: "))
        point = np.array([x, y, z])
        p_point = pressure_at_point_3d(p_full, nodes, point)
        _save_point_response(frequencies, p_point, point)
        plot_pressure_frequency_response(
            frequencies, p_point,
            label=f"x={x}, y={y}, z={z}",
        )

    # --- 3. sweep movie ---
    if _ask_yes_no("Create the frequency-sweep movie (|p| vs frequency)?"):
        part = _ask_part("abs")
        clip = _ask_clip(nodes)
        fname = input(f"  Filename [pressure_sweep_3d{ext}]: ").strip()
        animate_pressure_3d(
            nodes, elements, p_full, frequencies,
            mode="sweep",
            part=part,
            filename=fname or f"pressure_sweep_3d{ext}",
            clip=clip,
            faces=None if clip else faces,
        )

    # --- 4. time movie ---
    if _ask_yes_no("Create the time movie at one frequency (Re(p) vs time)?"):
        fid = _ask_index("Frequency index", n_freq)
        clip = _ask_clip(nodes)
        fname = input(f"  Filename [pressure_time_3d{ext}]: ").strip()
        raw = input("  Frames per period [default 36]: ").strip()
        animate_pressure_3d(
            nodes, elements, p_full, frequencies,
            mode="time",
            freq_id=fid,
            n_time_steps=int(raw) if raw else 36,
            filename=fname or f"pressure_time_3d{ext}",
            clip=clip,
            faces=None if clip else faces,
        )

    print("=== END 3D POST-PROCESSING ===\n")
