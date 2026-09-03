r"""
The pictures a run ends with, drawn the same way every time.

Two figures carry most of what an analysis has to say: a frequency
response, and a strip of mode shapes. Written out cell by cell they come
out slightly different in every notebook — a different colour scale, a
legend on one and not the other, a grid loud enough to compete with the
data. Here they are one call each.

The rules they follow:

* one quantity per axes, never two scales on one frame: a displacement
  and a pressure are two panels, so that neither curve can be read
  against the wrong ticks;
* a signed field is drawn on a **diverging** map about white, since zero
  is a real place on a mode shape; an amplitude is drawn on a **single
  hue**, light to dark, since zero is only the bottom of the scale;
* the grid and the axes stay recessive, and only the few peaks worth
  naming carry a number.

.. code-block:: python

    plot_frf(frequencies, {"|p| at the mic [Pa]": p_mic}, resonances=f_modal)
    plot_mode_grid(nodes, elements, shapes, f_modal)
"""

import numpy as np

__all__ = ["plot_frf", "plot_modes", "plot_mode_grid", "plot_field",
           "plot_matrix", "plot_shares",
           "SIGNED_CMAP", "AMPLITUDE_CMAP"]

# A mode shape is signed and zero means something: diverging, white in
# the middle. An amplitude only grows: one hue, light to dark.
SIGNED_CMAP = "RdBu_r"
AMPLITUDE_CMAP = "Blues"

# Ink, not series colour: labels and annotations never wear the colour
# of the curve they describe.
INK = "0.20"
MUTED = "0.45"
GRID = "0.85"
# Categorical slots, in fixed order and never cycled: two curves are
# always the first two, so a colour means the same thing from one figure
# to the next. Four is the cap that clears the colour-blind separation
# checks; past that the panels are split instead of adding a fifth hue.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
MAX_SERIES = len(SERIES)


def plot_frf(frequencies, signals, *, points=None, resonances=None,
             annotate=3, band=None, title=None, figsize=None, show=False):
    """
    A frequency response, one panel per quantity.

    Parameters
    ----------
    frequencies : array-like (N,)
        The sweep [Hz].
    signals : dict
        ``{label: values}``, one panel each, in order. The label is the
        y axis, so write it with its unit: ``"|p| at the mics [Pa]"``.
        A panel may hold several curves of the **same** quantity, as
        ``{label: {name: values}}`` — several microphones, say — and
        they are drawn together with a legend. Two different quantities
        never share a panel: that is what the outer dict is for.
        Complex input is read through its modulus.
    points : dict, optional
        ``{label: {name: (f, values)}}`` — a few lines computed some
        other way, drawn as markers on the panel of that ``label``
        instead of as a curve. A sweep too expensive to run line by
        line is read this way: the curve is the cheap model, the
        markers the lines the expensive one was actually run at.
    resonances : array-like, optional
        Frequencies to mark under the curves [Hz], typically the modes.
        Only those inside the sweep are drawn.
    annotate : int, optional
        How many peaks carry their frequency in writing, on a panel that
        holds a single curve. The tallest ones; a number on every peak
        is noise, and on several curves at once it is a thicket.
    band : tuple, optional
        ``(f1, f2)`` shaded, for a region worth pointing at.
    title : str, optional
    figsize : tuple, optional
    show : bool, optional

    Returns
    -------
    list of matplotlib axes

    Notes
    -----
    More curves than :data:`MAX_SERIES` in one panel would need a hue
    that no longer separates under colour blindness, so the panel is
    split into one per curve instead.
    """
    import matplotlib.pyplot as plt

    from .frf import find_response_peaks

    frequencies = np.asarray(frequencies, dtype=float)
    points = points or {}

    # A panel is (label, {name: values}); a bare array is the one-curve
    # case, and a panel with too many curves becomes several panels.
    panels = []
    for label, entry in signals.items():
        curves = entry if isinstance(entry, dict) else {label: entry}
        if len(curves) + len(points.get(label, {})) > MAX_SERIES:
            for name, values in curves.items():
                panels.append((f"{label}\n{name}", {name: values}))
        else:
            panels.append((label, curves))

    height = 2.9 * len(panels) + 1.0
    fig, axes = plt.subplots(len(panels), 1,
                             figsize=figsize or (10.5, height),
                             sharex=True, squeeze=False)
    axes = list(axes[:, 0])

    inside = None
    if resonances is not None and len(resonances):
        resonances = np.asarray(resonances, dtype=float)
        inside = resonances[(resonances >= frequencies[0])
                            & (resonances <= frequencies[-1])]

    for (label, curves), ax in zip(panels, axes):
        if band is not None:
            ax.axvspan(band[0], band[1], color=GRID, alpha=0.45, zorder=0)
        if inside is not None:
            for f in inside:
                ax.axvline(f, color=GRID, lw=1.0, zorder=1)

        for k, (name, values) in enumerate(curves.items()):
            amplitude = np.abs(np.asarray(values))
            ax.semilogy(frequencies, amplitude, lw=1.6, color=SERIES[k],
                        label=name, zorder=3)

            # Only on a panel that holds one curve: the tallest peaks are
            # named, since a number on every one of them is a second grid.
            if len(curves) == 1 and not points.get(label) and annotate:
                peaks = find_response_peaks(frequencies, amplitude)
                if len(peaks):
                    heights = np.interp(peaks, frequencies, amplitude)
                    for f in peaks[np.argsort(-heights)][:annotate]:
                        y = float(np.interp(f, frequencies, amplitude))
                        ax.annotate(f"{f:.1f} Hz", xy=(f, y),
                                    xytext=(0, 7), textcoords="offset points",
                                    ha="center", fontsize=8, color=INK,
                                    zorder=4)

        # The lines a second, dearer, model was run at: markers on the
        # same panel, taking the colour slots after the curves.
        marks = points.get(label, {})
        for k, (name, (f_points, values)) in enumerate(marks.items()):
            ax.plot(np.asarray(f_points, dtype=float),
                    np.abs(np.asarray(values)), ls="none", marker="o",
                    ms=5.5, mfc="none", mew=1.6,
                    color=SERIES[(len(curves) + k) % MAX_SERIES],
                    label=name, zorder=5)

        # Identity is never colour alone: two curves or more carry a
        # legend, in the same order as the colours were assigned.
        if len(curves) + len(marks) > 1:
            legend = ax.legend(frameon=False, fontsize=8, loc="upper right",
                               ncols=min(len(curves), 4))
            for text in legend.get_texts():
                text.set_color(INK)

        ax.set_ylabel(label, color=INK)
        ax.grid(True, which="major", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelcolor=INK)

    axes[-1].set_xlabel("frequency [Hz]", color=INK)
    axes[-1].set_xlim(frequencies[0], frequencies[-1])
    if inside is not None and len(inside):
        axes[0].annotate(f"{len(inside)} modes inside the sweep",
                         xy=(0.995, 1.02), xycoords="axes fraction",
                         ha="right", va="bottom", fontsize=8, color=MUTED)
    if title:
        fig.suptitle(title, y=0.99, color=INK)
    fig.tight_layout()
    if show:
        plt.show()
    return axes


def plot_mode_grid(nodes, elements, shapes, frequencies, *, columns=3,
                   max_modes=6, cmap=SIGNED_CMAP, title=None, figsize=None,
                   elev=22, azim=-52, show=False):
    """
    Mode shapes side by side, on one scale, with the axes out of the way.

    Each shape is normalised on its own peak and its sign fixed positive,
    since an eigenvector is defined up to a factor and the same mode
    would otherwise come out red or blue at random. The geometry is the
    same in every panel and says so once, so the axes are dropped and
    the colour bar is shared.

    Parameters
    ----------
    nodes : ndarray (N, 3)
    elements : dict
        Mesh connectivity, for the outer faces.
    shapes : ndarray (N, k)
        Nodal pressure of each mode, already expanded onto the mesh.
    frequencies : array-like (k,)
        Their frequencies [Hz].
    columns : int, optional
    max_modes : int, optional
        How many to draw.
    cmap : str, optional
        Diverging by default: a mode shape is signed.
    title : str, optional
    figsize : tuple, optional
    elev, azim : float, optional
    show : bool, optional

    Returns
    -------
    matplotlib figure
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm

    from .post_processing_3d import extract_surface_faces, plot_pressure_3d

    shapes = np.asarray(shapes)
    frequencies = np.asarray(frequencies, dtype=float)
    count = min(max_modes, shapes.shape[1], frequencies.size)
    rows = int(np.ceil(count / columns))

    # The faces are the same in every panel: extracted once, not once
    # per mode, which is most of the drawing time on a large mesh.
    faces = extract_surface_faces(elements)

    fig = plt.figure(figsize=figsize or (4.3 * columns, 3.7 * rows + 0.6))
    for k in range(count):
        shape = shapes[:, k]
        peak = np.abs(shape).max() or 1.0
        shape = shape / peak
        if np.real(shape[np.abs(shape).argmax()]) < 0:
            shape = -shape

        ax = fig.add_subplot(rows, columns, k + 1, projection="3d")
        plot_pressure_3d(nodes, elements, shape, part="real", cmap=cmap,
                         clim=(-1, 1), faces=faces, ax=ax, colorbar=False,
                         show=False, elev=elev, azim=azim,
                         title=f"mode {k + 1}\n{frequencies[k]:.1f} Hz")
        ax.set_axis_off()
        ax.title.set_color(INK)

    # A 3D axes leaves a wide margin of its own, so the panels are pushed
    # together by hand and the colour bar gets an axes of its own rather
    # than stealing width from all of them.
    fig.subplots_adjust(left=0.01, right=0.87, bottom=0.01,
                        top=0.88 if title else 0.93, wspace=0.02, hspace=0.14)
    bar = fig.colorbar(
        cm.ScalarMappable(norm=plt.Normalize(-1, 1), cmap=cmap),
        cax=fig.add_axes([0.90, 0.28, 0.015, 0.44]),
    )
    bar.set_label("pressure, normalised on each peak", color=INK)
    bar.outline.set_edgecolor(GRID)
    bar.ax.tick_params(colors=MUTED)
    if title:
        fig.suptitle(title, y=0.99, color=INK)
    if show:
        plt.show()
    return fig


def plot_field(nodes, elements, values, *, part="abs", cmap=None, title=None,
               label=None, clim=None, markers=None, elev=22, azim=-52,
               figsize=(7.5, 5.6), ax=None, show=False):
    """
    One field on the outer faces of the mesh, tidied for reading.

    The same picture as
    :func:`~pycafe.post_processing.post_processing_3d.plot_pressure_3d`,
    with the choices that keep it legible made here: an amplitude on a
    single hue and a signed field on a diverging one, four ticks per
    axis instead of a dozen that collide at the origin, and ink that is
    not the colour of the data.

    Parameters
    ----------
    nodes : ndarray (N, 3)
    elements : dict
    values : ndarray (N,)
        Nodal field, complex or real.
    part : {"abs", "spl", "real", "imag"}, optional
        ``"abs"`` and ``"spl"`` are magnitudes and take the sequential
        map; ``"real"`` and ``"imag"`` are signed and take the diverging
        one.
    cmap : str, optional
        Overrides that choice.
    title, label : str, optional
    clim : tuple, optional
    markers : dict or sequence, optional
        Points to mark on the field and name: ``{"mic 1": (x, y, z)}``,
        or a sequence of ``(name, (x, y, z))``. The colours are the
        categorical slots, in the order given, so a marker and the curve
        it belongs to on the response wear the same one.
    elev, azim : float, optional
    figsize : tuple, optional
    ax : matplotlib 3D axes, optional
    show : bool, optional

    Returns
    -------
    matplotlib 3D axes
    """
    import matplotlib.pyplot as plt

    from .post_processing_3d import plot_pressure_3d

    if cmap is None:
        cmap = AMPLITUDE_CMAP if part in ("abs", "spl") else SIGNED_CMAP

    fig, ax = plot_pressure_3d(nodes, elements, values, part=part, cmap=cmap,
                               clim=clim, title=title, elev=elev, azim=azim,
                               figsize=figsize, ax=ax, show=False)
    if markers:
        items = (list(markers.items()) if isinstance(markers, dict)
                 else list(markers))
        for k, (name, point) in enumerate(items):
            colour = SERIES[k % len(SERIES)]
            point = np.asarray(point, dtype=float)
            ax.scatter(*point, s=48, color=colour, edgecolor="white",
                       linewidth=0.9, depthshade=False, zorder=5)
            # A point inside the volume is behind the faces that are drawn,
            # so the label is what has to be readable: it carries the colour
            # of the curve that point produces on the response.
            ax.text(point[0], point[1], point[2], f" {name}", fontsize=8,
                    color=INK, zorder=6, va="center",
                    bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                              edgecolor=colour, linewidth=1.0, alpha=0.92))

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(plt.MaxNLocator(4))
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.title.set_color(INK)
    for name in ("xaxis", "yaxis", "zaxis"):
        getattr(ax, name).label.set_color(INK)
    if label:
        for bar in [c for c in fig.axes if c is not ax]:
            bar.set_ylabel(label, color=INK)
    if show:
        plt.show()
    return ax


def plot_modes(model, frequencies, pressure_shapes, *, structure_shapes=None,
               blocks=None, max_modes=6, title=None, show=False):
    """
    The mode shapes of a run, drawn the way that run is usually read.

    An acoustic model has one field per mode and they go in a strip. A
    coupled model has two, and they go side by side, one row per mode:
    the structure on the left, the cavity on the right. Each half is
    normalised on its own peak, because the eigenvector mixes metres and
    pascals and the ratio between the halves means nothing.

    Parameters
    ----------
    model : dict
        Output of :func:`~pycafe.core.model_spec.build_model`.
    frequencies : array-like (k,)
        The modal frequencies [Hz].
    pressure_shapes : ndarray (N, k)
        Nodal pressure of each mode, expanded onto the mesh.
    structure_shapes : ndarray (n_s + n_a, k), optional
        The coupled eigenvectors, for a vibroacoustic run. Their
        structural half is what is drawn on the left.
    blocks : dict, optional
        The blocks those eigenvectors were computed on, for ``n_s`` and
        ``idx_s``. Required with ``structure_shapes``.
    max_modes : int, optional
    title : str, optional
    show : bool, optional

    Returns
    -------
    matplotlib figure

    Notes
    -----
    A structure that is not one flat surface has no plane to be drawn
    on, so only the cavity is shown and a note says why.

    The left half is drawn by ``pycafe_vibro``: passing
    ``structure_shapes`` without that package installed raises, while
    an acoustic strip is drawn by pyCAFE alone.
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm

    from .post_processing_3d import extract_surface_faces, plot_pressure_3d

    nodes, elements = model["nodes"], model["elements"]
    frequencies = np.asarray(frequencies, dtype=float)

    structure_group = model["report"].roles.get("structure")
    panel = None
    if structure_shapes is not None and blocks is not None and structure_group:
        try:
            from pycafe_vibro.panel import describe_panel, plot_panel_mode
        except ImportError:
            raise ImportError(
                "drawing the structural half of a coupled mode needs "
                "pycafe_vibro: pip install pycafe-vibro. Without it, call "
                "plot_modes without structure_shapes to draw the cavity "
                "alone."
            ) from None
        try:
            panel = describe_panel(nodes,
                                   model["groups"][structure_group]["nodes"])
        except ValueError as why:
            print(f"the structure is drawn as a cavity field only: {why}")

    if panel is None:
        return plot_mode_grid(nodes, elements, pressure_shapes, frequencies,
                              max_modes=max_modes, title=title, show=show)

    count = min(max_modes, frequencies.size)
    faces = extract_surface_faces(elements)
    fig = plt.figure(figsize=(9.0, 3.6 * count + 0.6))
    for row in range(count):
        ax_s = fig.add_subplot(count, 2, 2 * row + 1, projection="3d")
        plot_panel_mode(panel, structure_shapes[:blocks["n_s"], row],
                        blocks["idx_s"], nodes.shape[0], ax=ax_s,
                        cmap=SIGNED_CMAP,
                        title=f"mode {row + 1}, {frequencies[row]:.1f} Hz"
                              "  ·  structure")
        for axis in (ax_s.xaxis, ax_s.yaxis, ax_s.zaxis):
            axis.set_major_locator(plt.MaxNLocator(4))
            axis.label.set_color(INK)
        ax_s.tick_params(colors=MUTED, labelsize=8)
        ax_s.title.set_color(INK)

        shape = pressure_shapes[:, row]
        peak = np.abs(shape).max() or 1.0
        ax_a = fig.add_subplot(count, 2, 2 * row + 2, projection="3d")
        plot_pressure_3d(nodes, elements, shape / peak, part="real",
                         cmap=SIGNED_CMAP, clim=(-1, 1), faces=faces,
                         ax=ax_a, colorbar=False, show=False,
                         title=f"mode {row + 1}, {frequencies[row]:.1f} Hz"
                               "  ·  cavity")
        ax_a.set_axis_off()
        ax_a.title.set_color(INK)

    fig.subplots_adjust(left=0.02, right=0.88, bottom=0.02,
                        top=0.93 if title else 0.96, wspace=0.05, hspace=0.22)
    bar = fig.colorbar(
        cm.ScalarMappable(norm=plt.Normalize(-1, 1), cmap=SIGNED_CMAP),
        cax=fig.add_axes([0.91, 0.30, 0.015, 0.40]),
    )
    bar.set_label("each field on its own peak", color=INK)
    bar.outline.set_edgecolor(GRID)
    bar.ax.tick_params(colors=MUTED)
    if title:
        fig.suptitle(title, y=0.995, color=INK)
    if show:
        plt.show()
    return fig


def plot_matrix(matrix, *, row_labels=None, col_labels=None, title=None,
                label="", cmap=AMPLITUDE_CMAP, gamma=0.5, annotate=True,
                normalize=True, xlabel="", ylabel="", figsize=None,
                show=False):
    """
    A rectangular matrix as a picture, with the small entries readable.

    For the matrices that say *which talks to which*: the modal coupling
    of a coupled model, a correlation table, a participation table. One
    hue, light to dark, because the quantity shown is a magnitude and
    zero is the bottom of the scale rather than a place in the middle.

    Parameters
    ----------
    matrix : ndarray (m, n)
        Non-negative. A signed matrix is passed through ``abs`` first by
        the caller, and said so in the label.
    row_labels, col_labels : sequence, optional
    title, label : str, optional
        ``label`` names the colour bar.
    cmap : str, optional
    gamma : float, optional
        Exponent of the power-law scale; below one it stretches the low
        end, which is where the interesting near-zeros are.
    normalize : bool, optional
        Divide by the largest entry, so the scale runs 0 to 1 and the
        colour bar says the same thing as the numbers in the cells.
        Default True: these matrices are read as ratios, not in their
        own units.
    annotate : bool, optional
        Write the value in each cell. Turned off automatically on a
        matrix too large for the numbers to fit.
    xlabel, ylabel : str, optional
    figsize : tuple, optional
    show : bool, optional

    Returns
    -------
    matplotlib axes
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm

    matrix = np.asarray(matrix, dtype=float)
    rows, columns = matrix.shape
    peak = float(np.abs(matrix).max()) or 1.0
    if normalize:
        matrix = matrix / peak
        peak = 1.0

    fig, ax = plt.subplots(figsize=figsize or (0.6 * columns + 3.4,
                                               0.5 * rows + 2.4))
    norm = PowerNorm(gamma=gamma, vmin=0.0, vmax=peak)
    image = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(columns),
                  [str(v) for v in (col_labels or range(1, columns + 1))],
                  rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(rows),
                  [str(v) for v in (row_labels or range(1, rows + 1))],
                  fontsize=8)
    ax.set_xticks(np.arange(-0.5, columns, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.1)
    ax.tick_params(which="both", length=0, colors=MUTED, labelcolor=INK)
    for spine in ax.spines.values():
        spine.set_visible(False)

    if annotate and rows * columns <= 240:
        for i in range(rows):
            for j in range(columns):
                value = matrix[i, j] / peak
                if value >= 0.005:
                    ax.text(j, i, f"{matrix[i, j]:.2f}".lstrip("0"),
                            ha="center",
                            va="center", fontsize=7.5,
                            color="white" if norm(matrix[i, j]) > 0.55
                            else "0.25")

    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel(ylabel, color=INK)
    if title:
        ax.set_title(title, fontsize=10, color=INK)
    bar = fig.colorbar(image, ax=ax, shrink=0.9)
    bar.set_label(label, color=INK)
    bar.outline.set_edgecolor(GRID)
    bar.ax.tick_params(colors=MUTED)
    fig.tight_layout()
    if show:
        plt.show()
    return ax


def plot_shares(labels, shares, *, series_names, title=None, xlabel="share",
                figsize=None, show=False):
    """
    What each item is made of: one stacked bar per item, two or more parts.

    The form for "how much of this mode is the plate and how much is the
    air": a magnitude that adds up to one, read along a common baseline.
    The parts keep the categorical colours in a fixed order, so the same
    part is the same colour from one figure to the next, and a 2 px gap
    between segments keeps the boundary visible without a border.

    Parameters
    ----------
    labels : sequence of str
        One per bar, top to bottom.
    shares : ndarray (n_items, n_parts)
        Rows summing to one (they are normalised here if they do not).
    series_names : sequence of str
        What each part is; a legend is always drawn, since identity is
        never colour alone.
    title, xlabel : str, optional
    figsize : tuple, optional
    show : bool, optional

    Returns
    -------
    matplotlib axes
    """
    import matplotlib.pyplot as plt

    shares = np.asarray(shares, dtype=float)
    total = shares.sum(axis=1, keepdims=True)
    shares = np.divide(shares, np.where(total == 0, 1.0, total))
    n_items, n_parts = shares.shape

    fig, ax = plt.subplots(figsize=figsize or (8.4, 0.42 * n_items + 1.8))
    y = np.arange(n_items)[::-1]
    left = np.zeros(n_items)
    for k in range(n_parts):
        ax.barh(y, shares[:, k], left=left, height=0.62,
                color=SERIES[k % len(SERIES)], label=series_names[k],
                edgecolor="white", linewidth=2.0)
        left = left + shares[:, k]

    for row, item in enumerate(shares):
        # One number per bar, the part that dominates it; a number on
        # every segment is a table, not a picture.
        k = int(np.argmax(item))
        ax.text(1.01, y[row], f"{item[k] * 100:.0f}% {series_names[k]}",
                va="center", fontsize=8, color=MUTED)

    ax.set_yticks(y, list(labels), fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel(xlabel, color=INK)
    ax.grid(True, axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    legend = ax.legend(frameon=False, fontsize=8, ncols=n_parts,
                       loc="lower center", bbox_to_anchor=(0.5, 1.0))
    for text in legend.get_texts():
        text.set_color(INK)
    if title:
        ax.set_title(title, pad=26, color=INK)
    fig.tight_layout()
    if show:
        plt.show()
    return ax
