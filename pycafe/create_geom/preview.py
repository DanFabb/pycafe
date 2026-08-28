r"""
Look at the geometry before computing on it.

A model is wrong long before the solver says so: the panel ends up on the
wrong wall, a CAD file comes in millimetres, the group meant to be the
inlet is the outlet. All of that is visible in one picture and in none of
the numbers, so :func:`plot_geometry` is drawn **before** the matrices are
assembled — :func:`~pycafe.core.model_spec.build_model` calls it by
itself unless asked not to.

What the picture shows, by role rather than by element type:

======================  ===============================================
fluid                   the outer skin of the acoustic domain, in grey
                        and transparent, so what is inside stays visible
pml                     the absorbing layer, in a warmer grey
structure               the flexible part, opaque and coloured — it is
                        also the interface
clamp                   the supported nodes, as dots
free names              every other physical group (``inlet``,
                        ``opening``, ``lined_wall``, ...), one colour each
======================  ===============================================

The legend carries the group names actually found in the mesh, which is
the fastest check that the contract of
:mod:`pycafe.create_geom.conventions` was satisfied.
"""

import numpy as np

from .conventions import role_of

# Roles get fixed colours so that two models are read the same way; free
# boundary names take the next colour of this cycle.
ROLE_STYLE = {
    "fluid": {"facecolor": "0.80", "edgecolor": "0.45", "alpha": 0.18,
              "linewidth": 0.2},
    "pml": {"facecolor": "#d9c6a5", "edgecolor": "0.45", "alpha": 0.35,
            "linewidth": 0.2},
    "structure": {"facecolor": "#c0392b", "edgecolor": "k", "alpha": 0.95,
                  "linewidth": 0.3},
}
BOUNDARY_COLORS = ("#2980b9", "#27ae60", "#8e44ad", "#e67e22", "#16a085",
                   "#c0392b", "#7f8c8d")


def _faces_of(group):
    """Renderable faces of one physical group, 0-based, or an empty list."""
    from pycafe.post_processing.post_processing_3d import extract_surface_faces

    try:
        return extract_surface_faces(group["elements"])
    except ValueError:
        return []                      # nodes or line elements only


def _nodes_of(group):
    """0-based node indices of a group."""
    return np.asarray(group.get("nodes", []), dtype=int) - 1


def plot_geometry(
    nodes,
    groups,
    *,
    title=None,
    show_boundaries=True,
    style="roles",
    ax=None,
    elev=22,
    azim=-60,
    figsize=(9.5, 5.0),
    show=True,
):
    """
    Draw the mesh coloured by the role each physical group plays.

    Parameters
    ----------
    nodes : ndarray (N, 3) or (N, 2)
        Node coordinates.
    groups : dict
        Physical groups with connectivity, from
        :func:`~pycafe.create_geom.visualize_mesh.load_mesh_with_groups`.
    title : str, optional
        Figure title.
    show_boundaries : bool, optional
        Also draw the free-name groups (``inlet``, ``opening``, ...) as
        coloured patches. Default True; ``False`` keeps only fluid,
        structure and layer.
    style : {"roles", "mesh"}, optional
        ``"roles"`` fills each group with the colour of the role it
        plays, which is what a model is checked with. ``"mesh"`` fades
        the fills and draws every element edge instead, which is what a
        mesh is looked at with — same picture, different question. See
        :func:`plot_mesh_3d`.
    ax : matplotlib 3D axes, optional
        Draw into an existing axes instead of making a figure.
    elev, azim : float, optional
        View angles.
    figsize : tuple, optional
    show : bool, optional
        Call ``plt.show()``. Default True.

    Returns
    -------
    matplotlib axes

    Examples
    --------
    >>> from pycafe.create_geom import load_mesh_with_groups, plot_geometry
    >>> nodes, elements, boundaries, groups = load_mesh_with_groups(
    ...     "box_plate.msh", verbose=False)                # doctest: +SKIP
    >>> plot_geometry(nodes, groups, title="what am I solving?")  # doctest: +SKIP
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    nodes = np.asarray(nodes, dtype=float)
    if nodes.shape[1] == 2:            # a 2D mesh lives on z = 0
        nodes = np.column_stack([nodes, np.zeros(len(nodes))])

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    handles = []
    colour = iter(BOUNDARY_COLORS)

    # Draw from the back forwards: the fluid box first, the named faces on
    # it, the flexible part last, so the part that matters stays on top
    # instead of being covered by the wall it is let into.
    order = {"fluid": 0, "pml": 1, None: 2, "structure": 3}
    ordered = sorted(groups.items(),
                     key=lambda item: order.get(role_of(item[0]), 2))

    for name, group in ordered:
        role = role_of(name)
        if role == "clamp":
            continue                   # drawn as dots below
        if role is None and not show_boundaries:
            continue

        faces = _faces_of(group)
        if not faces:
            continue

        if role in ROLE_STYLE:
            patch = dict(ROLE_STYLE[role])
        else:
            patch = {"facecolor": next(colour, "#7f8c8d"), "edgecolor": "0.35",
                     "alpha": 0.22, "linewidth": 0.2}
        if style == "mesh":
            # The elements are the subject here: the fill is only there to
            # tell one group from another, and the edges carry the picture.
            patch = dict(patch, alpha=0.10 if role != "structure" else 0.35,
                         edgecolor="0.25", linewidth=0.45)

        ax.add_collection3d(Poly3DCollection(
            [nodes[f] for f in faces], **patch
        ))
        label = f"{name}" + (f"  ({role})" if role else "")
        handles.append(plt.Line2D([0], [0], marker="s", linestyle="",
                                  markerfacecolor=patch["facecolor"],
                                  markeredgecolor="k", markersize=8,
                                  label=label))

    for name, group in groups.items():
        if role_of(name) != "clamp":
            continue
        held = _nodes_of(group)
        if held.size:
            ax.scatter(nodes[held, 0], nodes[held, 1], nodes[held, 2],
                       s=9, c="k", depthshade=False)
            handles.append(plt.Line2D([0], [0], marker="o", linestyle="",
                                      color="k", markersize=5,
                                      label=f"{name}  (clamp)"))

    lo, hi = nodes.min(axis=0), nodes.max(axis=0)
    span = np.where((hi - lo) > 0, hi - lo, 1.0)
    # A 2D mesh is flat in z: give the degenerate direction a thickness,
    # or matplotlib complains about a singular transformation.
    flat = (hi - lo) <= 0
    lo, hi = np.where(flat, lo - 0.05 * span, lo), np.where(flat,
                                                            hi + 0.05 * span,
                                                            hi)
    ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), zlim=(lo[2], hi[2]),
           xlabel="x [m]", ylabel="y [m]", zlabel="z [m]")
    ax.set_box_aspect(span)
    ax.view_init(elev=elev, azim=azim)
    ax.tick_params(labelsize=7)
    if title:
        ax.set_title(title)
    if handles:
        # Outside the axes: on a 3D plot a legend box lands on the model.
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                  fontsize=8, framealpha=0.9)

    fig.tight_layout()
    if show:
        plt.show()
    return ax


def plot_mesh_3d(nodes, groups, **kwargs):
    """
    Draw the mesh in 3D: every element of every surface, edges and all.

    The same picture as :func:`plot_geometry` and a different question.
    That one asks *what is this model* — which group plays which role,
    what colour the panel is. This one asks *what is this mesh* — how
    many elements there are across the panel, whether the fluid grid
    lines up with it, where the mesh is fine and where it is not. So the
    fills fade and the element edges are drawn.

    Only the surfaces are drawn, because the outside of a volume mesh is
    made of them: the faces of the hexahedra inside would hide each other
    and cost a great deal to render.

    Parameters
    ----------
    nodes : ndarray (N, 3) or (N, 2)
    groups : dict
        Physical groups with connectivity, from
        :func:`~pycafe.create_geom.visualize_mesh.load_mesh_with_groups`.
    **kwargs
        Everything :func:`plot_geometry` takes (``title``, ``ax``,
        ``elev``, ``azim``, ``figsize``, ``show``).

    Returns
    -------
    matplotlib axes

    Examples
    --------
    >>> plot_mesh_3d(nodes, groups, title="what am I meshing?")  # doctest: +SKIP
    """
    kwargs.setdefault("title", "mesh")
    return plot_geometry(nodes, groups, style="mesh", **kwargs)
