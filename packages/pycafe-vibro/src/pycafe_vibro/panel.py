r"""
The flexible panel of a coupled model: where it is, and how it moves.

A vibroacoustic mesh says which elements are structural, and nothing
else about them. Everything a plot needs — which plane the panel lies
in, which way its normal points, how its nodes fall on a grid — is in
the coordinates, so :func:`describe_panel` measures it once and hands
back a :class:`Panel` that the rest of a run reads instead of assuming
that a plate is flat in ``z``.

.. code-block:: python

    panel = describe_panel(nodes, groups["plate"]["nodes"])
    print(panel)                       # 0.800 x 0.600 m, normal +z
    plot_panel_mode(panel, w_red, idx_s, title="mode 1")

Only a flat panel on a structured grid is described here, which is what
a conforming interface with a hexahedral fluid gives. A curved or
unstructured one raises rather than being drawn wrong.
"""

from dataclasses import dataclass, field

import numpy as np

AXIS_NAMES = "xyz"


@dataclass
class Panel:
    """
    Where a flat structural surface is, and the grid its nodes fall on.

    Attributes
    ----------
    nodes0 : ndarray (n,)
        0-based indices of the panel nodes, in the mesh numbering.
    coordinates : ndarray (n, 3)
        Their coordinates, kept so that a field can be scattered onto
        the grid without the mesh being passed around again.
    normal : int
        Axis the panel does not spread along, 0, 1 or 2.
    plane : tuple of int
        The two axes it does spread along, in order.
    position : float
        Where it sits on the normal axis [m].
    sides : tuple of float
        Its two sides [m], along ``plane``.
    counts : tuple of int
        Elements along each side.
    grid : tuple of ndarray
        ``(u1, u2)``, the grid lines along ``plane``.
    """

    nodes0: np.ndarray
    coordinates: np.ndarray
    normal: int
    plane: tuple
    position: float
    sides: tuple
    counts: tuple
    grid: tuple
    _index: tuple = field(repr=False, default=())

    @property
    def spacing(self):
        """Element size along each side [m]."""
        return tuple(s / n for s, n in zip(self.sides, self.counts))

    @property
    def area(self):
        """Area of the panel [m2]."""
        return self.sides[0] * self.sides[1]

    @property
    def normal_name(self):
        """``"x"``, ``"y"`` or ``"z"``."""
        return AXIS_NAMES[self.normal]

    @property
    def plane_names(self):
        """The names of the two in-plane axes."""
        return tuple(AXIS_NAMES[i] for i in self.plane)

    @property
    def mesh(self):
        """``(X, Y)`` from :func:`numpy.meshgrid` on the two sides."""
        u1, u2 = self.grid
        return np.meshgrid(u1, u2)

    def field(self, values, component=None):
        """
        Scatter a nodal field onto the panel grid.

        Parameters
        ----------
        values : ndarray (num_nodes,) or (num_nodes, k)
            A field over the whole mesh. With two dimensions,
            ``component`` says which column to take; left out, it is the
            one along the normal.
        component : int, optional

        Returns
        -------
        ndarray (n2, n1)
            Real part of the field on the grid.
        """
        values = np.asarray(values)
        if values.ndim == 2:
            column = self.normal if component is None else int(component)
            values = values[:, column]
        i1, i2 = self._index
        u1, u2 = self.grid
        out = np.zeros((u2.size, u1.size))
        out[i2, i1] = np.real(values[self.nodes0])
        return out

    def centre(self):
        """Coordinates of the centre of the panel [m]."""
        return self.coordinates.mean(axis=0)

    def __str__(self):
        a, b = self.sides
        n1, n2 = self.counts
        return (f"{a:.3f} x {b:.3f} m in the "
                f"{self.plane_names[0]}-{self.plane_names[1]} plane at "
                f"{self.normal_name} = {self.position:.3f} m, "
                f"{n1} x {n2} elements")


def describe_panel(nodes, panel_nodes, *, tolerance=1e-6):
    """
    Measure where a structural surface is and how its nodes are laid out.

    Parameters
    ----------
    nodes : ndarray (N, 3)
        Mesh node coordinates.
    panel_nodes : array-like
        Node tags of the structural group, **1-based** as they come out
        of :func:`~pycafe.create_geom.visualize_mesh.load_mesh_with_groups`.
    tolerance : float, optional
        How flat is flat: the smallest spread is accepted as the normal
        direction when it is below ``tolerance`` times the largest.

    Returns
    -------
    Panel

    Raises
    ------
    ValueError
        If the group is not a flat surface — the case worth stopping
        on, since every plot below would otherwise be drawn on a plane
        that does not exist.

    Examples
    --------
    >>> panel = describe_panel(nodes, groups["plate"]["nodes"])  # doctest: +SKIP
    >>> panel.normal_name                                        # doctest: +SKIP
    'z'
    """
    nodes = np.asarray(nodes, dtype=float)
    nodes0 = np.unique(np.asarray(panel_nodes, dtype=int) - 1)
    coordinates = nodes[nodes0]

    spread = coordinates.max(axis=0) - coordinates.min(axis=0)
    normal = int(np.argmin(spread))
    if spread[normal] > tolerance * spread.max():
        raise ValueError(
            "the group is not a flat surface: its nodes spread "
            f"{np.round(spread, 6)} m along x, y and z. A panel is one face; "
            "several faces taken together have no single normal to move along."
        )

    plane = tuple(i for i in range(3) if i != normal)
    lines = []
    index = []
    for axis in plane:
        values, inverse = np.unique(np.round(coordinates[:, axis], 9),
                                    return_inverse=True)
        lines.append(values)
        index.append(inverse)

    return Panel(
        nodes0=nodes0,
        coordinates=coordinates,
        normal=normal,
        plane=plane,
        position=float(coordinates[:, normal].mean()),
        sides=(float(spread[plane[0]]), float(spread[plane[1]])),
        counts=(lines[0].size - 1, lines[1].size - 1),
        grid=(lines[0], lines[1]),
        _index=(index[0], index[1]),
    )


def panel_displacement(panel, w_red, idx_s, num_nodes):
    """
    Normal displacement of the panel, on its own grid.

    Parameters
    ----------
    panel : Panel
    w_red : ndarray (n_s,)
        Reduced structural DOFs, from a modal or forced solution.
    idx_s : ndarray (n_s,)
        Their global DOF indices, ``blocks["idx_s"]`` or
        ``result["idx_s"]``.
    num_nodes : int
        Number of nodes in the mesh.

    Returns
    -------
    ndarray (n2, n1)
    """
    from .solver_vibroacoustic import structural_displacement_field

    field_full = structural_displacement_field(w_red, idx_s, num_nodes)
    return panel.field(field_full)


def plot_panel_mode(panel, w_red, idx_s, num_nodes, *, ax=None, title=None,
                    normalize=True, cmap="RdBu_r", show=False):
    """
    Draw the panel deformed by one shape, on the plane it lies in.

    The axes are labelled with the two in-plane directions of the panel
    and the height with its normal one, so the same call draws a top
    wall and a side wall without an argument changing.

    Parameters
    ----------
    panel : Panel
    w_red, idx_s, num_nodes
        As in :func:`panel_displacement`.
    ax : matplotlib 3D axes, optional
    title : str, optional
    normalize : bool, optional
        Scale to a maximum of one. A modal shape has no unit of its own,
        so this is the default; a forced response in metres is drawn as
        it is with ``normalize=False``.
    cmap : str, optional
    show : bool, optional

    Returns
    -------
    matplotlib axes
    """
    import matplotlib.pyplot as plt

    un = panel_displacement(panel, w_red, idx_s, num_nodes)
    peak = np.abs(un).max()
    if normalize and peak > 0:
        un = un / peak

    if ax is None:
        fig = plt.figure(figsize=(5.5, 4.2))
        ax = fig.add_subplot(111, projection="3d")

    X, Y = panel.mesh
    limit = np.abs(un).max() or 1.0
    ax.plot_surface(X, Y, un, cmap=cmap, vmin=-limit, vmax=limit,
                    linewidth=0.15, edgecolor="k", alpha=0.95)
    label = "u" if not normalize else "u (norm.)"
    ax.set(xlabel=f"{panel.plane_names[0]} [m]",
           ylabel=f"{panel.plane_names[1]} [m]",
           zlabel=f"${label}_{panel.normal_name}$",
           zlim=(-1.1 * limit, 1.1 * limit))
    if title:
        ax.set_title(title)
    if show:
        plt.show()
    return ax
