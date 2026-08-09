"""
Read a Gmsh mesh into arrays: nodes, connectivity, physical groups.

Gmsh, matplotlib and tkinter are imported inside the functions that use
them, not at module level: this module is reached by every import of
:mod:`pycafe.create_geom`, and a headless or minimal environment must be
able to state a model without a display or a Tk installation.
"""

import numpy as np
import pathlib

def select_msh_file():
    """Open a file dialog to select a .msh file."""
    from tkinter import filedialog, Tk

    root = Tk()
    root.withdraw()
    filepath = filedialog.askopenfilename(
        title="Select Gmsh .msh file",
        filetypes=[("Gmsh mesh files", "*.msh")]
    )
    if not filepath:
        raise ValueError("No mesh file selected.")
    return filepath

class NodeIndex:
    """
    Translate Gmsh node tags into row indices of the coordinate array.

    Gmsh node tags are arbitrary labels: they are only contiguous,
    1-based and sorted for meshes generated in one shot. A mesh that
    was partitioned, merged, refined or renumbered can carry sparse or
    out-of-order tags, in which case using a tag as an array index
    silently picks the wrong coordinates (or raises ``IndexError``),
    corrupting K and M.

    This class maps every tag to ``row + 1``, where ``row`` is the
    position of that node in the array returned by ``getNodes()``. The
    result keeps pyCAFE's **1-based connectivity contract** (so the
    ``elem - 1`` conversions downstream stay valid) while making it
    true by construction rather than by luck.

    When the tags already are ``1..N`` in order, the mapping is the
    identity and the arrays are returned untouched.
    """

    def __init__(self, node_tags):
        self._tags = np.asarray(node_tags, dtype=np.int64)
        n = self._tags.size
        self.identity = bool(
            n and np.array_equal(self._tags, np.arange(1, n + 1, dtype=np.int64))
        )
        if not self.identity:
            self._order = np.argsort(self._tags)
            self._sorted = self._tags[self._order]

    def __call__(self, tags):
        """Remap an array of node tags (any shape) to 1-based row indices."""
        arr = np.asarray(tags, dtype=np.int64)
        if self.identity or arr.size == 0:
            return arr
        pos = np.searchsorted(self._sorted, arr)
        np.clip(pos, 0, self._sorted.size - 1, out=pos)
        if not np.array_equal(self._sorted[pos], arr):
            missing = np.unique(arr[self._sorted[pos] != arr])
            raise KeyError(
                f"Node tags not present in the mesh: {missing.tolist()}"
            )
        return self._order[pos] + 1


def extract_physical_boundaries(dims=(1, 2), node_index=None):
    """
    Extract physical boundary groups: returns dict[name] = list of node indices.

    Scans physical groups of the given dimensions (default: curves and
    surfaces, so that both 2D meshes with line boundaries and 3D meshes
    with surface boundaries are supported). Node indices are 1-based
    (Gmsh convention). Interior nodes of each group entity are included
    (``includeBoundary=True`` on the closure).

    Parameters
    ----------
    node_index : NodeIndex, optional
        Mapping from Gmsh node tags to 1-based rows of the coordinate
        array (see :class:`NodeIndex`). Required for meshes whose tags
        are not ``1..N`` in order; without it the raw tags are returned.
    """
    import gmsh

    boundary_dict = {}
    for dim, tag in gmsh.model.getPhysicalGroups():
        if dim in dims:
            name = gmsh.model.getPhysicalName(dim, tag)
            entities = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)
            all_nodes = []
            for ent in entities:
                try:
                    node_tags, _, _ = gmsh.model.mesh.getNodes(
                        dim, ent, includeBoundary=True
                    )
                except:
                    continue
                if node_index is not None:
                    node_tags = node_index(node_tags)
                all_nodes.extend(np.asarray(node_tags, dtype=int).tolist())
            boundary_dict[name] = sorted(set(all_nodes))
    return boundary_dict


def extract_physical_groups_with_connectivity(node_index=None):
    """
    Extract all physical groups with their element connectivity.

    Unlike :func:`extract_physical_boundaries` (node lists only), this
    returns, for every physical group of any dimension, the
    connectivity of the elements belonging to the group. This is what
    surface integration needs (impedance/velocity on 3D boundary
    faces, coupling interface between fluid and structure): the faces,
    not just their nodes.

    Parameters
    ----------
    node_index : NodeIndex, optional
        Mapping from Gmsh node tags to 1-based rows of the coordinate
        array (see :class:`NodeIndex`). Required for meshes whose tags
        are not ``1..N`` in order; without it the raw tags are returned.

    Returns
    -------
    dict
        ``groups[name] = {"dim": d, "tag": t, "nodes": [1-based node indices],
        "elements": {gmsh_element_name: (n_elem, n_nodes) int array}}``
        Connectivity is 1-based (Gmsh convention), consistent with the
        ``elements`` dict returned by :func:`load_mesh_and_elements`.
    """
    import gmsh

    groups = {}
    for dim, tag in gmsh.model.getPhysicalGroups():
        name = gmsh.model.getPhysicalName(dim, tag)
        if not name:
            name = f"group_dim{dim}_tag{tag}"
        elif name in groups:
            name = f"{name}_dim{dim}_tag{tag}"
        entities = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)

        conn_by_type = {}
        all_nodes = set()

        for ent in entities:
            elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim, ent)
            for etype, enodes in zip(elem_types, elem_node_tags):
                ename, _, _, num_nodes, *_ = gmsh.model.mesh.getElementProperties(etype)
                conn = np.array(enodes, dtype=int).reshape(-1, num_nodes)
                if node_index is not None:
                    conn = node_index(conn)
                if ename in conn_by_type:
                    conn_by_type[ename] = np.vstack([conn_by_type[ename], conn])
                else:
                    conn_by_type[ename] = conn
                all_nodes.update(conn.ravel().tolist())

        groups[name] = {
            "dim": dim,
            "tag": tag,
            "nodes": sorted(all_nodes),
            "elements": conn_by_type,
        }
    return groups

def _gmsh_start(filepath, verbose):
    """
    Open a mesh with gmsh, silencing its terminal output when asked.

    gmsh prints an "Info :" line for every meshing step, which drowns
    the output of a notebook; `verbose=False` turns that off before
    anything is read.
    """
    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
    gmsh.open(filepath)


def load_mesh_and_elements(filepath, verbose=True):
    """
    Load a Gmsh mesh and extract nodes, elements, and boundary groups.

    Parameters
    ----------
    filepath : str
        Path of the ``.msh`` file.
    verbose : bool, optional
        Print the element summary and let gmsh print its own progress.

    Notes
    -----
    Connectivity and boundary node lists are expressed as 1-based row
    indices of ``nodes``, not as raw Gmsh tags: sparse or unordered
    tags are remapped by :class:`NodeIndex`.
    """
    import gmsh

    _gmsh_start(filepath, verbose)

    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    nodes = node_coords.reshape(-1, 3)
    node_index = NodeIndex(node_tags)
    elements = {}

    for etype in gmsh.model.mesh.getElementTypes():
        name, dim, order, num_nodes, *_ = gmsh.model.mesh.getElementProperties(etype)
        element_tags, element_nodes = gmsh.model.mesh.getElementsByType(etype)
        if len(element_tags) == 0:
            continue
        conn = node_index(np.array(element_nodes).reshape(-1, num_nodes))
        elements[name] = conn
        if verbose:
            print(f"→ {name:<15} | Dim: {dim} | Order: {order} | Nodes/Elem: {num_nodes} | Count: {conn.shape[0]}")

    boundaries = extract_physical_boundaries(node_index=node_index)
    gmsh.finalize()
    return nodes, elements, boundaries


def load_mesh_with_groups(filepath, verbose=True):
    """
    Load a Gmsh mesh with full physical-group information.

    Same as :func:`load_mesh_and_elements`, but additionally returns
    the physical groups with their element connectivity (see
    :func:`extract_physical_groups_with_connectivity`) — needed for
    surface boundary integration on 3D meshes and for identifying
    the fluid/structure coupling interface.

    Returns
    -------
    nodes : ndarray (N, 3)
    elements : dict[gmsh_element_name -> (n_elem, n_nodes) int array]
        Connectivity, 1-based row indices of ``nodes`` (see
        :class:`NodeIndex`).
    boundaries : dict[name -> list of 1-based node indices]
    groups : dict
        Physical groups with dim, tag, nodes and per-type connectivity.
    """
    import gmsh

    _gmsh_start(filepath, verbose)

    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    nodes = node_coords.reshape(-1, 3)
    node_index = NodeIndex(node_tags)
    elements = {}

    for etype in gmsh.model.mesh.getElementTypes():
        name, dim, order, num_nodes, *_ = gmsh.model.mesh.getElementProperties(etype)
        element_tags, element_nodes = gmsh.model.mesh.getElementsByType(etype)
        if len(element_tags) == 0:
            continue
        conn = node_index(np.array(element_nodes).reshape(-1, num_nodes))
        elements[name] = conn
        if verbose:
            print(f"→ {name:<15} | Dim: {dim} | Order: {order} | Nodes/Elem: {num_nodes} | Count: {conn.shape[0]}")

    boundaries = extract_physical_boundaries(node_index=node_index)
    groups = extract_physical_groups_with_connectivity(node_index=node_index)
    gmsh.finalize()
    return nodes, elements, boundaries, groups

def plot_2d_mesh(nodes, elements):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    plotted = False

    for name, conn in elements.items():
        if "Line" in name:
            for e in conn:
                pts = nodes[e[[0, -1]] - 1, :2]
                ax.plot(pts[:, 0], pts[:, 1], 'k-', lw=0.5)
            plotted = True

        elif "Triangle" in name:
            for e in conn:
                pts = nodes[e[:3] - 1, :2]
                pts = np.vstack([pts, pts[0]])
                ax.plot(pts[:, 0], pts[:, 1], 'b-', lw=0.5)
            plotted = True

        elif "Quadrangle" in name or "Quadrilateral" in name:
            for e in conn:
                pts = nodes[e[:4] - 1, :2]
                pts = np.vstack([pts, pts[0]])
                ax.plot(pts[:, 0], pts[:, 1], 'r-', lw=0.5)
            plotted = True

    if plotted:
        ax.set_aspect("equal")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("2D Mesh Visualization")
        plt.grid(True)
        plt.show()
    else:
        print("⚠️ No 2D elements found to plot.")

if __name__ == "__main__":
    msh_file = select_msh_file()
    nodes, elements, boundaries = load_mesh_and_elements(msh_file)
    plot_2d_mesh(nodes, elements)

    # Save .npz next to the .msh
    output_path = pathlib.Path(msh_file).with_suffix('.npz')
    np.savez_compressed(
        output_path,
        nodes=nodes,
        elements=elements,
        boundaries=boundaries
    )
    print(f"✅ Mesh data saved to: {output_path}")
    
def print_available_boundaries(boundaries):
    print("\nAvailable boundary names in the mesh:")
    print("-------------------------------------")
    for name, nodes in boundaries.items():
        print(f"- {name:20s} | {len(nodes)} nodes")