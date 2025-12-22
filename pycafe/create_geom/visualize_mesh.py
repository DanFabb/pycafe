import gmsh
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog, Tk
import pathlib

def select_msh_file():
    """Open a file dialog to select a .msh file."""
    root = Tk()
    root.withdraw()
    filepath = filedialog.askopenfilename(
        title="Select Gmsh .msh file",
        filetypes=[("Gmsh mesh files", "*.msh")]
    )
    if not filepath:
        raise ValueError("No mesh file selected.")
    return filepath

def extract_physical_boundaries():
    """Extract physical 1D boundaries: returns dict[name] = list of node indices."""
    boundary_dict = {}
    for dim, tag in gmsh.model.getPhysicalGroups():
        if dim == 1:
            name = gmsh.model.getPhysicalName(dim, tag)
            entities = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)
            all_nodes = []
            for ent in entities:
                try:
                    node_tags, _, _ = gmsh.model.mesh.getNodes(dim, ent)
                    all_nodes.extend(node_tags.tolist())
                except:
                    continue
            boundary_dict[name] = sorted(set(all_nodes))
    return boundary_dict

def load_mesh_and_elements(filepath):
    """Load a Gmsh mesh and extract nodes, elements, and boundary groups."""
    gmsh.initialize()
    gmsh.open(filepath)

    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    nodes = node_coords.reshape(-1, 3)
    elements = {}

    for etype in gmsh.model.mesh.getElementTypes():
        name, dim, order, num_nodes, *_ = gmsh.model.mesh.getElementProperties(etype)
        element_tags, element_nodes = gmsh.model.mesh.getElementsByType(etype)
        if len(element_tags) == 0:
            continue
        conn = np.array(element_nodes).reshape(-1, num_nodes)
        elements[name] = conn
        print(f"→ {name:<15} | Dim: {dim} | Order: {order} | Nodes/Elem: {num_nodes} | Count: {conn.shape[0]}")

    boundaries = extract_physical_boundaries()
    gmsh.finalize()
    return nodes, elements, boundaries

def plot_2d_mesh(nodes, elements):
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