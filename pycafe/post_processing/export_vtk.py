import os
import numpy as np
import xml.etree.ElementTree as ET


def _promote_nodes_to_3d(nodes):
    pts = nodes.copy().astype(np.float64)
    if pts.shape[1] == 2:
        pts = np.column_stack([pts, np.zeros(pts.shape[0])])
    return pts


def _build_meshio_cells(elements):
    if not elements:
        return []
    cells = []
    skip_keywords = ("Line", "Point", "Triangle")
    quad4_keywords = ("Quadrilateral 4", "Quadrangle 4", "quad4")
    quad8_keywords = ("Quadrilateral 8", "Quadrangle 8", "quad8")
    for key, conn in elements.items():
        if any(kw in key for kw in skip_keywords):
            continue
        conn0 = np.asarray(conn, dtype=np.int64) - 1
        if conn0.ndim == 1:
            conn0 = conn0.reshape(1, -1)
        ncols = conn0.shape[1]
        if any(kw in key for kw in quad4_keywords) or ncols == 4:
            cells.append(("quad", conn0))
        elif any(kw in key for kw in quad8_keywords) or ncols == 8:
            cells.append(("quad8", conn0))
    return cells


def _write_pvd(pvd_path, vtu_filenames, time_values):
    root = ET.Element("VTKFile", type="Collection", version="0.1", byte_order="LittleEndian")
    collection = ET.SubElement(root, "Collection")
    for fname, t in zip(vtu_filenames, time_values):
        ET.SubElement(collection, "DataSet",
                      timestep=f"{t:.6g}",
                      group="",
                      part="0",
                      file=os.path.basename(fname))
    tree = ET.ElementTree(root)
    ET.indent(tree)
    with open(pvd_path, "wb") as fh:
        tree.write(fh, xml_declaration=True, encoding="utf-8")
    return pvd_path


def export_pressure_vtu(nodes, elements, p_full, frequencies, output_dir="."):
    """
    Export frequency-domain pressure results to VTU files + PVD collection.

    Parameters
    ----------
    nodes : ndarray (N, 2) or (N, 3)
    elements : dict  (from load_mesh_and_elements)
    p_full : ndarray (N, N_freq) complex128
    frequencies : ndarray (N_freq,) float64  [Hz]
    output_dir : str

    Returns
    -------
    str  path of the .pvd collection file
    """
    import meshio

    os.makedirs(output_dir, exist_ok=True)
    pts3d = _promote_nodes_to_3d(nodes)
    cells = _build_meshio_cells(elements)

    if not cells:
        print("Warning: no quad elements found; VTU files will contain point data only.")

    vtu_paths = []
    for i, f in enumerate(frequencies):
        p = p_full[:, i]
        point_data = {
            "pressure_real":  np.real(p).astype(np.float64),
            "pressure_imag":  np.imag(p).astype(np.float64),
            "pressure_abs":   np.abs(p).astype(np.float64),
            "pressure_phase": np.angle(p).astype(np.float64),
        }
        fname = os.path.join(output_dir, f"pressure_f{i:04d}_{f:.2f}Hz.vtu")
        meshio.write(fname, meshio.Mesh(points=pts3d, cells=cells, point_data=point_data))
        vtu_paths.append(fname)

    pvd_path = os.path.join(output_dir, "pressure_results.pvd")
    _write_pvd(pvd_path, vtu_paths, frequencies.tolist())

    print(f"Exported {len(vtu_paths)} VTU files to '{output_dir}/'")
    print(f"PVD collection: {pvd_path}")
    return pvd_path


def export_modes_vtu(nodes, elements, modes_red, freqs, idx_free, p0_nodes, output_dir="."):
    """
    Export modal analysis results to VTU files + PVD collection.

    Parameters
    ----------
    nodes : ndarray (N, 2) or (N, 3)
    elements : dict
    modes_red : ndarray (N_red, N_modes)
    freqs : ndarray (N_modes,) float64  [Hz]
    idx_free : ndarray  0-based free DOF indices
    p0_nodes : ndarray  0-based constrained node indices
    output_dir : str

    Returns
    -------
    str  path of the .pvd collection file
    """
    import meshio
    from pycafe.build_matrices.assembly_cquad8 import expand_mode_to_full

    os.makedirs(output_dir, exist_ok=True)
    pts3d = _promote_nodes_to_3d(nodes)
    cells = _build_meshio_cells(elements)

    if not cells:
        print("Warning: no quad elements found; VTU files will contain point data only.")

    num_nodes = nodes.shape[0]
    num_modes = modes_red.shape[1]
    vtu_paths = []

    for k in range(num_modes):
        mode_full = expand_mode_to_full(modes_red[:, k], idx_free, p0_nodes, num_nodes)
        mode_real = np.real(mode_full).astype(np.float64)
        peak = np.max(np.abs(mode_real))
        if peak > 1e-30:
            mode_real /= peak

        point_data = {"mode_shape": mode_real}
        fname = os.path.join(output_dir, f"mode_{k:04d}_{freqs[k]:.4f}Hz.vtu")
        meshio.write(fname, meshio.Mesh(points=pts3d, cells=cells, point_data=point_data))
        vtu_paths.append(fname)

    pvd_path = os.path.join(output_dir, "modal_results.pvd")
    _write_pvd(pvd_path, vtu_paths, freqs.tolist())

    print(f"Exported {num_modes} mode VTU files to '{output_dir}/'")
    print(f"PVD collection: {pvd_path}")
    return pvd_path
