"""
Explore how Gmsh physical groups arrive in pyCAFE for the coupled
box+plate mesh, and verify the properties the vibroacoustic coupling
matrix (Kc) will rely on.

Checks:
1. which element types the mesh loader returns;
2. node sets of each physical group (plate, rigid_walls, plate_clamp, fluid);
3. conformity: plate nodes are a subset of fluid nodes (same tags -> the
   structural and acoustic meshes share the interface nodes);
4. interface face connectivity: every plate QUAD4 is exactly the top
   face of one HEXA8 (same 4 node tags);
5. outward normal of the plate faces;
6. node tags are contiguous 1..N (assumption made by the mesh loader).

Run after make_box_plate_mesh.py:

    python coupling_geometry/explore_physical_groups.py
"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from pycafe.create_geom.visualize_mesh import load_mesh_with_groups  # noqa: E402

MESH = pathlib.Path(__file__).parent / "box_plate.msh"


def main():
    nodes, elements, boundaries, groups = load_mesh_with_groups(str(MESH))

    print("\n=== 1. Element types returned by the loader ===")
    for name, conn in elements.items():
        print(f"  {name:<18} {conn.shape}")

    print("\n=== 2. Physical groups ===")
    for name, g in groups.items():
        etypes = {k: v.shape for k, v in g["elements"].items()}
        print(f"  '{name}': dim={g['dim']} n_nodes={len(g['nodes'])} elements={etypes}")

    print("\n=== 3. Mesh conformity at the interface ===")
    fluid_nodes = set(groups["fluid"]["nodes"])
    plate_nodes = set(groups["plate"]["nodes"])
    clamp_nodes = set(groups["plate_clamp"]["nodes"])
    print(f"  plate ⊂ fluid nodes:  {plate_nodes <= fluid_nodes}")
    print(f"  clamp ⊂ plate nodes:  {clamp_nodes <= plate_nodes}")
    print(f"  |fluid|={len(fluid_nodes)}  |plate|={len(plate_nodes)}  |clamp|={len(clamp_nodes)}")

    print("\n=== 4. Plate faces vs hexa top faces ===")
    plate_conn = groups["plate"]["elements"]["Quadrilateral 4"]
    hexa_conn = groups["fluid"]["elements"]["Hexahedron 8"]
    hexa_face_sets = {}
    for ih, h in enumerate(hexa_conn):
        # le 6 facce di un hexa Gmsh (per set di nodi, l'ordine non conta qui)
        for face in ([0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
                     [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]):
            hexa_face_sets[frozenset(h[face])] = ih
    matched = sum(frozenset(q) in hexa_face_sets for q in plate_conn)
    print(f"  plate QUAD4 that are a hexa face: {matched}/{len(plate_conn)}")

    print("\n=== 5. Plate face normal (first face) ===")
    q = plate_conn[0] - 1  # 0-based
    x = nodes[q]
    n = np.cross(x[1] - x[0], x[3] - x[0])
    n /= np.linalg.norm(n)
    print(f"  normal of first plate face: {np.round(n, 6)}  (z=Lz face → expect ±[0,0,1])")

    print("\n=== 6. Node numbering ===")
    all_tags = sorted(set(int(t) for g in groups.values() for t in g["nodes"]))
    print(f"  total nodes in mesh array: {nodes.shape[0]}")
    print(f"  max tag in groups: {all_tags[-1]}  min: {all_tags[0]}")
    print(f"  contiguous 1..N: {all_tags[-1] == nodes.shape[0] and all_tags[0] == 1}")

    print("\n=== 7. Domain association (identify_domains) ===")
    from pycafe.build_matrices.domains import identify_domains
    domains = identify_domains(groups)
    for role, dom in domains.items():
        print(f"  {role:<10} <- group '{dom['group']}': "
              f"{dom['elem_type']} x {dom['conn0'].shape[0]} "
              f"({dom['spec'].dofs_per_node} DOF/nodo)")

    print("\nDone.")


if __name__ == "__main__":
    main()
