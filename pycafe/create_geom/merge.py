r"""
Put two meshes of the same object into one file, without welding them.

A conforming vibroacoustic mesh comes out of a single meshing pass: the
panel elements *are* faces of the fluid elements. Some geometries cannot
be meshed that way — a curved fluid volume that only tetrahedra can
fill, next to a shell that pyCAFE can only assemble as quadrilaterals.
The way out is to mesh the two domains separately, each with the element
its kernel needs, and to let the coupling be interpolated afterwards by
:mod:`pycafe.build_matrices.coupling_nonconforming`.

That is what this module writes: one Gmsh file holding both meshes side
by side, their nodes renumbered so that **nothing is shared**. The
interface is deliberately non-conforming; welding the two node sets
would be wrong here, since they do not describe the same points.

.. code-block:: python

    merge_meshes(["cavity_fluid.msh", "panel_shell.msh"], "model.msh")

Only elements that belong to a physical group are copied. Everything
pyCAFE does with a mesh goes through those groups (see
:mod:`pycafe.create_geom.conventions`), so an element outside all of
them has no role to play, and carrying it over would only make the file
larger and the validator noisier.

The output is written in the MSH 2.2 ASCII format, which stores exactly
what is needed here — node coordinates, element connectivity, physical
names — and is read back by Gmsh, and therefore by the pyCAFE loader,
without conversion.
"""

import pathlib

import numpy as np

# Gmsh element name -> MSH 2.2 element type number. The node ordering is
# the same on both sides, so the connectivity is copied verbatim.
MSH2_TYPES = {
    "Line 2": 1,
    "Triangle 3": 2,
    "Quadrilateral 4": 3,
    "Tetrahedron 4": 4,
    "Hexahedron 8": 5,
    "Prism 6": 6,
    "Pyramid 5": 7,
    "Line 3": 8,
    "Triangle 6": 9,
    "Quadrilateral 9": 10,
    "Quadrilateral 8": 16,
}


def merge_meshes(sources, output_path, *, rename=None, verbose=False):
    """
    Concatenate several meshes into one file, keeping their nodes apart.

    Parameters
    ----------
    sources : sequence of str or Path
        The meshes to merge, in order. Their physical groups are kept
        under their own names; a name appearing in two sources ends up
        as a single group holding the elements of both.
    output_path : str or Path
        Where to write the merged mesh.
    rename : sequence of dict, optional
        One ``{old_name: new_name}`` per source, applied to the physical
        group names before merging — handy when both files call their
        domain ``fluid``.
    verbose : bool, optional
        Print what was merged.

    Returns
    -------
    pathlib.Path
        The file written.

    Raises
    ------
    ValueError
        If a source holds an element type that the MSH 2.2 writer does
        not know, or if no group survives the merge.

    Examples
    --------
    >>> merge_meshes(["fluid.msh", "plate.msh"], "model.msh")   # doctest: +SKIP
    PosixPath('model.msh')
    """
    from .visualize_mesh import load_mesh_with_groups

    sources = [pathlib.Path(s) for s in sources]
    rename = list(rename) if rename is not None else [{}] * len(sources)
    if len(rename) != len(sources):
        raise ValueError("'rename' must hold one mapping per source.")

    all_nodes = []
    # {name: (dim, [(gmsh element name, connectivity 1-based, offset)])}
    collected = {}
    offset = 0

    for source, mapping in zip(sources, rename):
        nodes, _elements, _boundaries, groups = load_mesh_with_groups(
            str(source), verbose=False
        )
        all_nodes.append(np.asarray(nodes, dtype=float))

        for name, group in groups.items():
            name = mapping.get(name, name)
            entry = collected.setdefault(name, (int(group["dim"]), []))
            for etype, conn in group["elements"].items():
                if etype not in MSH2_TYPES:
                    raise ValueError(
                        f"{source.name} holds '{etype}' elements, which the "
                        "MSH 2.2 writer does not know. Known types: "
                        f"{sorted(MSH2_TYPES)}."
                    )
                entry[1].append((etype, np.asarray(conn, dtype=int) + offset))
        offset += nodes.shape[0]
        if verbose:
            print(f"  {source.name}: {nodes.shape[0]} nodes, groups "
                  f"{sorted(groups)}")

    if not collected:
        raise ValueError(
            "None of the sources has a physical group, so the merged mesh "
            "would carry no role at all."
        )

    nodes = np.vstack(all_nodes)
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = ["$MeshFormat", "2.2 0 8", "$EndMeshFormat",
             "$PhysicalNames", str(len(collected))]
    tags = {}
    for tag, (name, (dim, _)) in enumerate(sorted(collected.items()), start=1):
        tags[name] = tag
        lines.append(f'{dim} {tag} "{name}"')
    lines.append("$EndPhysicalNames")

    lines += ["$Nodes", str(nodes.shape[0])]
    lines += [f"{i} {x:.17g} {y:.17g} {z:.17g}"
              for i, (x, y, z) in enumerate(nodes, start=1)]
    lines.append("$EndNodes")

    element_lines = []
    for name, (dim, blocks) in sorted(collected.items()):
        tag = tags[name]
        for etype, conn in blocks:
            code = MSH2_TYPES[etype]
            for row in conn:
                element_lines.append(
                    f"{len(element_lines) + 1} {code} 2 {tag} {tag} "
                    + " ".join(str(int(n)) for n in row)
                )
    lines += ["$Elements", str(len(element_lines))] + element_lines
    lines.append("$EndElements")

    output_path.write_text("\n".join(lines) + "\n")
    if verbose:
        print(f"  -> {output_path}: {nodes.shape[0]} nodes, "
              f"{len(element_lines)} elements, groups {sorted(collected)}")
    return output_path
