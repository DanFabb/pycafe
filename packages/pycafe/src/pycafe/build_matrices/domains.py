# domains.py
#
# Association of mesh domains from Gmsh physical groups.
#
# The global `elements` dict returned by the mesh loader mixes every
# element of a given type regardless of role: on a coupled mesh, the
# plate QUAD4 and the hard-wall QUAD4 land in the same
# "Quadrilateral 4" array. Physical groups are the authoritative
# domain information: this module maps named groups to registered
# element types and returns per-domain connectivity for assembly.
#
# Naming convention (see docs and coupling_geometry/README.md):
#   fluid domain     : group named "fluid" (or "acoustic"; legacy 2D
#                      meshes use "domain")
#   structural domain: group named "plate", "structure" or "shell"
import numpy as np

from ..create_geom.conventions import (
    FLUID_GROUP_NAMES,
    STRUCTURE_GROUP_NAMES,
)
from .assembly import assemble_KM
from .element_registry import (
    ELEMENT_TYPES,
    find_acoustic_elements,
    load_element_plugins,
)


def _match_group_elements(group, field):
    """
    Match the element types of a physical group against the registry.

    Returns (elem_type, spec, conn0) for the highest-dimension
    registered type of the given field found in the group.
    """
    load_element_plugins()

    matches = []
    for gmsh_name, conn in group["elements"].items():
        for type_name, spec in ELEMENT_TYPES.items():
            if spec.field != field:
                continue
            if conn.shape[1] != spec.n_nodes:
                continue
            if any(n in gmsh_name for n in spec.gmsh_names):
                matches.append((type_name, spec, conn))
                break

    if not matches:
        return None

    matches.sort(key=lambda m: m[1].dim, reverse=True)
    elem_type, spec, conn = matches[0]
    return elem_type, spec, np.asarray(conn, dtype=int) - 1   # 0-based


def identify_domains(groups):
    """
    Identify fluid and structural domains from Gmsh physical groups.

    Parameters
    ----------
    groups : dict
        Physical groups with connectivity, as returned by
        :func:`pycafe.create_geom.visualize_mesh.load_mesh_with_groups`
        (``groups[name] = {"dim", "tag", "nodes", "elements"}``).

    Returns
    -------
    domains : dict
        ``domains["fluid"]`` and (if present) ``domains["structure"]``,
        each ``{"group": group_name, "elem_type": registry_key,
        "spec": ElementSpec, "conn0": 0-based connectivity,
        "nodes": 1-based node tags of the group}``.

    Raises
    ------
    RuntimeError
        If no fluid domain can be identified, or a named group
        contains no registered element type.
    """
    domains = {}

    for role, names, field in (
        ("fluid", FLUID_GROUP_NAMES, "acoustic"),
        ("structure", STRUCTURE_GROUP_NAMES, "structural"),
    ):
        group_name = next(
            (g for g in groups if g.lower() in names), None
        )
        if group_name is None:
            continue
        group = groups[group_name]
        match = _match_group_elements(group, field)
        if match is None:
            raise RuntimeError(
                f"Physical group '{group_name}' ({role}) contains no "
                f"registered {field} element type. Group element types: "
                f"{list(group['elements'].keys())}"
            )
        elem_type, spec, conn0 = match
        domains[role] = {
            "group": group_name,
            "elem_type": elem_type,
            "spec": spec,
            "conn0": conn0,
            "nodes": group["nodes"],
        }

    if "fluid" not in domains:
        raise RuntimeError(
            "No fluid domain found: expected a physical group named "
            f"one of {FLUID_GROUP_NAMES}. Available groups: "
            f"{list(groups.keys())}"
        )

    return domains


def build_KM_acoustic_domain(nodes, domains, c0, **kwargs):
    """
    Assemble the acoustic K, M from the fluid domain connectivity.

    Same output contract as ``build_KM_acoustic`` but restricted to
    the elements of the ``fluid`` physical group.
    """
    dom = domains["fluid"]
    kernel = dom["spec"].kernel
    if kernel is None:
        raise NotImplementedError(
            f"Element type '{dom['elem_type']}' has no kernel."
        )
    K, M, debug_data = assemble_KM(nodes, dom["conn0"], kernel, c0, **kwargs)
    return K, M, debug_data, dom["elem_type"]
