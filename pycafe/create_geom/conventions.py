"""
The mesh contract: what pyCAFE expects to find in a Gmsh file.

Everything pyCAFE knows about a mesh beyond raw coordinates comes from
**physical groups**. The element arrays returned by the loader mix roles
— on a coupled mesh the plate quads and the rigid-wall quads land in the
same ``"Quadrilateral 4"`` array — so the groups are the only thing that
says which elements are the fluid, which are the structure, and which
nodes are clamped.

Two kinds of group names exist:

* **role names**, listed below, which the solvers look for by name;
* **free names**, anything else, which become selectable boundaries for
  :class:`~pycafe.boundary_condition.acoustic_bc.AcousticBC`
  (``add_impedance("lined_wall", ...)``, ``add_velocity("inlet", ...)``).
  Any name works, as long as it is not one of the role names.

A rigid wall needs no group at all — it is the natural boundary
condition of the acoustic problem — but naming it is still useful for
post-processing and for switching it to a lined wall later.

This module is the single source of truth for those names; the domain
identification, the system preparation and the mesh validator all read
it from here.
"""

from typing import NamedTuple, Tuple


class RoleSpec(NamedTuple):
    """One role a physical group can play in a pyCAFE model."""

    role: str
    names: Tuple[str, ...]      # accepted group names, lower case
    dim_offset: int             # 0 = same dim as the model, -1 = boundary
    field: str                  # "acoustic", "structural" or "" (nodes only)
    what: str                   # one-line description


ROLES = (
    RoleSpec(
        role="fluid",
        names=("fluid", "acoustic", "domain"),
        dim_offset=0,
        field="acoustic",
        what="the acoustic domain: CHEXA8 in 3D, CQUAD4/8 in 2D",
    ),
    RoleSpec(
        role="structure",
        names=("plate", "structure", "shell"),
        dim_offset=0,
        field="structural",
        what="the structural domain (CQUAD4F shells); on a coupled mesh "
             "its elements are also the interface faces",
    ),
    RoleSpec(
        role="clamp",
        names=("plate_clamp", "clamp", "fixed"),
        dim_offset=-1,
        field="",
        what="nodes where the structure is supported (clamped or pinned, "
             "chosen by the 'support' argument, not by the name)",
    ),
)

ROLE_BY_NAME = {name: spec for spec in ROLES for name in spec.names}

# Kept as module constants for the code that imported them before this
# module existed.
FLUID_GROUP_NAMES = ROLE_BY_NAME["fluid"].names
STRUCTURE_GROUP_NAMES = ROLE_BY_NAME["plate"].names
CLAMP_GROUP_NAMES = ROLE_BY_NAME["clamp"].names

# Canonical name to write when generating a mesh: the first alias.
CANONICAL = {spec.role: spec.names[0] for spec in ROLES}

# Names suggested (never required) for the free groups, so that meshes
# from different scripts stay readable.
SUGGESTED_BOUNDARY_NAMES = {
    "rigid_walls": "acoustically rigid walls; no BC needed, but handy to name",
    "opening": "radiating or open face, usually given an anechoic impedance",
    "inlet": "driven face, usually a prescribed normal velocity",
    "outlet": "terminated face, usually an impedance",
    "lined_wall": "absorbing treatment",
}

# What each kind of analysis needs to find in the mesh.
REQUIRED_ROLES = {
    "acoustic": ("fluid",),
    "structural": ("structure",),
    "vibroacoustic": ("fluid", "structure"),
}


def role_of(group_name):
    """
    Role played by a physical group, or ``None`` for a free name.

    Parameters
    ----------
    group_name : str

    Returns
    -------
    str or None
        ``"fluid"``, ``"structure"``, ``"clamp"``, or ``None`` when the
        name is a free boundary name.

    Examples
    --------
    >>> role_of("plate"), role_of("Fluid"), role_of("lined_wall")
    ('structure', 'fluid', None)
    """
    spec = ROLE_BY_NAME.get(str(group_name).strip().lower())
    return None if spec is None else spec.role


def names_for(role):
    """Accepted group names for a role, canonical one first."""
    for spec in ROLES:
        if spec.role == role:
            return spec.names
    raise ValueError(f"Unknown role '{role}'. Known roles: "
                     f"{[s.role for s in ROLES]}.")


def find_group(groups, role):
    """
    Name of the group playing ``role`` in a loaded mesh, or ``None``.

    Parameters
    ----------
    groups : dict
        Physical groups from ``load_mesh_with_groups``.
    role : str
        ``"fluid"``, ``"structure"`` or ``"clamp"``.
    """
    accepted = names_for(role)
    return next((g for g in groups if str(g).strip().lower() in accepted), None)


def describe_conventions():
    """Printable summary of the contract — the answer to "what do I name it?"."""
    lines = ["Physical groups pyCAFE looks for by name:", ""]
    for spec in ROLES:
        lines.append(f"  {spec.names[0]:<12} {spec.what}")
        if len(spec.names) > 1:
            lines.append(f"  {'':<12} also accepted: "
                         f"{', '.join(spec.names[1:])}")
        lines.append("")
    lines.append("Any other group is a free boundary name, selectable in "
                 "AcousticBC. Common ones:")
    lines.append("")
    for name, what in SUGGESTED_BOUNDARY_NAMES.items():
        lines.append(f"  {name:<12} {what}")
    lines.append("")
    lines.append("Dimensions: the fluid and the structure carry the "
                 "elements of the domain itself;")
    lines.append("boundary groups carry elements one dimension below "
                 "(faces in 3D, edges in 2D),")
    lines.append("and the clamp group carries the nodes of the support "
                 "line.")
    return "\n".join(lines)
