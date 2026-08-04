"""
Element registry for pyCAFE.

Central catalogue of the finite element types known to pyCAFE.
Element types are identified by their Gmsh element name (e.g.
``"Quadrilateral 8"``), not by the number of nodes, so that types
with the same node count (CQUAD8 vs CHEXA8) do not collide.

Each entry is an :class:`ElementSpec`:

- ``gmsh_names`` : Gmsh element-name substrings that identify the type
- ``n_nodes``    : nodes per element
- ``dim``        : topological dimension of the element (2 or 3)
- ``field``      : ``"acoustic"`` (scalar pressure) or ``"structural"``
- ``dofs_per_node`` : degrees of freedom per node (1 for acoustic
  pressure; ``None`` if not yet defined)
- ``kernel``     : callable ``kernel(x_e, c) -> (K_e, M_e)`` computing
  the element matrices, or ``None`` if the kernel is not implemented yet

Registering a new element type means adding one entry to
``ELEMENT_TYPES`` — assembly and dispatching pick it up automatically.
"""

from typing import Callable, NamedTuple, Optional, Tuple

from .element_cquad4 import element_matrices_cquad4
from .element_cquad4f import element_matrices_cquad4f
from .element_cquad8 import element_matrices_cquad8
from .element_hex8 import element_matrices_hex8
from .element_tetra4 import element_matrices_tetra4


class ElementSpec(NamedTuple):
    gmsh_names: Tuple[str, ...]
    n_nodes: int
    dim: int
    field: str
    dofs_per_node: Optional[int]
    kernel: Optional[Callable]


ELEMENT_TYPES = {
    "CQUAD4": ElementSpec(
        gmsh_names=("Quadrilateral 4", "Quadrangle 4"),
        n_nodes=4,
        dim=2,
        field="acoustic",
        dofs_per_node=1,
        kernel=element_matrices_cquad4,
    ),
    "CQUAD8": ElementSpec(
        gmsh_names=("Quadrilateral 8", "Quadrangle 8"),
        n_nodes=8,
        dim=2,
        field="acoustic",
        dofs_per_node=1,
        kernel=element_matrices_cquad8,
    ),
    "CHEXA8": ElementSpec(
        gmsh_names=("Hexahedron 8",),
        n_nodes=8,
        dim=3,
        field="acoustic",
        dofs_per_node=1,
        kernel=element_matrices_hex8,
    ),
    "CTETRA4": ElementSpec(
        gmsh_names=("Tetrahedron 4",),
        n_nodes=4,
        dim=3,
        field="acoustic",
        dofs_per_node=1,
        kernel=element_matrices_tetra4,
    ),
    # Structural shell, Nastran/MacNeal style: selective reduced
    # integration plus residual bending flexibility for transverse
    # shear. 6 DOF/node; the kernel needs the properties
    # (t, rho, E, nu), to be passed through `kernel_args` in
    # `assemble_KM` together with `dofs_per_node=6`.
    "CQUAD4F": ElementSpec(
        gmsh_names=("Quadrilateral 4", "Quadrangle 4"),
        n_nodes=4,
        dim=2,
        field="structural",
        dofs_per_node=6,
        kernel=element_matrices_cquad4f,
    ),
}


def find_acoustic_elements(elements):
    """
    Identify the acoustic element type present in a mesh.

    Matches the keys of the ``elements`` dictionary (Gmsh element
    names) against the registered acoustic element types. When a mesh
    contains both volume and surface elements (e.g. a hexahedral mesh
    whose boundary faces appear as ``"Quadrilateral 4"``), the type
    with the highest topological dimension is selected: lower-dim
    elements are boundary entities, not the acoustic domain.

    Parameters
    ----------
    elements : dict
        Mapping of Gmsh element names to connectivity arrays of shape
        ``(n_elem, n_nodes)``, as returned by the mesh loader.

    Returns
    -------
    elem_type : str
        Registry key of the detected type (e.g. ``"CQUAD8"``).
    elem_key : str
        Key in ``elements`` holding the corresponding connectivity.
    spec : ElementSpec
        Registry entry for the detected type.

    Raises
    ------
    RuntimeError
        If no registered acoustic element type is found in the mesh.
    """
    matches = []

    for type_name, spec in ELEMENT_TYPES.items():
        if spec.field != "acoustic":
            continue
        for key, conn in elements.items():
            if getattr(conn, "ndim", None) != 2 or conn.shape[1] != spec.n_nodes:
                continue
            if any(gname in key for gname in spec.gmsh_names):
                matches.append((type_name, key, spec))
                break

    if not matches:
        raise RuntimeError(
            "No supported acoustic elements found in the mesh.\n"
            f"Registered acoustic types: "
            f"{[k for k, s in ELEMENT_TYPES.items() if s.field == 'acoustic']}\n"
            f"Mesh element groups: {list(elements.keys())}"
        )

    # Il dominio acustico è il tipo a dimensione massima presente.
    matches.sort(key=lambda m: m[2].dim, reverse=True)
    best = matches[0]

    # Mesh 3D con dominio non registrato (es. tetraedri): il match 2D
    # sarebbero solo le facce di bordo — avvisa chiaramente.
    if best[2].dim == 2:
        unregistered_3d = [
            key for key in elements
            if any(n in key for n in ("Hexahedron", "Tetrahedron", "Prism", "Pyramid"))
        ]
        if unregistered_3d:
            raise RuntimeError(
                f"Mesh contains 3D element groups {unregistered_3d} that are "
                "not registered as acoustic types; the 2D match "
                f"'{best[1]}' is likely boundary faces, not the acoustic "
                "domain. Register the 3D element type first."
            )

    return best


def get_kernel(elem_type):
    """
    Return the element-matrix kernel for a registered element type.

    Raises
    ------
    NotImplementedError
        If the type is registered but its kernel is not implemented yet.
    """
    spec = ELEMENT_TYPES[elem_type]
    if spec.kernel is None:
        raise NotImplementedError(
            f"Element type '{elem_type}' is registered but its kernel is "
            "not implemented yet (shape functions to be provided)."
        )
    return spec.kernel
