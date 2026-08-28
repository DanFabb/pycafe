"""
Geometry and mesh handling: the standard Gmsh workflow of pyCAFE.

Seven modules, one per step:

=========================================  =====================================
:mod:`~pycafe.create_geom.ask`             where is the file? (``uigetfile``)
:mod:`~pycafe.create_geom.conventions`     what to name the physical groups
:mod:`~pycafe.create_geom.gmsh_workflow`   how to write a geometry script
:mod:`~pycafe.create_geom.library`         ready-made parametric geometries
:mod:`~pycafe.create_geom.external`        CAD files and foreign meshes
:mod:`~pycafe.create_geom.nastran`         Nastran bulk data decks
:mod:`~pycafe.create_geom.validation`      is this mesh usable?
=========================================  =====================================

The short version:

.. code-block:: python

    from pycafe.create_geom import (
        box_with_plate, describe_mesh, load_mesh_with_groups,
    )

    msh = box_with_plate(Lx=0.4, Ly=0.3, Lz=0.5, nx=12, ny=9, nz=12)
    describe_mesh(msh, f_max=500.0)
    nodes, elements, boundaries, groups = load_mesh_with_groups(str(msh))

and, for geometry that comes from somewhere else,
:func:`~pycafe.create_geom.external.inspect_cad` to see what a STEP file
holds, then :func:`~pycafe.create_geom.external.cad_to_mesh` or
:func:`~pycafe.create_geom.external.retag_mesh`.

To state a whole model in one place — geometry, materials and the
frequency that sizes the mesh — see
:class:`pycafe.core.model_spec.ModelSpec`.

:func:`~pycafe.create_geom.conventions.describe_conventions` prints the
naming contract at any time.
"""

from .ask import (
    ACOUSTIC_BOUNDARY_CONDITIONS,
    ACOUSTIC_POINT_SOURCES,
    FILE_KINDS,
    ask_for_acoustic_bc,
    ask_for_choice,
    ask_for_many,
    ask_for_file,
    ask_for_geometry,
    ask_for_number,
    ask_for_point,
    ask_for_points,
    ask_for_structure,
)
from .conventions import (
    CANONICAL,
    REQUIRED_ROLES,
    ROLES,
    describe_conventions,
    find_group,
    names_for,
    role_of,
)
from .external import (
    EntityInfo,
    TagRule,
    assign_groups,
    by_tag,
    cad_to_mesh,
    everything,
    import_cad,
    in_box,
    inspect_cad,
    largest,
    list_groups,
    on_plane,
    rest,
    retag_mesh,
)
from .gmsh_workflow import GmshModel
from .nastran import (
    BdfDeck,
    NastranMaterial,
    NastranProperty,
    SpcSet,
    bdf_to_mesh,
    free_faces,
    inspect_bdf,
    read_bdf,
    tag_faces,
)
from .library import (
    GEOMETRIES,
    box_cavity,
    box_with_plate,
    build,
    duct_2d,
    duct_with_flush_plate,
    plate,
)
from .preview import plot_geometry, plot_mesh_3d
from .validation import (
    MeshReport,
    describe_mesh,
    frequency_limits,
    validate_mesh,
)
from .visualize_mesh import (
    NodeIndex,
    load_mesh_and_elements,
    load_mesh_with_groups,
    open_in_gmsh,
    plot_2d_mesh,
)

__all__ = [
    # conventions
    "ROLES", "CANONICAL", "REQUIRED_ROLES",
    "role_of", "names_for", "find_group", "describe_conventions",
    # workflow
    "GmshModel",
    # preview
    "plot_geometry",
    "plot_mesh_3d",
    # library
    "GEOMETRIES", "build",
    "box_cavity", "box_with_plate", "duct_2d", "duct_with_flush_plate",
    "plate",
    # external geometry
    "TagRule", "EntityInfo", "assign_groups", "cad_to_mesh",
    "import_cad", "inspect_cad",
    "on_plane", "in_box", "by_tag", "largest", "everything", "rest",
    "list_groups", "retag_mesh",
    # asking the user where a file is
    "ACOUSTIC_BOUNDARY_CONDITIONS", "ACOUSTIC_POINT_SOURCES",
    "FILE_KINDS", "ask_for_acoustic_bc", "ask_for_choice",
    "ask_for_many",
    "ask_for_point_source",
    "ask_for_file", "ask_for_geometry",
    "ask_for_number",
    "ask_for_point",
    "ask_for_points",
    "ask_for_structure",
    # Nastran bulk data
    "BdfDeck", "NastranProperty", "NastranMaterial", "SpcSet",
    "read_bdf", "inspect_bdf", "bdf_to_mesh", "free_faces", "tag_faces",
    # validation
    "MeshReport", "validate_mesh", "describe_mesh", "frequency_limits",
    # loading
    "NodeIndex", "load_mesh_with_groups", "load_mesh_and_elements",
    "open_in_gmsh", "plot_2d_mesh",
]
