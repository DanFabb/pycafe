"""
Vibroacoustic coupling test geometry: rigid box cavity + flexible plate.

Builds the canonical coupled case with gmsh:

- acoustic cavity Lx x Ly x Lz, structured HEXA8 mesh (transfinite);
- structural plate on the top face z = Lz, meshed with QUAD4 that are
  EXACTLY the top faces of the hexahedra (conforming mesh: plate nodes
  coincide with the fluid interface nodes, same Gmsh node tags).

Physical groups:
- dim 3  "fluid"       : the acoustic volume
- dim 2  "plate"       : top surface (structural domain AND coupling interface)
- dim 2  "rigid_walls" : the other 5 faces (acoustically rigid)
- dim 1  "plate_clamp" : the 4 edges of the plate (clamped boundary)

Fully scriptable — no prompts. Run:

    python coupling_geometry/make_box_plate_mesh.py

The geometry itself now lives in
:func:`pycafe.create_geom.library.box_with_plate`, together with the
other parametric models; this script stays as the entry point that
writes ``box_plate.msh`` next to itself.
"""

import pathlib
import sys


def make_box_plate_mesh(
    Lx=0.8,
    Ly=0.6,
    Lz=0.5,
    nx=8,
    ny=6,
    nz=5,
    output_path=None,
    show_gui=False,
    verbose=True,
):
    """
    Generate the box+plate coupling mesh and write it to ``output_path``.

    Parameters
    ----------
    Lx, Ly, Lz : float
        Cavity dimensions [m]. The plate lies on z = Lz.
    nx, ny, nz : int
        Number of elements along each direction.
    output_path : str or pathlib.Path
        Destination ``.msh`` file. Defaults to ``box_plate.msh`` next
        to this script.
    show_gui : bool
        If True, open the Gmsh GUI before finalizing.
    verbose : bool
        Print the summary and let gmsh print its meshing progress.

    Returns
    -------
    pathlib.Path
        Path of the written mesh file.
    """
    if output_path is None:
        output_path = pathlib.Path(__file__).parent / "box_plate.msh"

    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
    from pycafe.create_geom import box_with_plate

    output_path = box_with_plate(
        Lx=Lx, Ly=Ly, Lz=Lz, nx=nx, ny=ny, nz=nz,
        output_path=output_path, verbose=verbose,
    )
    if show_gui:
        import gmsh

        gmsh.initialize()
        try:
            gmsh.open(str(output_path))
            gmsh.fltk.run()
        finally:
            gmsh.finalize()

    return output_path


if __name__ == "__main__":
    make_box_plate_mesh()
