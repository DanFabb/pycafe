r"""
Build every geometry pyCAFE makes itself, once, into this directory.

The library builders take parameters and write a mesh; ``Library/`` is
where the result of each of them is kept — next to the CAD files and the
Nastran decks, so that one folder holds everything a run can be pointed
at and the file dialog has one place to open. One mesh per entry of
:data:`pycafe.create_geom.library.GEOMETRIES`.

Run it from anywhere::

    python Library/make_library.py

Each case is stated as a :class:`~pycafe.core.model_spec.ModelSpec`, so
the element size is not written here either: it follows from the
``f_max`` of the case, and that frequency is the one the mesh is good
for. Rebuild a case at a higher ``f_max`` when a study needs more.
"""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from pycafe.core.model_spec import (  # noqa: E402
    AIR, Library, ModelSpec, aluminium, build_mesh,
)

# WHAT SETS THE FREQUENCY EACH CASE IS QUOTED AT
#
# Two rules, and the smaller of the two wins:
#
#   fluid      13 elements per acoustic wavelength for ~1% frequency error,
#              f = c0 / (13 h); six is the floor pyCAFE refuses to go below.
#   structure  six elements per *bending* wavelength of the plate,
#              lambda_b = 2 pi (D / (m omega^2))^(1/4), which is much
#              shorter than the acoustic one and is what actually binds on
#              a coupled mesh.
#
# For 1 mm aluminium (D = 6.55 N m, m = 2.7 kg/m2) in air:
#
#   h = 40 mm   plate up to  170 Hz   fluid up to  660 Hz
#   h = 25 mm   plate up to  435 Hz   fluid up to 1055 Hz
#
# So the two block cases below are cut at 25 mm and quoted at 400 Hz: the
# plate is the limit, the fluid has room to spare. A thicker plate is
# stiffer and moves its limit up; the number quoted here assumes the
# thinnest plate these meshes are meant to carry.

# Not part of the first release: pyCAFE reads gmsh meshes, and the CAD
# files and the deck it can also read live in Library/WIP.
STEP = HERE / "WIP" / "Tubo_1m_1m.stp"

# name -> (geometry, f_max [Hz], what it is for)
CASES = {
    "box_cavity": (
        Library("box_cavity", Lx=0.4, Ly=0.3, Lz=0.5),
        1000.0,
        "rigid air box, CHEXA8: the analytical modal case of notebook 01, "
        "whose twelfth mode is at 924 Hz",
    ),
    "box_with_plate": (
        # The acoustic rule sizes the depth, but not the two directions the
        # plate lives in: a 1 mm aluminium plate has a bending wavelength of
        # 156 mm at 400 Hz, five times shorter than the acoustic one, so x
        # and y are given here to keep six plate elements across it.
        Library("box_with_plate", Lx=0.8, Ly=0.6, Lz=0.5, nx=31, ny=23, nz=6),
        400.0,
        "the same box closed by a flexible plate on z = Lz, conforming",
    ),
    "cavity_plate_FG": (
        # Fahy & Gardonio, Sound and Structural Vibration, 2nd ed.: the
        # worked coupled example. Both the box and its mesh are theirs -
        # 0.414 x 0.314 x 0.360 m, 16 x 16 x 8 hexahedral acoustic
        # elements and 16 x 16 plate elements on z = Lz - so the element
        # counts are pinned here rather than derived from f_max, which
        # is the only way the frequencies below can be quoted against
        # the book.
        Library("box_with_plate", Lx=0.414, Ly=0.314, Lz=0.360,
                nx=16, ny=16, nz=8),
        500.0,
        "the Fahy & Gardonio coupled example: 0.414 x 0.314 x 0.360 m air "
        "box closed by a 1 mm CCCC aluminium plate, on their own mesh",
    ),
    "cavity_plate_FG_directresponse": (
        # The same box as cavity_plate_FG, cut finer so that the direct
        # frequency response can be run to 1000 Hz. The plate is what
        # binds: 1 mm aluminium has a bending wavelength of 99 mm at
        # 1000 Hz, so 13 mm elements leave 7.6 across it, above the six
        # pyCAFE refuses below. The fluid has room to spare - 22.5 mm
        # elements are 15 per acoustic wavelength at 1000 Hz, inside the
        # 1% rule - so the depth is cut half as fine as the plate plane.
        Library("box_with_plate", Lx=0.414, Ly=0.314, Lz=0.360,
                nx=32, ny=24, nz=16),
        1000.0,
        "the Fahy & Gardonio box on a mesh cut for 1000 Hz, for the direct "
        "forced response rather than the book's eighteen modes",
    ),
    "duct_2d": (
        Library("duct_2d", L=1.0, H=0.1),
        500.0,
        "2D duct with inlet and outlet: plane waves, impedance ends",
    ),
    "plate": (
        Library("plate", Lx=0.4, Ly=0.3),
        500.0,
        "a flat shell on its own, clamped edge, for structural checks",
    ),
    "duct_with_flush_plate": (
        Library("duct_with_flush_plate", cad_path=str(STEP),
                plate_length=1.0, plate_face="z_min"),
        200.0,
        "3 m duct from CAD with a 1 m panel let into the floor",
    ),
    "box_faces_plate": (
        # The same coupled box as box_with_plate, but every wall is its own
        # physical group: a run can then be asked what each of the five
        # faces does instead of being handed one "rigid_walls" to take or
        # leave. 40 mm elements keep 5 to 6 across the bending wavelength
        # of the plate at 200 Hz.
        Library("block_cavity", x=[0.0, 0.8], y=[0.0, 0.6], z=[0.0, 0.5],
                nx=[20], ny=[15], nz=[10], plate=("z", 0.5)),
        200.0,
        "box closed by a plate on z = Lz, with every wall a group of its own",
    ),
    "l_cavity_plate": (
        # An L-shaped room: one arm twice as tall, the plate closing the
        # tall arm. Three blocks of one lattice, so the corner where they
        # meet shares its nodes and the mesh stays hexahedral.
        Library("block_cavity", x=[0.0, 0.6, 1.2], y=[0.0, 0.6],
                z=[0.0, 0.4, 0.8], cells=[(0, 0, 0), (1, 0, 0), (1, 0, 1)],
                nx=[15, 15], ny=[15], nz=[10, 10], plate=("z", 0.8)),
        200.0,
        "L-shaped cavity, plate on the ceiling of the tall arm",
    ),
    "step_duct_plate": (
        # A duct that changes section halfway, with a panel let into the
        # floor of the low part: the classic flush-panel case on a geometry
        # that is not a plain box. 25 mm elements: see the note below on
        # what sets the frequency this mesh is quoted at.
        Library("block_cavity", x=[0.0, 0.6, 1.2, 2.0], y=[0.0, 0.4],
                z=[0.0, 0.2, 0.5],
                cells=[(0, 0, 0), (1, 0, 0), (2, 0, 0), (2, 0, 1)],
                nx=[24, 24, 32], ny=[16], nz=[8, 12],
                plate=("z", 0.0, {"x": (0.6, 1.2)})),
        400.0,
        "duct with a step in section and a panel let into its floor",
    ),
    "t_cavity_plate": (
        # A T in plan: a corridor with a branch, the plate closing the top
        # of the branch. 25 mm elements, quoted at 400 Hz.
        Library("block_cavity", x=[0.0, 0.4, 0.8, 1.2], y=[0.0, 0.4, 0.8],
                z=[0.0, 0.4],
                cells=[(0, 0, 0), (1, 0, 0), (2, 0, 0), (1, 1, 0)],
                nx=[16, 16, 16], ny=[16, 16], nz=[16],
                plate=("z", 0.4, {"y": (0.4, 0.8)})),
        400.0,
        "T-shaped cavity, plate on the ceiling of the branch",
    ),
}


def build_all(output_dir=HERE, verbose=True, only=None):
    """Build every case (or the ones named in ``only``) and return their paths."""
    output_dir = pathlib.Path(output_dir)
    built = {}
    wanted = set(only) if only else set(CASES)
    for name, (geometry, f_max, _what) in CASES.items():
        if name not in wanted:
            continue
        spec = ModelSpec(
            geometry=geometry,
            fluid=AIR,
            structure=aluminium(t=2.0e-3, support="clamped"),
            f_max=f_max,
            work_dir=output_dir,
            output_path=output_dir / f"{name}.msh",
        )
        path, report = build_mesh(spec)
        built[name] = pathlib.Path(path)
        if verbose:
            print(f"{name}: {path}")
            print(report)
            print()
    return built


if __name__ == "__main__":
    build_all()
