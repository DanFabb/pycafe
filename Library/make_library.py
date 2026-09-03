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
for.

**Every case but one is cut to the same number: 1500 Hz at six elements
per acoustic wavelength**, i.e. no fluid element longer than 38.1 mm. The
exception is ``cavity_plate_FG``, which keeps Fahy & Gardonio's own
16 x 16 x 8 mesh because reproducing their eighteen coupled modes means
reproducing their mesh; it is quoted at 500 Hz. Rebuild a case at a
higher ``f_max`` when a study needs more.
"""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from pycafe.core.model_spec import (  # noqa: E402
    AIR, Library, ModelSpec, aluminium, build_mesh,
)

# Six per wavelength at 1500 Hz is what this folder guarantees, so it is
# the number the specs are built with rather than the default ten: ten at
# 1500 Hz would be 22.9 mm, finer than anything here needs to be.
ELEMENTS_PER_WAVELENGTH = 6
TARGET_F_MAX = 1500.0

# THE ONE RULE THIS FOLDER IS CUT TO
#
# Every mesh here resolves 1500 Hz at six elements per acoustic
# wavelength. That fixes the element size once, for the whole folder:
#
#     h = c0 / (n_lambda f_max) = 343 / (6 * 1500) = 38.1 mm
#
# so no fluid element in any of these files is longer than 38.1 mm on its
# longest side, and frequency_limits() reports f_floor >= 1500 Hz for all
# of them. Six per wavelength is the floor pyCAFE refuses to go below, not
# a comfortable number: the dispersion of the trilinear element goes as
# 164 / n_lambda^2, about 4.6% at six. Thirteen per wavelength is the 1%
# figure, and it is what frequency_limits() calls f_1pct - it lands near
# 690 Hz on a 38.1 mm mesh, so a run that wants 1% accuracy across the
# whole band still has to say so with a lower f_max.
#
# ONE CASE IS OUT: cavity_plate_FG. Its element counts are Fahy &
# Gardonio's own (16 x 16 x 8, h = 45 mm), because reproducing their
# eighteen coupled modes means reproducing their mesh too. It is quoted at
# 500 Hz and refused above 1270 Hz, and it is the modal case only - the
# forced run uses cavity_plate_FG_directresponse, which is cut to the rule.
#
# WHAT THE RULE DOES NOT COVER: the plate. A thin panel has a *bending*
# wavelength, lambda_b = 2 pi (D / (m omega^2))^(1/4), far shorter than the
# acoustic one, and 1 mm aluminium (D = 6.55 N m, m = 2.7 kg/m2) is at
# lambda_b = 81 mm at 1500 Hz - two elements across it on a 38.1 mm mesh.
# So 1500 Hz here is a statement about the fluid, and the cases that were
# already finer than the rule for the plate's sake are left finer: their
# counts are pinned below, with the reason.

# Not part of the first release: pyCAFE reads gmsh meshes, and the CAD
# files and the deck it can also read live in Library/WIP.
STEP = HERE / "WIP" / "Tubo_1m_1m.stp"

# name -> (geometry, f_max [Hz], what it is for)
CASES = {
    "box_cavity": (
        # 33.3 mm, finer than the 38.1 mm the rule asks for, so the counts
        # are pinned rather than derived: notebook 01 prints the error of
        # every mode against the closed form, and those numbers move if the
        # mesh does. Refused above 1715 Hz.
        Library("box_cavity", Lx=0.4, Ly=0.3, Lz=0.5, nx=12, ny=9, nz=15),
        1500.0,
        "rigid air box, CHEXA8: the analytical modal case of notebook 01, "
        "whose twelfth mode is at 924 Hz",
    ),
    "box_with_plate": (
        # x and y are the plate's plane and are cut for its *bending*
        # wavelength, not the acoustic one: 25.8 and 26.1 mm, which keeps
        # six elements across the 156 mm the 1 mm plate has at 400 Hz. z is
        # the fluid depth and was six elements of 83 mm, which is what held
        # this mesh at 686 Hz; 14 elements of 35.7 mm bring it to the rule.
        Library("box_with_plate", Lx=0.8, Ly=0.6, Lz=0.5, nx=31, ny=23, nz=14),
        1500.0,
        "the same box closed by a flexible plate on z = Lz, conforming",
    ),
    "cavity_plate_FG": (
        # THE ONE CASE OUTSIDE THE 1500 Hz RULE. Fahy & Gardonio, Sound
        # and Structural Vibration, 2nd ed.: the worked coupled example.
        # 45 mm elements, so it is refused above 1270 Hz and quoted at
        # 500 Hz. Both the box and its mesh are theirs -
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
        # The same box as cavity_plate_FG, cut fine enough for the forced
        # run. The plate is what binds here: 1 mm aluminium has a bending
        # wavelength of 99 mm at 1000 Hz, so 13 mm elements in the plane of
        # the plate leave 7.6 across it. That is already well past the
        # folder rule, and the depth at 22.5 mm is too - 15 elements per
        # acoustic wavelength at 1000 Hz - so the counts stay as they are
        # and the mesh is refused only above 2541 Hz.
        Library("box_with_plate", Lx=0.414, Ly=0.314, Lz=0.360,
                nx=32, ny=24, nz=16),
        1500.0,
        "the Fahy & Gardonio box on a mesh cut past 1500 Hz, for the direct "
        "forced response rather than the book's eighteen modes",
    ),
    "duct_2d": (
        Library("duct_2d", L=1.0, H=0.1),
        1500.0,
        "2D duct with inlet and outlet: plane waves, impedance ends",
    ),
    "plate": (
        # No fluid in this one, so the folder rule has nothing to bind on;
        # the size follows it anyway, to keep one number for the folder.
        Library("plate", Lx=0.4, Ly=0.3),
        1500.0,
        "a flat shell on its own, clamped edge, for structural checks",
    ),
    "duct_with_flush_plate": (
        # This builder takes the size straight from the spec, so the rule
        # sets it: 167 mm elements before, 38.1 mm now, on a 3 x 1 x 1 m
        # duct - about 57000 hexahedra.
        Library("duct_with_flush_plate", cad_path=str(STEP),
                plate_length=1.0, plate_face="z_min"),
        1500.0,
        "3 m duct from CAD with a 1 m panel let into the floor",
    ),
    "box_faces_plate": (
        # The same coupled box as box_with_plate, but every wall is its own
        # physical group: a run can then be asked what each of the five
        # faces does instead of being handed one "rigid_walls" to take or
        # leave. block_cavity is not one of the builders the spec can size
        # itself, so the counts are worked out by hand from the same
        # 38.1 mm: 38.1, 37.5 and 35.7 mm, against the 40 and 50 mm that
        # held this mesh at 1143 Hz.
        Library("block_cavity", x=[0.0, 0.8], y=[0.0, 0.6], z=[0.0, 0.5],
                nx=[21], ny=[16], nz=[14], plate=("z", 0.5)),
        1500.0,
        "box closed by a plate on z = Lz, with every wall a group of its own",
    ),
    "l_cavity_plate": (
        # An L-shaped room: one arm twice as tall, the plate closing the
        # tall arm. Three blocks of one lattice, so the corner where they
        # meet shares its nodes and the mesh stays hexahedral. Counts by
        # hand again, to the same 38.1 mm: 37.5 mm across each 0.6 m span
        # and 36.4 mm across each 0.4 m one, against a flat 40 mm before.
        Library("block_cavity", x=[0.0, 0.6, 1.2], y=[0.0, 0.6],
                z=[0.0, 0.4, 0.8], cells=[(0, 0, 0), (1, 0, 0), (1, 0, 1)],
                nx=[16, 16], ny=[16], nz=[11, 11], plate=("z", 0.8)),
        1500.0,
        "L-shaped cavity, plate on the ceiling of the tall arm",
    ),
    "step_duct_plate": (
        # A duct that changes section halfway, with a panel let into the
        # floor of the low part: the classic flush-panel case on a geometry
        # that is not a plain box. 25 mm elements, already finer than the
        # 38.1 mm the rule asks for, so the counts stand: refused only
        # above 2287 Hz.
        Library("block_cavity", x=[0.0, 0.6, 1.2, 2.0], y=[0.0, 0.4],
                z=[0.0, 0.2, 0.5],
                cells=[(0, 0, 0), (1, 0, 0), (2, 0, 0), (2, 0, 1)],
                nx=[24, 24, 32], ny=[16], nz=[8, 12],
                plate=("z", 0.0, {"x": (0.6, 1.2)})),
        1500.0,
        "duct with a step in section and a panel let into its floor",
    ),
    "t_cavity_plate": (
        # A T in plan: a corridor with a branch, the plate closing the top
        # of the branch. 25 mm elements, finer than the rule already, so
        # the counts stand: refused only above 2287 Hz.
        Library("block_cavity", x=[0.0, 0.4, 0.8, 1.2], y=[0.0, 0.4, 0.8],
                z=[0.0, 0.4],
                cells=[(0, 0, 0), (1, 0, 0), (2, 0, 0), (1, 1, 0)],
                nx=[16, 16, 16], ny=[16, 16], nz=[16],
                plate=("z", 0.4, {"y": (0.4, 0.8)})),
        1500.0,
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
            elements_per_wavelength=ELEMENTS_PER_WAVELENGTH,
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
