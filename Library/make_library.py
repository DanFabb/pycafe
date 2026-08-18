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

STEP = HERE / "Tubo_1m_1m.stp"

# name -> (geometry, f_max [Hz], what it is for)
CASES = {
    "box_cavity": (
        Library("box_cavity", Lx=0.4, Ly=0.3, Lz=0.5),
        500.0,
        "rigid air box, CHEXA8: the analytical modal case",
    ),
    "box_with_plate": (
        Library("box_with_plate", Lx=0.8, Ly=0.6, Lz=0.5),
        400.0,
        "the same box closed by a flexible plate on z = Lz, conforming",
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
}


def build_all(output_dir=HERE, verbose=True):
    """Build every case and return ``{name: path}``."""
    output_dir = pathlib.Path(output_dir)
    built = {}
    for name, (geometry, f_max, _what) in CASES.items():
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
