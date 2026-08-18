# `Library/` — everything a run can be pointed at

One folder, three kinds of file: the meshes pyCAFE builds itself, the CAD files
it is given and meshes on the spot, and the Nastran decks it reads as they
stand. `ask_for_geometry()` opens its dialog here, so the choice between "a
mesh" and "a geometry" is made in the dialog and nowhere else — the extension of
what you pick decides how it is read.

## Geometries we are given

| file | what it is | read as |
|---|---|---|
| `Tubo_1m_1m.stp` | 3 × 1 × 1 m duct, one six-sided solid — meshable in hexahedra, so it can carry a coupled model (notebook 03) | `CadFile`, meshed here |
| `Vibro_ac1.step` | the COMSOL *Acoustic-Structure Interaction* geometry: a 60 mm sphere of water with a 10 × 20 mm cylinder cut out of it (notebook 05); drawn in millimetres, so `units="mm"` | `CadFile`, meshed here |
| `demo_cavity.bdf` | small vibroacoustic deck: 0.40 × 0.30 × 0.50 m air cavity (CHEXA, PID 101) with a 2 mm aluminium plate on `z = 0` (CQUAD4, PID 202), clamped by `SPC1 10` | `NastranFile`, already meshed |

## Geometries pyCAFE builds

One mesh per entry of `pycafe.create_geom.library.GEOMETRIES`, built once so
that a run can pick a `.msh` instead of meshing anything.

| file | what it is | built for |
|---|---|---|
| `box_cavity.msh` | rigid air box 0.40 × 0.30 × 0.50 m, CHEXA8 — the analytical modal case of notebook 01, whose twelfth mode is at 924 Hz | 1000 Hz |
| `box_with_plate.msh` | the same box 0.80 × 0.60 × 0.50 m closed by a flexible plate on `z = Lz`; the plate quadrilaterals *are* faces of the fluid hexahedra, so the interface is conforming | 400 Hz |
| `duct_2d.msh` | 2D duct 1.0 × 0.1 m with `inlet` and `outlet` — plane waves, impedance ends | 500 Hz |
| `plate.msh` | a flat 0.40 × 0.30 m shell on its own, clamped edge, for structural checks | 500 Hz |
| `duct_with_flush_plate.msh` | the 3 m duct of `Tubo_1m_1m.stp` with a 1 m panel let into the floor, cut into structured blocks | 200 Hz |

**The frequency a mesh was built for is the frequency it is good for.** Each
case is stated as a `ModelSpec`, so the element size is not written by hand: it
follows from that `f_max` and the elements-per-wavelength rule,
$h = c_0 / (f_{max} n_\lambda)$. Handing one of these files to a spec with a
higher `f_max` does not refine it — the validator simply reports how far it
still resolves.

## Rebuilding, and adding a case

```bash
python Library/make_library.py
```

`CASES` in that script is the whole content of this folder: a geometry, the
frequency it is built for, and one line on what it is for. Add an entry and run
it again.

## Using one

```python
from pycafe.core.model_spec import AIR, MeshFile, ModelSpec, aluminium, build_model

spec = ModelSpec(
    geometry=MeshFile("Library/box_with_plate.msh"),
    fluid=AIR,
    structure=aluminium(t=2e-3, support="clamped"),
    f_max=200.0,
)
model = build_model(spec)
```

or let the user pick it — `examples/00_master_geometry.ipynb` is the notebook
that explains the whole geometry stage, this folder included.
