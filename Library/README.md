# `Library/` — every mesh a run can be pointed at

One folder of gmsh meshes, which is what this release reads.
`ask_for_geometry()` opens its dialog here, filtered on `.msh`, so picking a
model is picking a file and nothing else has to be said about it: a mesh
carries its own element sizes and its own physical group names.

## What is not in this release

Nothing here but meshes: **this release reads gmsh meshes and nothing else**.
The CAD files and the Nastran deck pyCAFE can also read are in `WIP/`, out of
the file dialog.

| file | what it is | read as |
|---|---|---|
| `WIP/Tubo_1m_1m.stp` | 3 × 1 × 1 m duct, one six-sided solid, meshable in hexahedra | `CadFile`, **work in progress** |
| `WIP/Vibro_ac1.step` | the COMSOL *Acoustic-Structure Interaction* geometry: a 60 mm sphere of water with a 10 × 20 mm cylinder cut out of it; drawn in millimetres | `CadFile`, **work in progress** |
| `WIP/demo_cavity.bdf` | small vibroacoustic deck: 0.40 × 0.30 × 0.50 m air cavity (CHEXA, PID 101) with a 2 mm aluminium plate on `z = 0` (CQUAD4, PID 202), clamped by `SPC1 10` | `NastranFile`, **work in progress** |

## Geometries pyCAFE builds

One mesh per entry of `pycafe.create_geom.library.GEOMETRIES`, built once so
that a run can pick a `.msh` instead of meshing anything.

| file | what it is | built for |
|---|---|---|
| `box_cavity.msh` | hard-walled air box 0.40 × 0.30 × 0.50 m, CHEXA8 — the analytical modal case of notebook 01, whose twelfth mode is at 924 Hz | 1000 Hz |
| `box_with_plate.msh` | the same box 0.80 × 0.60 × 0.50 m closed by a flexible plate on `z = Lz` — the general coupled box, used by notebook 03; the plate quadrilaterals *are* faces of the fluid hexahedra, so the interface is conforming, and `x`/`y` are meshed for the bending wavelength of the plate rather than the acoustic one | 400 Hz |
| `cavity_plate_FG.msh` | **Fahy & Gardonio's coupled example**: 0.414 × 0.314 × 0.360 m air box closed by a 1 mm CCCC aluminium plate on `z = Lz`, on the book's own mesh — 16 × 16 × 8 `CHEXA8` and 16 × 16 `CQUAD4`. The element counts are pinned, not derived from `f_max`: they are part of the case being reproduced. Notebook `02_cavity_plate_coupled_FG` | 500 Hz |
| `cavity_plate_FG_directresponse.msh` | **the same Fahy & Gardonio box, cut for a forced run**: 0.414 × 0.314 × 0.360 m, 32 × 24 × 16 `CHEXA8` and 32 × 24 `CQUAD4F`, conforming. The plate sets the size — a bending wavelength of 99 mm at 1000 Hz against 343 mm of sound — so 13 mm elements in the plane of the plate and 22.5 mm through the depth. Notebook `02_cavity_plate_coupled_FG_directresponse` | 1000 Hz |
| `duct_2d.msh` | 2D duct 1.0 × 0.1 m with `inlet` and `outlet` — plane waves, impedance ends | 500 Hz |
| `plate.msh` | a flat 0.40 × 0.30 m shell on its own, clamped edge, for structural checks | 500 Hz |
| `duct_with_flush_plate.msh` | the 3 m duct of `WIP/Tubo_1m_1m.stp` with a 1 m panel let into the floor, cut into structured blocks | 200 Hz |
| `box_faces_plate.msh` | the same coupled box, 0.80 × 0.60 × 0.50 m, with **every wall its own physical group** (`wall_x_0mm`, `wall_z_0mm`, …): a run can be asked what each face does, one question per face | 200 Hz |
| `l_cavity_plate.msh` | L-shaped room 1.2 × 0.6 m in plan, one arm twice as tall, plate closing the ceiling of the tall arm; 7 wall groups | 200 Hz |
| `step_duct_plate.msh` | duct 2 m long that changes section halfway, with a 0.6 m panel let into the floor of the low part; 8 wall groups, 25 mm elements, 19125 nodes | 400 Hz |
| `t_cavity_plate.msh` | T-shaped cavity, a corridor with a branch, plate on the ceiling of the branch; 9 wall groups, 25 mm elements, 18785 nodes | 400 Hz |

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

The last four are built by `block_cavity`, which fills cells of a rectangular
lattice: any L, step or T that can be written that way comes out structured,
hexahedral and conforming, and **each outer plane becomes a physical group of
its own**, named after where it is (`wall_x_600mm` is the face at x = 0.6 m).
That is what lets a run set a different boundary condition on every face
instead of taking one `rigid_walls` whole: the run asks **which** faces are
not hard walls, and only those. All four carry a `plate` and its
`plate_clamp`, so they are vibroacoustic as they stand; drop the structure
from the spec and they are acoustic cavities.

## Where the frequency in the last column comes from

Two rules, and the smaller of the two wins:

| | rule | at h = 25 mm | at h = 40 mm |
|---|---|---|---|
| fluid | 13 elements per acoustic wavelength, $f = c_0/(13h)$, for about 1% frequency error; six is the floor pyCAFE refuses to go below | 1055 Hz | 660 Hz |
| structure | six elements per **bending** wavelength, $\lambda_b = 2\pi\,(D/m\omega^2)^{1/4}$ | 435 Hz | 170 Hz |

The bending wavelength of a thin plate is far shorter than the acoustic one
at the same frequency, so on a coupled mesh **the plate is what binds**: a
1 mm aluminium plate ($D = 6.55$ N m, $m = 2.7$ kg/m²) at 400 Hz has
$\lambda_b = 156$ mm, against 858 mm in air. That is why the two block cases
are cut at 25 mm and quoted at 400 Hz rather than at the 1055 Hz the fluid
alone would allow, and why the coarser 40 mm cases above are quoted at
200 Hz. A thicker plate is stiffer and moves its own limit up; the number in
the table assumes the thinnest plate the mesh is meant to carry.

`frequency_limits(path)` reports the fluid half of this straight off a file,
and `validate_mesh(path, f_max=...)` refuses a model whose fluid falls below
six elements per wavelength.
