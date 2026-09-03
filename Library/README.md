# `Library/` — every mesh a run can be pointed at

One folder of gmsh meshes, which is what this release reads.
`ask_for_geometry()` opens its dialog here, filtered on `.msh`, so picking a
model is picking a file and nothing else has to be said about it: a mesh
carries its own element sizes and its own physical group names.

## Every mesh here is cut for 1500 Hz

**Six elements per acoustic wavelength up to 1500 Hz**, which fixes the element
size once for the whole folder:

$$h = \frac{c_0}{n_\lambda\,f_{max}} = \frac{343}{6 \times 1500} = 38.1\ \text{mm}$$

No fluid element in any of these files is longer than 38.1 mm on its longest
side, so `frequency_limits(path)` reports `f_floor` ≥ 1500 Hz for all of them
and none is refused by a spec asking for a band that high. The dialog says so
when it opens, and prints the caveats below to the console with it.

**One file is outside the rule: `cavity_plate_FG.msh`.** Its element counts are
Fahy & Gardonio's own — 16 × 16 × 8, 45 mm elements — because reproducing their
eighteen coupled modes means reproducing their mesh too. It is quoted at 500 Hz
and refused above 1270 Hz, and it is the **modal** case only: the forced run of
the same box uses `cavity_plate_FG_directresponse.msh`, which is cut to the rule
and past it.

Two things the number does **not** promise:

- **Six per wavelength is a floor, not an accuracy target.** The trilinear
  acoustic element disperses as $164/n_\lambda^2$, about 4.6% at six. Thirteen
  per wavelength is the 1% figure, and on a 38.1 mm mesh that lands near 690 Hz
  — the `f_1pct` column below. A study that wants 1% across the whole band asks
  for a lower `f_max`, or a finer mesh.
- **The plate is not covered.** A thin panel has a *bending* wavelength far
  shorter than the acoustic one — 81 mm for 1 mm aluminium at 1500 Hz against
  229 mm in air — so near the top of the band a coupled run is resolving the
  fluid, not the panel. Where a case was already finer than 38.1 mm for the
  plate's sake, it is left finer; see the last column.

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

`h` is the largest fluid element; `f_floor` is where six elements per
wavelength are left, i.e. the highest `f_max` the file accepts; `f_1pct` is
where thirteen are left, i.e. where it is still worth 1%. Both come straight
off the file, from `frequency_limits(path)`.

| file | what it is | `h` | `f_floor` | `f_1pct` |
|---|---|---|---|---|
| `box_cavity.msh` | hard-walled air box 0.40 × 0.30 × 0.50 m, 12 × 9 × 15 `CHEXA8` — the analytical modal case of notebook 01, whose twelfth mode is at 924 Hz. Already finer than the folder rule, so its counts are pinned: notebook 01 prints the error of every mode against the closed form, and those numbers move if the mesh does | 33.3 mm | 1715 Hz | 792 Hz |
| `box_with_plate.msh` | the same box 0.80 × 0.60 × 0.50 m closed by a flexible plate on `z = Lz`, 31 × 23 × 14 — the general coupled box, used by notebook 03; the plate quadrilaterals *are* faces of the fluid hexahedra, so the interface is conforming. `x`/`y` are cut at 26 mm for the **bending** wavelength of the plate, `z` at 35.7 mm for the acoustic one | 35.7 mm | 1601 Hz | 739 Hz |
| `cavity_plate_FG.msh` | **the one file outside the 1500 Hz rule. Fahy & Gardonio's coupled example**: 0.414 × 0.314 × 0.360 m air box closed by a 1 mm CCCC aluminium plate on `z = Lz`, on the book's own mesh — 16 × 16 × 8 `CHEXA8` and 16 × 16 `CQUAD4`. The element counts are pinned, not derived from `f_max`: they are part of the case being reproduced. Modal only; the forced run of the same box is the next row. Notebook `02_cavity_plate_coupled_FG` | 45.0 mm | **1270 Hz** | 586 Hz |
| `cavity_plate_FG_directresponse.msh` | **the same Fahy & Gardonio box, cut for a forced run**: 0.414 × 0.314 × 0.360 m, 32 × 24 × 16 `CHEXA8` and 32 × 24 `CQUAD4F`, conforming. The plate sets the size — a bending wavelength of 99 mm at 1000 Hz against 343 mm of sound — so 13 mm elements in the plane of the plate and 22.5 mm through the depth. Well past the folder rule, and left that way. Notebook `02_cavity_plate_coupled_FG_directresponse` | 22.5 mm | 2541 Hz | 1173 Hz |
| `duct_2d.msh` | 2D duct 1.0 × 0.1 m with `inlet` and `outlet`, 27 × 3 — plane waves, impedance ends | 37.0 mm | 1543 Hz | 712 Hz |
| `plate.msh` | a flat 0.40 × 0.30 m shell on its own, 11 × 8, clamped edge, for structural checks. No fluid, so the acoustic rule has nothing to bind on; the size follows it anyway, to keep one number for the folder | — | — | — |
| `duct_with_flush_plate.msh` | the 3 m duct of `WIP/Tubo_1m_1m.stp` with a 1 m panel let into the floor, cut into structured blocks. The folder rule sizes this one directly: 59049 `CHEXA8`, 64288 nodes, the largest file here | 37.0 mm | 1543 Hz | 712 Hz |
| `box_faces_plate.msh` | the same coupled box, 0.80 × 0.60 × 0.50 m, 21 × 16 × 14, with **every wall its own physical group** (`wall_x_0mm`, `wall_z_0mm`, …): a run can be asked what each face does, one question per face | 38.1 mm | 1501 Hz | 693 Hz |
| `l_cavity_plate.msh` | L-shaped room 1.2 × 0.6 m in plan, one arm twice as tall, plate closing the ceiling of the tall arm; 7 wall groups, 8448 `CHEXA8` | 37.5 mm | 1524 Hz | 704 Hz |
| `step_duct_plate.msh` | duct 2 m long that changes section halfway, with a 0.6 m panel let into the floor of the low part; 8 wall groups, 19125 nodes. Already finer than the rule, so its 25 mm elements stand | 25.0 mm | 2287 Hz | 1055 Hz |
| `t_cavity_plate.msh` | T-shaped cavity, a corridor with a branch, plate on the ceiling of the branch; 9 wall groups, 18785 nodes. Already finer than the rule, so its 25 mm elements stand | 25.0 mm | 2287 Hz | 1055 Hz |

**The frequency a mesh was built for is the frequency it is good for.** Each
case is stated as a `ModelSpec`, so the element size is not written by hand: it
follows from that `f_max` and the elements-per-wavelength rule,
$h = c_0 / (f_{max} n_\lambda)$, and for this folder both numbers are the same
for every case — `f_max = 1500` Hz, `n_lambda = 6`. Handing one of these files
to a spec with a *higher* `f_max` does not refine it — the validator simply
reports how far it still resolves.

## Rebuilding, and adding a case

```bash
python Library/make_library.py
```

`CASES` in that script is the whole content of this folder: a geometry, the
frequency it is built for, and one line on what it is for. Add an entry and run
it again. `ELEMENTS_PER_WAVELENGTH = 6` and `f_max = 1500.0` at the top are the
folder rule; a new case that keeps both is cut like the rest of them.

A case whose builder is not one the spec can size itself — anything built by
`block_cavity`, and `duct_with_flush_plate` — carries its counts by hand, so
check them against 38.1 mm when adding one. `build_mesh` refuses the case
outright if it comes out below six elements per wavelength at its `f_max`,
which is how the rule is enforced rather than trusted.

## Using one

```python
from pycafe.core.model_spec import AIR, MeshFile, ModelSpec, aluminium, build_model

spec = ModelSpec(
    geometry=MeshFile("Library/box_with_plate.msh"),
    fluid=AIR,
    structure=aluminium(t=2e-3, support="clamped"),
    f_max=1500.0,          # every mesh here accepts this; see the table
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

## Why the acoustic rule, and not the plate's

Two rules could size a coupled mesh, and they give very different answers:

| | rule | at h = 25 mm | at h = 38.1 mm |
|---|---|---|---|
| fluid | 13 elements per acoustic wavelength, $f = c_0/(13h)$, for about 1% frequency error; **six is the floor** pyCAFE refuses to go below, $f = c_0/(6h)$ | 1055 Hz / 2287 Hz | 693 Hz / **1500 Hz** |
| structure | six elements per **bending** wavelength, $\lambda_b = 2\pi\,(D/m\omega^2)^{1/4}$, for 1 mm aluminium | 435 Hz | 187 Hz |

**This folder is cut to the fluid row.** 1500 Hz here means 1500 Hz of sound in
air, at the floor of six elements per wavelength. The plate row is the reason
that number is not a licence to run a coupled model at 1500 Hz and believe the
panel: at 1500 Hz a 1 mm aluminium plate has $\lambda_b = 81$ mm, so a 38.1 mm
mesh puts two elements across it. Where that mattered — `box_with_plate`,
`cavity_plate_FG_directresponse` — the plane of the plate is cut finer than the
rule asks and left that way.

The bending wavelength of a thin plate is far shorter than the acoustic one
at the same frequency, so on a coupled mesh **the plate is what binds**: a
1 mm aluminium plate ($D = 6.55$ N m, $m = 2.7$ kg/m²) at 400 Hz has
$\lambda_b = 156$ mm, against 858 mm in air. A thicker plate is stiffer and
moves its own limit up; the numbers in the table assume the thinnest plate a
mesh is meant to carry, 1 mm.

So read the `f_floor` column for what it is: **the highest `f_max` the file
accepts**, which is a statement about the fluid mesh. A coupled study that
means to trust the panel that high needs the plate row too, and should either
stay well below `f_floor` or rebuild the case finer — `CASES` in
`make_library.py` is where the counts are.

`frequency_limits(path)` reports the fluid half of this straight off a file,
and `validate_mesh(path, f_max=...)` refuses a model whose fluid falls below
six elements per wavelength.
