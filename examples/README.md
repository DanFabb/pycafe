# pyCAFE examples

Five notebooks, in reading order. Each one is self-contained and runs from
this directory.

All of them open the same way and state their model the same way, as a
`ModelSpec` (`pycafe.core.model_spec`): where the geometry comes from, the two
materials, and the highest frequency of interest. The mesh size is derived from
that frequency, $h = c_0 / (f_{max}\,n_\lambda)$; `build_model` meshes,
validates, draws and assembles in one call. Both notebook 02s pin the element
counts explicitly instead, because their numbers are quoted against a published
mesh — that same override is what a convergence study uses, and what a layer
that has to be an exact number of elements deep needs.

The opening block states two things before any file is opened: **which run** —
`ANALYSIS = "acoustic"` (the fluid alone) or `"vibroacoustic"` (fluid and
structure, coupled) — and **what the domains are made of**, one material each
(`FLUID`, `STRUCTURE`). The analysis goes into the spec, so a geometry that
cannot serve it is refused at validation instead of being quietly downgraded,
and `describe_domains(report, fluid, structure)` prints which physical group
turned out to be which domain and which material fills it.

**This release reads one kind of geometry: a mesh gmsh has already cut.**
`ask_for_geometry()` opens a dialog on `../Library/`, filtered on `.msh`, and
reads whatever is picked; a `.msh` carries its own element sizes and its own
physical group names, so loading it is the last question about the geometry.
The dialog says the 1500 Hz rule and its exception in its own prompt, and
prints the caveats below to the console as it opens, so the resolution of the
file being picked is on screen at the moment of picking it.
**There is no fallback file**: a cancelled dialog or an empty answer raises,
rather than quietly loading some other geometry. For a run with nobody at the
keyboard (CI, or `nbconvert` executing a notebook) the answer comes from the
environment: `PYCAFE_FILE_GEOMETRY`. Notebooks 01 and 02 are the exceptions:
each names its file instead of asking, because its numbers are quoted against
that one mesh and no other.

## Every mesh these notebooks read is cut for 1500 Hz

The `.msh` files in `../Library/` are all cut to one rule: **six elements per
acoustic wavelength up to 1500 Hz**, i.e. no fluid element longer than

$$h = \frac{c_0}{n_\lambda\,f_{max}} = \frac{343}{6 \times 1500} = 38.1\ \text{mm}$$

so whichever file you pick, a run can ask for an `f_max` anywhere up to 1500 Hz
and pyCAFE will not refuse it. `frequency_limits(path)` reports `f_floor` ≥
1500 Hz for every one of them, and the notebooks print that line before they
ask for a frequency.

**One file is outside the rule: `cavity_plate_FG.msh`**, used by the modal
notebook `02_cavity_plate_coupled_FG`. Its 16 × 16 × 8 element counts are Fahy
& Gardonio's own, because reproducing their eighteen coupled modes means
reproducing their mesh; its elements are 45 mm and it stops at 1270 Hz. The
forced version of the same case,
`02_cavity_plate_coupled_FG_directresponse`, reads
`cavity_plate_FG_directresponse.msh`, which is cut past the rule at 22.5 mm.

Two things 1500 Hz does **not** mean:

- **Six per wavelength is a floor, not an accuracy target.** The trilinear
  acoustic element disperses as $164/n_\lambda^2$, about 4.6% at six per
  wavelength. Thirteen per wavelength is the 1% figure, and on a 38.1 mm mesh
  that is only ~690 Hz — `f_1pct` in the notebooks' printout. Ask for a lower
  `f_max` when the whole band has to be worth 1%.
- **It is a statement about the fluid, not the panel.** A 1 mm aluminium plate
  has a bending wavelength of 81 mm at 1500 Hz against 229 mm of sound in air,
  so near the top of the band a coupled run is resolving the cavity and not the
  plate. The cases where that mattered are cut finer in the plane of the plate;
  `Library/README.md` has the per-file numbers.


Meshing a CAD file (`CadFile`) and reading a Nastran deck (`NastranFile`) are
written and work when called directly, but they are not in this release: those
notebooks are in `WIP/` and the files they read in `../Library/WIP/`.

| Notebook | What it covers |
|---|---|
| `00_master_geometry.ipynb` | **Start here.** No physics: which run you want, what its domains are made of, and how a mesh gets in. The dialog, the `Library/` folder, the naming contract, what each physical group has to be called. |
| `01_cavity_acoustic_3d.ipynb` | Rigid-walled air cavity meshed in `CHEXA8`: the first twelve modes against the analytical modes of a rectangular box. No dialog — `Library/box_cavity.msh` is named in the code, because the closed form describes that box and no other geometry. |
| `02_cavity_plate_coupled_FG.ipynb` | **Fahy & Gardonio's worked coupled example**, geometry and mesh both theirs: a 0.414 × 0.314 × 0.360 m air box with one face a 1 mm CCCC aluminium plate (`CQUAD4F`), 16 × 16 × 8 fluid elements and 16 × 16 plate elements. The nine uncoupled plate modes and the sealed cavity first, then the eighteen coupled ones, matched to the uncoupled lists by shape. No dialog either: `Library/cavity_plate_FG.msh` is named in the code. |
| `02_cavity_plate_coupled_FG_directresponse.ipynb` | **The same box, driven.** Fahy & Gardonio's geometry, air and plate on a mesh cut past 1500 Hz (32 × 24 × 16, 22.5 mm), with 1 N normal to the centre of the plate and $\eta_s = 0.02$: the coupled system solved directly at every line of a 20–1000 Hz sweep, no basis in between. Displacement at the drive, three microphones, the volume-averaged pressure, every peak paired with a coupled mode — and the modes a centred force cannot reach. `Library/cavity_plate_FG_directresponse.msh` is named in the code. |
| `03_any_geometry_any_analysis.ipynb` | **The driver the others are special cases of.** Nothing is fixed and everything is asked: which mesh, acoustic or vibroacoustic, which fluid, what the structure is made of, which faces are not hard walls and what they do instead, whether to compute a modal analysis, a direct frequency response or a modal frequency response, and where the microphones are. The modal sweep projects the model on its own modes -- acoustic or coupled -- and solves an m x m system per line instead of factorizing the whole one; it says so and solves directly when it meets what it cannot project, a pressure held at a value, a PML or a radiating boundary. Any mesh in `../Library/`; `PYCAFE_ANSWER_*` answers the lot with nobody at the keyboard. |

Work in progress, not part of this release: `WIP/` — the CAD workflow and the
non-conforming scattering study. See `WIP/README.md`.

## Other files

- `../Library/` — every mesh a run can be pointed at, in one folder, all cut
  for 1500 Hz at six elements per wavelength, rebuilt by
  `python Library/make_library.py`. The CAD files and the Nastran deck are
  in `../Library/WIP/`, out of the dialog. See `Library/README.md`.
- `data/` — MSC.Nastran reference `.pch` files, used by the test suite
  (`tests/test_element_cquad4f.py`) and by the validation notebooks.
- `validation/` — element and solver validations kept out of the way of the
  examples: `CQUAD4F` vs Nastran, pyCAFE vs FEniCSx, h- and p-convergence,
  analytical Helmholtz cases. Supporting material rather than tutorials.
