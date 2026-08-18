# pyCAFE examples

Six notebooks, in reading order. Each one is self-contained and runs from
this directory.

All of them open the same way and state their model the same way, as a
`ModelSpec` (`pycafe.core.model_spec`): where the geometry comes from, the two
materials, and the highest frequency of interest. The mesh size is derived from
that frequency, $h = c_0 / (f_{max}\,n_\lambda)$, with at least 10 elements per
wavelength; `build_model` meshes, validates, draws and assembles in one call.
Notebooks 01–03 pin the element counts explicitly because their numbers are
quoted against a known mesh — that override is what a convergence study uses.

The opening block states two things before any file is opened: **which run** —
`ANALYSIS = "acoustic"` (the fluid alone) or `"vibroacoustic"` (fluid and
structure, coupled) — and **what the domains are made of**, one material each
(`FLUID`, `STRUCTURE`). The analysis goes into the spec, so a geometry that
cannot serve it is refused at validation instead of being quietly downgraded,
and `describe_domains(report, fluid, structure)` prints which physical group
turned out to be which domain and which material fills it.

The geometry is named the same way everywhere, and there are four ways in:

- **`Library`** — pyCAFE builds it from its parameters (a box, a duct, a
  plate). Every such case is also kept pre-built in `../Library/`.
- **`CadFile`** — a STEP/IGES/BREP file the user brings, meshed here. The path
  is never hard-coded: `ask_for_file("cad")` opens a file dialog (`uigetfile`),
  and `inspect_cad` says which entity tag is fluid and which is structure.
- **`NastranFile`** — a `.bdf`/`.nas` deck, already meshed; `inspect_bdf` says
  which property id is which role.
- **`MeshFile`** — a mesh that already carries its physical groups.

`ask_for_geometry()` does all three file cases at once: one dialog on
`../Library/` — which holds the meshes, the CAD files and the decks together —
and the extension of the answer decides how it is read. **There is no fallback
file**: a cancelled dialog or an empty answer raises, rather than quietly
loading some other geometry. For a run with nobody at the keyboard (CI, or
`nbconvert` executing a notebook) the answer comes from the environment:
`PYCAFE_FILE_GEOMETRY`, `PYCAFE_FILE_CAD`, `PYCAFE_FILE_NASTRAN`, or
`PYCAFE_FILE` for any kind.

| Notebook | What it covers |
|---|---|
| `00_master_geometry.ipynb` | **Start here.** No physics: which run you want and what its domains are made of, then load the mesh or the geometry and let pyCAFE work out which it is. The dialog, the four sources, what each one guesses and what it must be told, the `Library/` folder, the naming contract. |
| `01_cavity_acoustic_3d.ipynb` | Rigid-walled air cavity meshed in `CHEXA8`: the first twelve modes against the analytical modes of a rectangular box. The dialog is the same as everywhere else, but this one accepts **`Library/box_cavity.msh` and nothing else** — the closed form describes that box and no other geometry, so anything else picked stops the run. Sides and element counts are measured on the mesh, never written in the notebook. |
| `02_cavity_plate_coupled.ipynb` | The same kind of box, but one face is a flexible CCCC aluminium plate (`CQUAD4F`). Vibroacoustic coupling: the plate radiates into the cavity and the cavity loads the plate back. Mesh: `plate_cavity.msh`. |
| `03_duct_from_step_radiation.ipynb` | Full workflow from CAD: `Library/Tubo_1m_1m.stp` → gmsh → physical groups → duct with a flush-mounted panel, driven by a piston and truncated with the plane-wave radiation condition (`add_anechoic`) at both ends. Mesh: `tubo_flush_plate.msh`. |
| `04_duct_pml_vs_sommerfeld.ipynb` | The two ways of ending a 3D tube, side by side: a perfectly matched layer (a physical group named `pml`) against the plane-wave radiation condition $\zeta = 1$. Below the first cross mode they agree; above it, the wave reaches the end at an angle and only the layer still absorbs it. Meshes are built in the notebook. |
| `05_shell_in_cavity_nonconforming.ipynb` | A plane wave at 60 kHz scattered by a cylinder in water (`Library/Vibro_ac1.step`, the COMSOL *Acoustic-Structure Interaction* geometry): spherical wave radiation with an incident field on the outer sphere, the cylinder sound-hard and then as a 1 mm aluminium tube. Fluid and shell are meshed separately (`independent_structure=True`), merged without sharing a node, and coupled by the interpolation method of Mi & Zheng (2018); a shell made stiff enough to be a wall reproduces the sound-hard field to 0.03%. Mesh: `scatter_coupled.msh` (built by the notebook). |

## Other files

- `../Library/` — everything a run can be pointed at, in one folder: one mesh
  per geometry pyCAFE builds itself (rebuilt by `python Library/make_library.py`),
  plus the CAD files (`Tubo_1m_1m.stp`, `Vibro_ac1.step`) and the Nastran deck
  (`demo_cavity.bdf`) it is given. See `Library/README.md`.
- `data/` — MSC.Nastran reference `.pch` files, used by the test suite
  (`tests/test_element_cquad4f.py`) and by the validation notebooks.
- `validation/` — element and solver validations kept out of the way of the
  examples: `CQUAD4F` vs Nastran, pyCAFE vs FEniCSx, h- and p-convergence,
  analytical Helmholtz cases. Supporting material rather than tutorials.
