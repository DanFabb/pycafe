# pyCAFE examples

Four notebooks, in reading order. Each one is self-contained and runs from
this directory.

All of them state their model the same way, as a `ModelSpec`
(`pycafe.core.model_spec`): where the geometry comes from — parameters we ask
for (`Library`), a CAD file the user brings (`CadFile`, after `inspect_cad`
says which entity is fluid and which is structure), or an existing mesh
(`MeshFile`) — the two materials, and the highest frequency of interest. The
mesh size is derived from that frequency, $h = c_0 / (f_{max}\,n_\lambda)$,
with at least 10 elements per wavelength; `build_model` meshes, validates and
assembles in one call. Notebooks 01–03 pin the element counts explicitly
because their numbers are quoted against a known mesh — that override is what
a convergence study uses.

| Notebook | What it covers |
|---|---|
| `01_cavity_acoustic_3d.ipynb` | Rigid-walled air cavity, 0.414 × 0.314 × 0.360 m, meshed with 16 × 16 × 16 `CHEXA8` acoustic elements. Modal analysis, compared against the analytical modes of a rectangular box. |
| `02_cavity_plate_coupled.ipynb` | The same kind of box, but one face is a flexible CCCC aluminium plate (`CQUAD4F`). Vibroacoustic coupling: the plate radiates into the cavity and the cavity loads the plate back. Mesh: `plate_cavity.msh`. |
| `03_duct_from_step_radiation.ipynb` | Full workflow from CAD: `Geom/Tubo_1m_1m.stp` → gmsh → physical groups → duct with a flush-mounted panel, driven by a piston and truncated with the plane-wave radiation condition (`add_anechoic`) at both ends. Mesh: `tubo_flush_plate.msh`. |
| `04_duct_pml_vs_sommerfeld.ipynb` | The two ways of ending a 3D tube, side by side: a perfectly matched layer (a physical group named `pml`) against the plane-wave radiation condition $\zeta = 1$. Below the first cross mode they agree; above it, the wave reaches the end at an angle and only the layer still absorbs it. Meshes are built in the notebook. |
| `05_shell_in_cavity_nonconforming.ipynb` | A plane wave at 60 kHz scattered by a cylinder in water (`Geom/Vibro_ac1.step`, the COMSOL *Acoustic-Structure Interaction* geometry): spherical wave radiation with an incident field on the outer sphere, the cylinder sound-hard and then as a 1 mm aluminium tube. Fluid and shell are meshed separately (`independent_structure=True`), merged without sharing a node, and coupled by the interpolation method of Mi & Zheng (2018); a shell made stiff enough to be a wall reproduces the sound-hard field to 0.03%. Mesh: `scatter_coupled.msh` (built by the notebook). |

## Other files

- `data/` — MSC.Nastran reference `.pch` files, used by the test suite
  (`tests/test_element_cquad4f.py`) and by the validation notebooks.
- `validation/` — element and solver validations kept out of the way of the
  examples: `CQUAD4F` vs Nastran, pyCAFE vs FEniCSx, h- and p-convergence,
  analytical Helmholtz cases. Supporting material rather than tutorials.
