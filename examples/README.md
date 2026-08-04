# pyCAFE examples

Three notebooks, in reading order. Each one is self-contained and runs from
this directory.

| Notebook | What it covers |
|---|---|
| `01_cavity_acoustic_3d.ipynb` | Rigid-walled air cavity, 0.414 × 0.314 × 0.360 m, meshed with 16 × 16 × 16 `CHEXA8` acoustic elements. Modal analysis, compared against the analytical modes of a rectangular box. |
| `02_cavity_plate_coupled.ipynb` | The same kind of box, but one face is a flexible CCCC aluminium plate (`CQUAD4F`). Vibroacoustic coupling: the plate radiates into the cavity and the cavity loads the plate back. Mesh: `plate_cavity.msh`. |
| `03_duct_from_step_radiation.ipynb` | Full workflow from CAD: `Geom/Tubo_1m_1m.stp` → gmsh → physical groups → duct with a flush-mounted panel, driven by a piston and truncated with the plane-wave radiation condition (`add_anechoic`) at both ends. Mesh: `tubo_flush_plate.msh`. |

## Other files

- `data/` — MSC.Nastran reference `.pch` files, used by the test suite
  (`tests/test_element_cquad4f.py`) and by the validation notebooks.
- `validation/` — element and solver validations kept out of the way of the
  examples: `CQUAD4F` vs Nastran, pyCAFE vs FEniCSx, h- and p-convergence,
  analytical Helmholtz cases. Supporting material rather than tutorials.
