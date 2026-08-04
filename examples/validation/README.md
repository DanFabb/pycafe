# Validation

Element and solver validations. Not tutorials — these back up the numbers
quoted in the documentation and in the response to the reviewers.

| File | Against what |
|---|---|
| `cquad4_plate_validation.ipynb` | `CQUAD4F` element stiffness, term by term, vs MSC.Nastran (`../data/*.pch`) |
| `confronto_K_nastran.ipynb` | Same comparison, focused on the assembled `K` |
| `benchmark_cquad4f.py` | Single-element response + plate convergence vs Kirchhoff |
| `analytical_validation.ipynb` | `CQUAD8` cavity modes vs the analytical rectangle |
| `comparison_20modes.ipynb` | First 20 modes, pyCAFE vs reference |
| `comparison_pycafe_fenics.ipynb` | Modal analysis, pyCAFE vs FEniCSx |
| `helmholtz_sweep_pycafe_fenics.ipynb` | Forced frequency sweep, pyCAFE vs FEniCSx |
| `test_direct_helmholtz_cases.py` | Five direct Helmholtz cases vs 1D analytical solutions |
| `test_direct_helmholtz_pycafe_fenics.py` | Direct Helmholtz sweep vs FEniCSx |
| `convergence_hp.ipynb`, `pycafe_convergence_hp.ipynb` | h- and p-convergence rates |

The notebooks that import `pycafe` directly need the package installed
(`pip install -e .` from the repository root). The scripts add the repository
root to `sys.path` themselves, so they run either from here or from the root.

The FEniCSx comparisons additionally need `dolfinx`, `mpi4py`, `ufl` and
`petsc4py`, which are not pyCAFE dependencies.
