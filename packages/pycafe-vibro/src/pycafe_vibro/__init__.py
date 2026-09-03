"""
pycafe_vibro — the structural and coupled half of pyCAFE.

pyCAFE proper is acoustic: fluid elements, acoustic boundary
conditions, the Helmholtz solvers and the fields they produce.
Everything that needs a *shell* is here — the CQUAD4F element, the
plate material, the fluid-structure coupling matrix and the coupled
solvers — and installing this package installs pyCAFE with it.

Importing it registers the structural element in pyCAFE's element
registry, so ``pycafe.build_matrices`` dispatches on CQUAD4F from that
moment on. The names of both packages are reachable from either side:

    >>> import pycafe_vibro as pcv          # doctest: +SKIP
    >>> spec = pcv.ModelSpec(...)           # acoustic name, re-exported
    >>> pcv.aluminium(t=2e-3)               # structural name

and, for scripts written against pyCAFE alone, ``pycafe.aluminium``
keeps working as soon as this package is installed.
"""

from pycafe.build_matrices.element_registry import ElementSpec, register_element

from .element_cquad4f import element_matrices_cquad4f

__version__ = "1.1.0"


# Structural shell, Nastran/MacNeal style: selective reduced
# integration plus residual bending flexibility for transverse shear.
# 6 DOF/node; the kernel needs the properties (t, rho, E, nu), to be
# passed through `kernel_args` in `assemble_KM` together with
# `dofs_per_node=6`.
register_element(
    "CQUAD4F",
    ElementSpec(
        gmsh_names=("Quadrilateral 4", "Quadrangle 4"),
        n_nodes=4,
        dim=2,
        field="structural",
        dofs_per_node=6,
        kernel=element_matrices_cquad4f,
    ),
    replace=True,
)


# Names re-exported from their module on first use, so that importing
# pycafe_vibro does not pull in scipy and the rest.
_LAZY = {
    # what a shell is made of
    "Structure": "pycafe_vibro.structure",
    "aluminium": "pycafe_vibro.structure",
    "steel": "pycafe_vibro.structure",
    # the shell itself
    "element_matrices_cquad4f": "pycafe_vibro.element_cquad4f",
    "build_KM_structural_domain": "pycafe_vibro.domains",
    # fluid meets structure
    "build_coupling_matrix": "pycafe_vibro.coupling",
    "acoustic_coupling_matrix": "pycafe_vibro.coupling",
    "interface_area_vector": "pycafe_vibro.coupling",
    "interface_normals": "pycafe_vibro.coupling",
    "interface_is_conforming": "pycafe_vibro.coupling",
    "build_nonconforming_coupling": "pycafe_vibro.coupling_nonconforming",
    "prepare_vibroacoustic_system": (
        "pycafe_vibro.prepare_vibroacoustic_system"),
    # coupled solvers
    "build_coupled_blocks": "pycafe_vibro.solver_vibroacoustic",
    "coupled_blocks_from_bc": "pycafe_vibro.solver_vibroacoustic",
    "solve_vibroacoustic_modal": "pycafe_vibro.solver_vibroacoustic",
    "solve_vibroacoustic_frequency_sweep": "pycafe_vibro.solver_vibroacoustic",
    "structural_point_load": "pycafe_vibro.solver_vibroacoustic",
    "expand_pressure": "pycafe_vibro.solver_vibroacoustic",
    "structural_displacement_field": "pycafe_vibro.solver_vibroacoustic",
    "build_cms_basis": "pycafe_vibro.solver_vibroacoustic_modal",
    "build_coupled_modal_basis": "pycafe_vibro.solver_vibroacoustic_modal",
    "project_coupled_system": "pycafe_vibro.solver_vibroacoustic_modal",
    "solve_coupled_modal_frequency_sweep": (
        "pycafe_vibro.solver_vibroacoustic_modal"),
    "reduced_model_error": "pycafe_vibro.solver_vibroacoustic_modal",
    # the closed forms a plate is checked against
    "clamped_plate_modes": "pycafe_vibro.analytical",
    "warburton_cccc": "pycafe_vibro.analytical",
    # reading the answer on the panel
    "Panel": "pycafe_vibro.panel",
    "describe_panel": "pycafe_vibro.panel",
    "panel_displacement": "pycafe_vibro.panel",
    "plot_panel_mode": "pycafe_vibro.panel",
}


def __getattr__(name):
    """Names of this package first, then pyCAFE's own."""
    import importlib

    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name]), name)

    import pycafe

    if name in pycafe.__all__:
        return getattr(pycafe, name)
    raise AttributeError(f"module 'pycafe_vibro' has no attribute '{name}'")


def __dir__():
    import pycafe

    return sorted(set(_LAZY) | set(pycafe.__all__))


# One import is the whole API, acoustic half included: ``from
# pycafe_vibro import *`` brings in the names above and everything
# ``from pycafe import *`` would.
def _all():
    import pycafe

    return sorted(set(_LAZY) | set(pycafe.__all__))


__all__ = _all()
