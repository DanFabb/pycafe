__version__ = "1.1.0"

# Names re-exported from their module on first use, so that importing
# pycafe does not pull in gmsh, scipy and the rest.
_LAZY = {
    # what a model is made of
    "Library": "pycafe.core.model_spec",
    "CadFile": "pycafe.core.model_spec",
    "NastranFile": "pycafe.core.model_spec",
    "MeshFile": "pycafe.core.model_spec",
    "Fluid": "pycafe.core.model_spec",
    "AIR": "pycafe.core.model_spec",
    "WATER": "pycafe.core.model_spec",
    "describe_domains": "pycafe.core.model_spec",
    # asking the user what a script cannot know
    "ask_for_acoustic_bc": "pycafe.create_geom.ask",
    "ACOUSTIC_BOUNDARY_CONDITIONS": "pycafe.create_geom.ask",
    "ACOUSTIC_POINT_SOURCES": "pycafe.create_geom.ask",
    "ask_for_point_source": "pycafe.create_geom.ask",
    "ask_for_choice": "pycafe.create_geom.ask",
    "ask_for_many": "pycafe.create_geom.ask",
    "ask_for_number": "pycafe.create_geom.ask",
    "ask_for_point": "pycafe.create_geom.ask",
    "ask_for_points": "pycafe.create_geom.ask",
    "ask_for_structure": "pycafe.create_geom.ask",
    # looking at the mesh
    "plot_mesh_3d": "pycafe.create_geom.preview",
    "open_in_gmsh": "pycafe.create_geom.visualize_mesh",
    "describe_conventions": "pycafe.create_geom.conventions",
    "GmshModel": "pycafe.create_geom.gmsh_workflow",
    "describe_mesh": "pycafe.create_geom.validation",
    "frequency_limits": "pycafe.create_geom.validation",
    # boundary conditions
    "build_impedance_operator": "pycafe.boundary_condition.acoustic_bc",
    "build_radiation_operator": "pycafe.boundary_condition.acoustic_bc",
    "build_velocity_operator": "pycafe.boundary_condition.acoustic_bc",
    "build_source_operator": "pycafe.boundary_condition.acoustic_bc",
    "build_acoustic_load": "pycafe.boundary_condition.acoustic_bc",
    "prescribed_pressure": "pycafe.boundary_condition.acoustic_bc",
    # solvers
    "solve_modal_acoustic_reduced": "pycafe.solver.solver_modale",
    "acoustic_modal_system": "pycafe.solver.solver_modale",
    "solve_acoustic_frequency_sweep": "pycafe.solver.solver_helmholtz_1",
    "build_modal_basis": "pycafe.solver.solver_modal_forced",
    "modal_damping_matrix": "pycafe.solver.solver_modal_forced",
    "solve_modal_frequency_sweep": "pycafe.solver.solver_modal_forced",
    # reading the answer
    "find_response_peaks": "pycafe.post_processing.frf",
    "missing_from_response": "pycafe.post_processing.frf",
    "plot_mac_matrix": "pycafe.post_processing.mac",
    "plot_frf": "pycafe.post_processing.plots",
    "plot_modes": "pycafe.post_processing.plots",
    "plot_mode_grid": "pycafe.post_processing.plots",
    "plot_field": "pycafe.post_processing.plots",
    "plot_matrix": "pycafe.post_processing.plots",
    "plot_shares": "pycafe.post_processing.plots",
    "AMPLITUDE_CMAP": "pycafe.post_processing.plots",
    "SIGNED_CMAP": "pycafe.post_processing.plots",
    # the closed forms a model is checked against
    "rectangular_cavity_modes": "pycafe.analytical",
    "cavity_mode_shape": "pycafe.analytical",
    "identify_cavity_modes": "pycafe.analytical",
    "mac": "pycafe.analytical",
}


# Names that live in pycafe_vibro, the structural and coupled half
# of pyCAFE. They stay reachable as ``pycafe.<name>`` so that a
# script does not have to know which of the two packages defines
# what; asking for one without that package installed says so.
_VIBRO = {
    "Structure": "pycafe_vibro.structure",
    "aluminium": "pycafe_vibro.structure",
    "steel": "pycafe_vibro.structure",
    "build_coupling_matrix": "pycafe_vibro.coupling",
    "build_KM_structural_domain": "pycafe_vibro.domains",
    "prepare_vibroacoustic_system": "pycafe_vibro.prepare_vibroacoustic_system",
    "solve_vibroacoustic_modal": "pycafe_vibro.solver_vibroacoustic",
    "solve_vibroacoustic_frequency_sweep": "pycafe_vibro.solver_vibroacoustic",
    "build_coupled_blocks": "pycafe_vibro.solver_vibroacoustic",
    "coupled_blocks_from_bc": "pycafe_vibro.solver_vibroacoustic",
    "structural_point_load": "pycafe_vibro.solver_vibroacoustic",
    "expand_pressure": "pycafe_vibro.solver_vibroacoustic",
    "structural_displacement_field": "pycafe_vibro.solver_vibroacoustic",
    "build_cms_basis": "pycafe_vibro.solver_vibroacoustic_modal",
    "build_coupled_modal_basis": "pycafe_vibro.solver_vibroacoustic_modal",
    "project_coupled_system": "pycafe_vibro.solver_vibroacoustic_modal",
    "solve_coupled_modal_frequency_sweep": "pycafe_vibro.solver_vibroacoustic_modal",
    "reduced_model_error": "pycafe_vibro.solver_vibroacoustic_modal",
    "clamped_plate_modes": "pycafe_vibro.analytical",
    "warburton_cccc": "pycafe_vibro.analytical",
    "Panel": "pycafe_vibro.panel",
    "describe_panel": "pycafe_vibro.panel",
    "panel_displacement": "pycafe_vibro.panel",
    "plot_panel_mode": "pycafe_vibro.panel",
}


def __getattr__(name):
    import importlib

    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name]), name)
    if name in _VIBRO:
        try:
            module = importlib.import_module(_VIBRO[name])
        except ImportError:
            raise ImportError(
                f"pycafe.{name} is part of pycafe_vibro, the structural "
                "and coupled half of pyCAFE: pip install pycafe-vibro."
            ) from None
        return getattr(module, name)
    raise AttributeError(f"module 'pycafe' has no attribute '{name}'")


def load_mesh(*args, **kwargs):
    from pycafe.core.load_mesh import load_mesh as _load_mesh

    return _load_mesh(*args, **kwargs)


def load_fluid(*args, **kwargs):
    from pycafe.core.load_fluid import load_fluid as _load_fluid

    return _load_fluid(*args, **kwargs)


def create_geometry(*args, **kwargs):
    from pycafe.core.create_geom import create_geometry as _create_geometry

    return _create_geometry(*args, **kwargs)


def AcousticBC(*args, **kwargs):
    """Boundary condition container; see
    :class:`pycafe.boundary_condition.acoustic_bc.AcousticBC`."""
    from pycafe.boundary_condition.acoustic_bc import AcousticBC as _AcousticBC

    return _AcousticBC(*args, **kwargs)


def IncidentPlaneWave(*args, **kwargs):
    """Plane wave crossing a radiation boundary; see
    :class:`pycafe.boundary_condition.acoustic_bc.IncidentPlaneWave`."""
    from pycafe.boundary_condition.acoustic_bc import (
        IncidentPlaneWave as _IncidentPlaneWave,
    )

    return _IncidentPlaneWave(*args, **kwargs)


def assign_boundary_conditions(*args, **kwargs):
    from pycafe.core.assign_boundary_conditions import (
        assign_boundary_conditions as _assign_boundary_conditions,
    )

    return _assign_boundary_conditions(*args, **kwargs)


def prepare_acoustic_system(*args, **kwargs):
    from pycafe.core.prepare_acoustic_system import (
        prepare_acoustic_system as _prepare_acoustic_system,
    )

    return _prepare_acoustic_system(*args, **kwargs)


def run_analysis(*args, **kwargs):
    from pycafe.core.run_analysis import run_analysis as _run_analysis

    return _run_analysis(*args, **kwargs)


def ModelSpec(*args, **kwargs):
    """Declarative model description; see
    :class:`pycafe.core.model_spec.ModelSpec`."""
    from pycafe.core.model_spec import ModelSpec as _ModelSpec

    return _ModelSpec(*args, **kwargs)


def build_model(*args, **kwargs):
    """Mesh, validate and assemble a :class:`ModelSpec`; see
    :func:`pycafe.core.model_spec.build_model`."""
    from pycafe.core.model_spec import build_model as _build_model

    return _build_model(*args, **kwargs)


def build_mesh(*args, **kwargs):
    """Mesh of a :class:`ModelSpec`; see
    :func:`pycafe.core.model_spec.build_mesh`."""
    from pycafe.core.model_spec import build_mesh as _build_mesh

    return _build_mesh(*args, **kwargs)


def plot_geometry(*args, **kwargs):
    """Mesh coloured by role, before any matrix; see
    :func:`pycafe.create_geom.preview.plot_geometry`."""
    from pycafe.create_geom.preview import plot_geometry as _plot_geometry

    return _plot_geometry(*args, **kwargs)


def preview(*args, **kwargs):
    """Build a spec's mesh and look at it; see
    :func:`pycafe.core.model_spec.preview`."""
    from pycafe.core.model_spec import preview as _preview

    return _preview(*args, **kwargs)


def ask_for_file(*args, **kwargs):
    """Where is the file? — the ``uigetfile`` of pyCAFE; see
    :func:`pycafe.create_geom.ask.ask_for_file`."""
    from pycafe.create_geom.ask import ask_for_file as _ask_for_file

    return _ask_for_file(*args, **kwargs)


def ask_for_geometry(*args, **kwargs):
    """Ask where the geometry is and read whichever kind it is; see
    :func:`pycafe.create_geom.ask.ask_for_geometry`."""
    from pycafe.create_geom.ask import ask_for_geometry as _ask_for_geometry

    return _ask_for_geometry(*args, **kwargs)


def inspect_cad(*args, **kwargs):
    """What is inside a CAD file, tag by tag; see
    :func:`pycafe.create_geom.external.inspect_cad`."""
    from pycafe.create_geom.external import inspect_cad as _inspect_cad

    return _inspect_cad(*args, **kwargs)


def inspect_bdf(*args, **kwargs):
    """What is inside a Nastran deck, property id by property id; see
    :func:`pycafe.create_geom.nastran.inspect_bdf`."""
    from pycafe.create_geom.nastran import inspect_bdf as _inspect_bdf

    return _inspect_bdf(*args, **kwargs)


def load_mesh_with_groups(*args, **kwargs):
    from pycafe.create_geom.visualize_mesh import (
        load_mesh_with_groups as _load_mesh_with_groups,
    )

    return _load_mesh_with_groups(*args, **kwargs)


def plot_pressure_3d(*args, **kwargs):
    """3D pressure field plot; see
    :func:`pycafe.post_processing.post_processing_3d.plot_pressure_3d`."""
    from pycafe.post_processing.post_processing_3d import (
        plot_pressure_3d as _plot_pressure_3d,
    )

    return _plot_pressure_3d(*args, **kwargs)


def animate_pressure_3d(*args, **kwargs):
    """3D pressure field movie; see
    :func:`pycafe.post_processing.post_processing_3d.animate_pressure_3d`."""
    from pycafe.post_processing.post_processing_3d import (
        animate_pressure_3d as _animate_pressure_3d,
    )

    return _animate_pressure_3d(*args, **kwargs)


def run_post_processing_3d(*args, **kwargs):
    """Interactive 3D post-processing menu; see
    :func:`pycafe.post_processing.post_processing_3d.run_post_processing_3d`."""
    from pycafe.post_processing.post_processing_3d import (
        run_post_processing_3d as _run_post_processing_3d,
    )

    return _run_post_processing_3d(*args, **kwargs)


def clip_elements(*args, **kwargs):
    """Cutting plane exposing the interior; see
    :func:`pycafe.post_processing.post_processing_3d.clip_elements`."""
    from pycafe.post_processing.post_processing_3d import (
        clip_elements as _clip_elements,
    )

    return _clip_elements(*args, **kwargs)


def pressure_at_point_3d(*args, **kwargs):
    """Frequency response at a 3D point; see
    :func:`pycafe.post_processing.post_processing_3d.pressure_at_point_3d`."""
    from pycafe.post_processing.post_processing_3d import (
        pressure_at_point_3d as _pressure_at_point_3d,
    )

    return _pressure_at_point_3d(*args, **kwargs)


def _vibro_is_installed():
    """Is the structural half there? Asked without importing it."""
    import importlib.util

    try:
        return importlib.util.find_spec("pycafe_vibro") is not None
    except (ImportError, ValueError):        # pragma: no cover
        return False


# One import is the whole API: ``from pycafe import *`` brings in the
# names above and the wrappers defined here, each module still loaded
# only when the name it holds is first used. The structural names join
# them only when pycafe_vibro is installed, so that the star import of
# an acoustic-only install pulls in what it actually has; asking for
# one of them by name says where it lives either way.
__all__ = sorted(
    set(_LAZY)
    | (set(_VIBRO) if _vibro_is_installed() else set())
    | {name for name, value in list(globals().items())
       if callable(value) and not name.startswith("_")}
)
