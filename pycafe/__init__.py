__version__ = "1.0.0"

# Names re-exported from their module on first use, so that importing
# pycafe does not pull in gmsh, scipy and the rest.
_LAZY = {
    "Library": "pycafe.core.model_spec",
    "CadFile": "pycafe.core.model_spec",
    "MeshFile": "pycafe.core.model_spec",
    "Fluid": "pycafe.core.model_spec",
    "Structure": "pycafe.core.model_spec",
    "AIR": "pycafe.core.model_spec",
    "WATER": "pycafe.core.model_spec",
    "aluminium": "pycafe.core.model_spec",
    "steel": "pycafe.core.model_spec",
}


def __getattr__(name):
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name]), name)
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


def inspect_cad(*args, **kwargs):
    """What is inside a CAD file, tag by tag; see
    :func:`pycafe.create_geom.external.inspect_cad`."""
    from pycafe.create_geom.external import inspect_cad as _inspect_cad

    return _inspect_cad(*args, **kwargs)


def load_mesh_with_groups(*args, **kwargs):
    from pycafe.create_geom.visualize_mesh import (
        load_mesh_with_groups as _load_mesh_with_groups,
    )

    return _load_mesh_with_groups(*args, **kwargs)


def prepare_vibroacoustic_system(*args, **kwargs):
    from pycafe.core.prepare_vibroacoustic_system import (
        prepare_vibroacoustic_system as _prepare_vibroacoustic_system,
    )

    return _prepare_vibroacoustic_system(*args, **kwargs)


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


def build_coupling_matrix(*args, **kwargs):
    """Fluid-structure coupling matrix; see
    :func:`pycafe.build_matrices.coupling.build_coupling_matrix`."""
    from pycafe.build_matrices.coupling import (
        build_coupling_matrix as _build_coupling_matrix,
    )

    return _build_coupling_matrix(*args, **kwargs)


def solve_vibroacoustic_modal(*args, **kwargs):
    """Coupled natural frequencies; see
    :func:`pycafe.solver.solver_vibroacoustic.solve_vibroacoustic_modal`."""
    from pycafe.solver.solver_vibroacoustic import (
        solve_vibroacoustic_modal as _solve_vibroacoustic_modal,
    )

    return _solve_vibroacoustic_modal(*args, **kwargs)


def solve_vibroacoustic_frequency_sweep(*args, **kwargs):
    """Coupled frequency response; see
    :func:`pycafe.solver.solver_vibroacoustic.solve_vibroacoustic_frequency_sweep`."""
    from pycafe.solver.solver_vibroacoustic import (
        solve_vibroacoustic_frequency_sweep as _sweep,
    )

    return _sweep(*args, **kwargs)
