__version__ = "1.0.0"


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
