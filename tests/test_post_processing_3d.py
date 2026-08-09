"""
Tests for the 3D post-processing module and the anechoic boundary.

Covered:
- surface extraction: only the outer skin of a hexahedral block is
  rendered, interior faces are dropped; a 2D mesh falls back to its own
  elements;
- movie writing with matplotlib alone: an animated GIF needs no external
  codec, and a video container without FFMpeg fails with a clear error;
- ``AcousticBC.add_anechoic``: the named plane-wave radiation condition
  is exactly ``Z = rho0 * c0`` and reaches ``C`` as ``rho0 * A = 1 / c0``,
  so that ``+j omega C = j k S`` damps an outgoing wave.
"""

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")
from matplotlib import animation  # noqa: E402

from pycafe.boundary_condition.acoustic_bc import (  # noqa: E402
    AcousticBC,
    make_admittance,
)
from pycafe.post_processing.post_processing import (  # noqa: E402
    is_three_dimensional,
    weighted_pressure_at_point,
)
from pycafe.post_processing.post_processing_3d import (  # noqa: E402
    P_REF,
    animate_pressure_3d,
    clip_elements,
    default_movie_extension,
    ensure_ffmpeg,
    extract_surface_faces,
    plot_pressure_3d,
    pressure_at_point_3d,
)

RHO = 1.225
C0 = 343.0


def _hex_block(nx=3, ny=2, nz=2):
    """Structured hexahedral block, Gmsh 'Hexahedron 8' ordering, 1-based."""
    xs = np.linspace(0.0, 1.0, nx + 1)
    ys = np.linspace(0.0, 0.5, ny + 1)
    zs = np.linspace(0.0, 0.4, nz + 1)

    idx = {}
    nodes = []
    for iz, z in enumerate(zs):
        for iy, y in enumerate(ys):
            for ix, x in enumerate(xs):
                idx[(ix, iy, iz)] = len(nodes)
                nodes.append([x, y, z])

    conn = []
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                conn.append([
                    idx[(ix, iy, iz)], idx[(ix + 1, iy, iz)],
                    idx[(ix + 1, iy + 1, iz)], idx[(ix, iy + 1, iz)],
                    idx[(ix, iy, iz + 1)], idx[(ix + 1, iy, iz + 1)],
                    idx[(ix + 1, iy + 1, iz + 1)], idx[(ix, iy + 1, iz + 1)],
                ])

    nodes = np.array(nodes, dtype=float)
    elements = {"Hexahedron 8": np.array(conn, dtype=int) + 1}
    return nodes, elements, (nx, ny, nz)


#  SURFACE EXTRACTION
def test_skin_of_hex_block_has_no_interior_face():
    nodes, elements, (nx, ny, nz) = _hex_block()
    faces = extract_surface_faces(elements)

    # A box skin holds 2*(nx*ny + ny*nz + nx*nz) quads.
    assert len(faces) == 2 * (nx * ny + ny * nz + nx * nz)

    # Every rendered face must be unique and lie on the bounding box.
    keys = {tuple(sorted(f.tolist())) for f in faces}
    assert len(keys) == len(faces)

    lo, hi = nodes.min(axis=0), nodes.max(axis=0)
    for f in faces:
        xyz = nodes[f]
        on_box = [
            np.allclose(xyz[:, k], lo[k]) or np.allclose(xyz[:, k], hi[k])
            for k in range(3)
        ]
        assert any(on_box)


def test_surface_mesh_falls_back_to_its_own_elements():
    elements = {"Quadrilateral 4": np.array([[1, 2, 3, 4], [2, 5, 6, 3]])}
    faces = extract_surface_faces(elements)
    assert [f.tolist() for f in faces] == [[0, 1, 2, 3], [1, 4, 5, 2]]


def test_unknown_element_type_is_reported():
    with pytest.raises(ValueError, match="No renderable element type"):
        extract_surface_faces({"Point 1": np.array([[1]])})


#  ANIMATION
def _sweep_field(nodes, frequencies):
    """Plane wave along x, one column per frequency."""
    k = 2.0 * np.pi * np.asarray(frequencies) / C0
    return np.exp(-1j * np.outer(nodes[:, 0], k))


def test_gif_movie_needs_no_external_codec(tmp_path):
    nodes, elements, _ = _hex_block()
    frequencies = np.linspace(100.0, 400.0, 4)
    p_full = _sweep_field(nodes, frequencies)

    out = tmp_path / "sweep.gif"
    anim = animate_pressure_3d(
        nodes, elements, p_full, frequencies,
        mode="sweep", filename=str(out), fps=4,
    )
    assert out.exists() and out.stat().st_size > 0
    assert anim is not None


def test_time_mode_animates_one_frequency_over_a_period(tmp_path):
    nodes, elements, _ = _hex_block()
    frequencies = np.linspace(100.0, 400.0, 4)
    p_full = _sweep_field(nodes, frequencies)

    out = tmp_path / "time.gif"
    animate_pressure_3d(
        nodes, elements, p_full, frequencies,
        mode="time", freq_id=2, n_time_steps=8, filename=str(out), fps=8,
    )
    assert out.exists() and out.stat().st_size > 0


def test_bad_mode_and_bad_extension_are_rejected(tmp_path):
    nodes, elements, _ = _hex_block()
    frequencies = np.array([100.0, 200.0])
    p_full = _sweep_field(nodes, frequencies)

    with pytest.raises(ValueError, match="Unknown mode"):
        animate_pressure_3d(nodes, elements, p_full, frequencies, mode="xyz")

    with pytest.raises(ValueError, match="Unsupported movie extension"):
        animate_pressure_3d(
            nodes, elements, p_full, frequencies,
            filename=str(tmp_path / "movie.txt"),
        )


def test_video_movie_when_ffmpeg_is_installed(tmp_path):
    nodes, elements, _ = _hex_block()
    frequencies = np.linspace(100.0, 400.0, 4)
    p_full = _sweep_field(nodes, frequencies)
    out = tmp_path / "sweep.mp4"

    if ensure_ffmpeg() is None:
        # No binary: the failure must name the missing dependency
        # rather than surfacing a matplotlib backend error.
        with pytest.raises(RuntimeError, match="No FFmpeg binary"):
            animate_pressure_3d(
                nodes, elements, p_full, frequencies, filename=str(out),
            )
        return

    animate_pressure_3d(
        nodes, elements, p_full, frequencies, filename=str(out), fps=4,
    )
    assert out.exists() and out.stat().st_size > 0


def test_default_movie_extension_matches_ffmpeg_availability():
    expected = ".mp4" if ensure_ffmpeg() is not None else ".gif"
    assert default_movie_extension() == expected


def test_mismatched_sweep_length_is_rejected():
    nodes, elements, _ = _hex_block()
    p_full = np.ones((nodes.shape[0], 3), dtype=complex)
    with pytest.raises(ValueError, match="3 columns"):
        animate_pressure_3d(nodes, elements, p_full, [100.0, 200.0])


def test_static_plot_runs_on_a_hex_block():
    nodes, elements, _ = _hex_block()
    p = _sweep_field(nodes, [250.0])[:, 0]
    for part in ("abs", "spl", "real", "imag"):
        fig, ax = plot_pressure_3d(nodes, elements, p, part=part, show=False)
        assert fig is not None and ax is not None


def test_unknown_part_is_rejected():
    nodes, elements, _ = _hex_block()
    p = _sweep_field(nodes, [250.0])[:, 0]
    with pytest.raises(ValueError, match="Unknown part"):
        plot_pressure_3d(nodes, elements, p, part="modulus", show=False)


def test_spl_is_the_level_of_the_amplitude(tmp_path):
    nodes, elements, _ = _hex_block()
    frequencies = np.array([250.0])
    p_full = np.full((nodes.shape[0], 1), 2.0 + 0.0j)  # |p| = 2 Pa

    from pycafe.post_processing.post_processing_3d import _extract_part

    spl = _extract_part(p_full[:, 0], "spl")
    assert np.allclose(spl, 20.0 * np.log10(2.0 / P_REF))

    # A nodal line must not blow the scale up: the default window is
    # 60 dB below the peak.
    p_full[0, 0] = 0.0
    out = tmp_path / "spl.gif"
    animate_pressure_3d(
        nodes, elements, p_full, frequencies,
        mode="sweep", part="spl", filename=str(out), fps=2,
    )
    assert out.exists()


#  CLIPPING
def test_clip_keeps_one_side_and_exposes_the_cut():
    nodes, elements, (nx, ny, nz) = _hex_block(nx=4, ny=2, nz=2)
    x_mid = 0.5 * (nodes[:, 0].min() + nodes[:, 0].max())

    clipped = clip_elements(nodes, elements, axis="x", value=x_mid, keep="<")
    assert clipped["Hexahedron 8"].shape[0] == (nx // 2) * ny * nz

    # Every surviving centroid is on the kept side.
    conn = clipped["Hexahedron 8"] - 1
    assert np.all(nodes[conn, 0].mean(axis=1) < x_mid)

    # The cut plane itself now appears among the rendered faces.
    faces = extract_surface_faces(clipped)
    on_cut = [f for f in faces if np.allclose(nodes[f, 0], x_mid)]
    assert len(on_cut) == ny * nz


def test_clip_rejects_a_plane_outside_the_mesh():
    nodes, elements, _ = _hex_block()
    with pytest.raises(ValueError, match="leaves no element"):
        clip_elements(nodes, elements, axis="z", value=-1.0, keep="<")
    with pytest.raises(ValueError, match="Unknown axis"):
        clip_elements(nodes, elements, axis="w")
    with pytest.raises(ValueError, match="Unknown side"):
        clip_elements(nodes, elements, axis="x", keep="=")


def test_clipped_movie_is_written(tmp_path):
    nodes, elements, _ = _hex_block()
    frequencies = np.linspace(100.0, 300.0, 3)
    p_full = _sweep_field(nodes, frequencies)
    out = tmp_path / "cut.gif"

    animate_pressure_3d(
        nodes, elements, p_full, frequencies,
        mode="sweep", clip={"axis": "y", "keep": "<"},
        filename=str(out), fps=3,
    )
    assert out.exists() and out.stat().st_size > 0


#  POINT RESPONSE AND DISPATCH
def test_point_response_interpolates_in_3d():
    nodes, elements, _ = _hex_block()
    frequencies = np.linspace(100.0, 400.0, 4)
    p_full = _sweep_field(nodes, frequencies)

    # On top of a node, the response is that node's own response.
    target = nodes[7]
    p_point = pressure_at_point_3d(p_full, nodes, target, num_closest=1)
    assert np.allclose(p_point, p_full[7])
    assert p_point.shape == frequencies.shape


def test_weighted_point_still_works_in_2d_and_accepts_3d():
    nodes2d = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    p_full = np.array([[1.0 + 0j], [2.0 + 0j], [3.0 + 0j]])
    assert np.allclose(
        weighted_pressure_at_point(p_full, nodes2d, [0.0, 0.0], num_closest=1),
        [1.0],
    )

    # A 2D point on a 3D mesh is read on the plane z = 0.
    nodes3d = np.column_stack([nodes2d, np.zeros(3)])
    assert np.allclose(
        weighted_pressure_at_point(p_full, nodes3d, [1.0, 0.0], num_closest=1),
        [2.0],
    )


def test_dispatch_detects_a_3d_mesh():
    nodes, elements, _ = _hex_block()
    assert is_three_dimensional(nodes, elements)

    # A flat mesh made of surface elements stays on the 2D menu.
    flat = np.column_stack([nodes[:, :2], np.zeros(nodes.shape[0])])
    assert not is_three_dimensional(flat, {"Quadrilateral 4": np.array([[1, 2, 3, 4]])})
    assert not is_three_dimensional(flat[:, :2])


#  ANECHOIC BOUNDARY
def test_add_anechoic_is_the_characteristic_impedance():
    bc = AcousticBC()
    bc.add_anechoic("Omega_e")

    assert len(bc.impedance) == 1
    entry = bc.impedance[0]
    assert entry.selection == ["Omega_e"]

    admittance = make_admittance(entry.Z, RHO, C0)
    for omega in (2.0 * np.pi * 50.0, 2.0 * np.pi * 2000.0):
        # Absolute impedance is rho0*c0, so the term entering C,
        # rho0 * A, is 1/c0: j*omega*C then equals j*k*S.
        assert np.isclose(1.0 / admittance(omega), RHO * C0)
        assert np.isclose(RHO * admittance(omega), 1.0 / C0)


def test_anechoic_aliases_are_the_same_condition():
    assert AcousticBC.add_radiation is AcousticBC.add_anechoic
    assert AcousticBC.add_non_reflecting is AcousticBC.add_anechoic

    bc = AcousticBC()
    bc.add_radiation(["Omega_e", "outlet"])
    assert bc.impedance[0].selection == ["Omega_e", "outlet"]
    assert bc.impedance[0].Z == 1.0
