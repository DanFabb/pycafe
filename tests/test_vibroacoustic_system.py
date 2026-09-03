"""
Tests for prepare_vibroacoustic_system: the pipeline must deliver the
two separate domains (acoustic + structural, with clamp applied) and
the interface data. The coupling built on top of them is checked in
test_coupling.py.
"""

import pathlib
import sys

import numpy as np
import pytest
from scipy.sparse.linalg import eigsh

from pycafe_vibro.prepare_vibroacoustic_system import (
    prepare_vibroacoustic_system,
)
from pycafe.solver.solver_modale import solve_modal_acoustic_reduced

LX, LY, LZ = 0.8, 0.6, 0.5
NX, NY, NZ = 8, 6, 5

RHO0, C0 = 1.204, 343.0
T, RHO_S, E, NU = 0.002, 7800.0, 210e9, 0.3


@pytest.fixture(scope="module")
def system(tmp_path_factory):
    pytest.importorskip("gmsh")
    repo_root = pathlib.Path(__file__).parent.parent
    sys.path.insert(0, str(repo_root / "coupling_geometry"))
    from make_box_plate_mesh import make_box_plate_mesh

    msh = make_box_plate_mesh(
        Lx=LX, Ly=LY, Lz=LZ, nx=NX, ny=NY, nz=NZ,
        output_path=tmp_path_factory.mktemp("mesh") / "box_plate.msh",
    )
    from pycafe.create_geom.visualize_mesh import load_mesh_with_groups
    nodes, _, _, groups = load_mesh_with_groups(str(msh))

    sys_ = prepare_vibroacoustic_system(
        nodes=nodes, groups=groups,
        rho0=RHO0, c0=C0, t=T, rho_s=RHO_S, E=E, nu=NU,
    )
    sys_["_nodes"] = nodes
    return sys_


class TestTwoDomains:

    def test_domains_present(self, system):
        assert system["acoustic"]["elem_type"] == "CHEXA8"
        assert system["structural"]["elem_type"] == "CQUAD4F"
        # The coupling is now assembled by default; see test_coupling.py
        # for its properties.
        assert set(system["coupling"]) == {"Kc", "Mc", "area_vector"}

    def test_acoustic_shapes(self, system):
        n = system["_nodes"].shape[0]
        assert system["acoustic"]["K"].shape == (n, n)
        assert np.isclose(
            system["acoustic"]["M"].sum(), LX * LY * LZ / C0**2, rtol=1e-10
        )

    def test_structural_clamp_applied(self, system):
        n_plate = (NX + 1) * (NY + 1)
        n_clamp = 2 * NX + 2 * NY
        n_free_nodes = n_plate - n_clamp
        assert system["structural"]["K_red"].shape == (
            6 * n_free_nodes, 6 * n_free_nodes
        )
        assert len(system["structural"]["clamped_nodes0"]) == n_clamp

    def test_interface_faces(self, system):
        assert system["interface"]["conn0"].shape == (NX * NY, 4)
        iface = set(np.unique(system["interface"]["conn0"]))
        assert iface == set(system["structural"]["nodes0"].tolist())

    def test_acoustic_modes(self, system):
        freqs, _ = solve_modal_acoustic_reduced(
            system["acoustic"]["K"], system["acoustic"]["M"], num_modes=1
        )
        f_exact = C0 / (2 * LX)
        assert abs(freqs[0] - f_exact) / f_exact < 0.01

    def test_structural_modes_positive(self, system):
        """Clamped plate: no rigid modes, first frequencies finite > 0."""
        K = system["structural"]["K_red"].tocsc()
        M = system["structural"]["M_red"].tocsc()
        vals, _ = eigsh(K, k=3, M=M, sigma=0.0, which="LM")
        freqs = np.sqrt(np.abs(np.sort(vals))) / (2 * np.pi)
        assert np.all(freqs > 1.0)
        assert np.all(np.isfinite(freqs))

    def test_structural_first_mode_plausible(self, system):
        """
        Clamped rectangular plate a=0.8, b=0.6 (a/b=1.33), steel t=2mm.
        Leissa CCCC: omega = lambda^2/a^2 * sqrt(D/(rho t)), with
        lambda^2 ~ 52.5 interpolated between a/b=1.0 (35.99) and
        a/b=1.5 (60.77) -> f11 ~ 41 Hz. Coarse 8x6 mesh: 10% band.
        """
        K = system["structural"]["K_red"].tocsc()
        M = system["structural"]["M_red"].tocsc()
        vals, _ = eigsh(K, k=1, M=M, sigma=0.0, which="LM")
        f1 = np.sqrt(abs(vals[0])) / (2 * np.pi)

        D = E * T**3 / (12 * (1 - NU**2))
        f_ref = 52.5 / (2 * np.pi * LX**2) * np.sqrt(D / (RHO_S * T))
        assert abs(f1 - f_ref) / f_ref < 0.10, (
            f"f1={f1:.2f} Hz vs stima Leissa {f_ref:.2f} Hz"
        )
