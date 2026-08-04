"""
Tests for domain association from Gmsh physical groups (domains.py).

On the coupled box+plate mesh the global elements dict mixes plate and
wall quads in one "Quadrilateral 4" array: the physical groups must
drive the domain split — fluid = "fluid" group (HEXA8 only),
structure = "plate" group (48 plate QUAD4 only, walls excluded).
"""

import pathlib
import sys

import numpy as np
import pytest

from pycafe.build_matrices.domains import (
    identify_domains,
    build_KM_acoustic_domain,
    build_KM_structural_domain,
)
from pycafe.solver.solver_modale import solve_modal_acoustic_reduced

LX, LY, LZ = 0.8, 0.6, 0.5
NX, NY, NZ = 8, 6, 5


@pytest.fixture(scope="module")
def box_plate(tmp_path_factory):
    gmsh = pytest.importorskip("gmsh")  # noqa: F841
    repo_root = pathlib.Path(__file__).parent.parent
    sys.path.insert(0, str(repo_root / "coupling_geometry"))
    from make_box_plate_mesh import make_box_plate_mesh

    msh = make_box_plate_mesh(
        Lx=LX, Ly=LY, Lz=LZ, nx=NX, ny=NY, nz=NZ,
        output_path=tmp_path_factory.mktemp("mesh") / "box_plate.msh",
    )

    from pycafe.create_geom.visualize_mesh import load_mesh_with_groups
    return load_mesh_with_groups(str(msh))


class TestIdentifyDomains:

    def test_fluid_and_structure_found(self, box_plate):
        _, _, _, groups = box_plate
        domains = identify_domains(groups)

        assert domains["fluid"]["elem_type"] == "CHEXA8"
        assert domains["fluid"]["group"] == "fluid"
        assert domains["fluid"]["conn0"].shape == (NX * NY * NZ, 8)

        assert domains["structure"]["elem_type"] == "CQUAD4F"
        assert domains["structure"]["group"] == "plate"
        # solo i quad della piastra, non i 188 delle pareti
        assert domains["structure"]["conn0"].shape == (NX * NY, 4)

    def test_structure_nodes_subset_of_fluid(self, box_plate):
        _, _, _, groups = box_plate
        domains = identify_domains(groups)
        s_nodes = set(domains["structure"]["nodes"])
        f_nodes = set(domains["fluid"]["nodes"])
        assert s_nodes <= f_nodes

    def test_missing_fluid_raises(self):
        with pytest.raises(RuntimeError, match="No fluid domain"):
            identify_domains({"qualcosa": {"dim": 2, "tag": 1,
                                           "nodes": [], "elements": {}}})


class TestDomainAssembly:

    def test_acoustic_from_fluid_group(self, box_plate):
        nodes, _, _, groups = box_plate
        domains = identify_domains(groups)
        K, M, _, elem_type = build_KM_acoustic_domain(nodes, domains, 343.0)

        assert elem_type == "CHEXA8"
        n = nodes.shape[0]
        assert K.shape == (n, n)
        # massa acustica totale = V / c0^2
        V = LX * LY * LZ
        assert np.isclose(M.sum(), V / 343.0**2, rtol=1e-10)

        # sanity modale vs analitico (primo modo lungo x)
        freqs, _ = solve_modal_acoustic_reduced(K, M, num_modes=1)
        f_exact = 343.0 / (2.0 * LX)
        assert abs(freqs[0] - f_exact) / f_exact < 0.01

    def test_structural_from_plate_group(self, box_plate):
        nodes, _, _, groups = box_plate
        domains = identify_domains(groups)
        t, rho, E, nu = 0.002, 7800.0, 210e9, 0.3
        K, M, _, elem_type = build_KM_structural_domain(
            nodes, domains, t, rho, E, nu
        )

        assert elem_type == "CQUAD4F"
        n = nodes.shape[0]
        assert K.shape == (6 * n, 6 * n)

        # massa strutturale = rho * t * area della PIASTRA sola
        # (le pareti non devono contribuire)
        uz = np.zeros(6 * n)
        uz[2::6] = 1.0
        assert np.isclose(uz @ (M @ uz), rho * t * LX * LY, rtol=1e-10)

        # i DOF dei nodi non-piastra sono vuoti
        plate_nodes0 = np.array(domains["structure"]["nodes"], dtype=int) - 1
        mask = np.ones(n, dtype=bool)
        mask[plate_nodes0] = False
        empty_dofs = (np.where(mask)[0][:, None] * 6
                      + np.arange(6)).ravel()
        assert K[empty_dofs, :].count_nonzero() == 0

    def test_dispatcher_with_groups(self, box_plate):
        nodes, elements, _, groups = box_plate
        from pycafe.build_matrices.assembly_dispatcher import build_KM_acoustic

        K, M, _, elem_type = build_KM_acoustic(
            nodes, elements, 343.0, groups=groups
        )
        assert elem_type == "CHEXA8"
        assert np.isclose(M.sum(), LX * LY * LZ / 343.0**2, rtol=1e-10)
