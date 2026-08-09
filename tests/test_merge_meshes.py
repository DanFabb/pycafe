"""
Tests for merging independently meshed domains into one file.

Covered:
- two meshes end up in one file with disjoint node numbering, every
  group preserved, and the loader reads it back;
- the merged mesh is what the validator calls a usable non-conforming
  vibroacoustic model;
- `rename` maps foreign group names onto the pyCAFE roles;
- a source without groups is refused;
- the CadFile split pipeline (`independent_structure=True`) produces a
  tetrahedral fluid and a quadrilateral shell from one STEP-like solid,
  and the assembled system carries the interpolated coupling.
"""

import numpy as np
import pytest

from pycafe.create_geom.merge import merge_meshes

gmsh = pytest.importorskip("gmsh")

from pycafe.create_geom.gmsh_workflow import GmshModel  # noqa: E402
from pycafe.create_geom.validation import validate_mesh  # noqa: E402
from pycafe.create_geom.visualize_mesh import load_mesh_with_groups  # noqa: E402


def _fluid_box(path, name="fluid"):
    with GmshModel("f") as m:
        box = m.occ.addBox(0, 0, 0, 0.4, 0.3, 0.2)
        m.occ.synchronize()
        m.size(0.1)
        m.physical(3, [box], name)
        m.generate(3, kernel="occ")
        return m.write(path)


def _plate_on_top(path, name="plate"):
    with GmshModel("p", recombine=True) as m:
        rect = m.occ.addRectangle(0, 0, 0.2, 0.4, 0.3)
        m.occ.synchronize()
        m.size(0.07)
        m.physical(2, [rect], name)
        m.physical(1, [abs(t) for _d, t in m.model.getBoundary(
            [(2, rect)], oriented=False, combined=True)], "plate_clamp")
        m.generate(2, kernel="occ")
        return m.write(path)


class TestMergeMeshes:
    def test_nodes_stay_apart_and_groups_survive(self, tmp_path):
        p1 = _fluid_box(tmp_path / "f.msh")
        p2 = _plate_on_top(tmp_path / "p.msh")
        out = merge_meshes([p1, p2], tmp_path / "m.msh")

        nodes, _, _, groups = load_mesh_with_groups(str(out), verbose=False)
        n1 = load_mesh_with_groups(str(p1), verbose=False)[0].shape[0]
        n2 = load_mesh_with_groups(str(p2), verbose=False)[0].shape[0]
        assert nodes.shape[0] == n1 + n2
        assert set(groups) == {"fluid", "plate", "plate_clamp"}

        fluid_nodes = {int(n) for conn in groups["fluid"]["elements"].values()
                       for n in np.unique(conn)}
        plate_nodes = {int(n) for conn in groups["plate"]["elements"].values()
                       for n in np.unique(conn)}
        assert not fluid_nodes & plate_nodes

    def test_merged_mesh_is_usable_nonconforming(self, tmp_path):
        out = merge_meshes(
            [_fluid_box(tmp_path / "f.msh"), _plate_on_top(tmp_path / "p.msh")],
            tmp_path / "m.msh",
        )
        report = validate_mesh(str(out))
        assert report.analysis == "vibroacoustic"
        assert report.ok
        assert any("not conforming" in w for w in report.warnings)

    def test_rename_maps_foreign_names(self, tmp_path):
        p1 = _fluid_box(tmp_path / "f.msh", name="AIR")
        p2 = _plate_on_top(tmp_path / "p.msh")
        out = merge_meshes([p1, p2], tmp_path / "m.msh",
                           rename=[{"AIR": "fluid"}, {}])
        _, _, _, groups = load_mesh_with_groups(str(out), verbose=False)
        assert "fluid" in groups and "AIR" not in groups

    def test_groupless_sources_are_refused(self, tmp_path):
        with GmshModel("bare") as m:
            m.occ.addBox(0, 0, 0, 1, 1, 1)
            m.occ.synchronize()
            m.size(0.5)
            m.generate(3, kernel="occ")
            bare = m.write(tmp_path / "bare.msh")
        with pytest.raises(ValueError, match="physical group"):
            merge_meshes([bare], tmp_path / "m.msh")


class TestIndependentStructure:
    """The split CadFile pipeline, on a cylinder through a box."""

    @pytest.fixture(scope="class")
    def step_file(self, tmp_path_factory):
        """A box with a cylindrical hole, plus the cylinder: a curved
        wet surface that only a split mesh can carry as quads."""
        path = tmp_path_factory.mktemp("cad") / "box_cyl.step"
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("cad")
        box = gmsh.model.occ.addBox(-0.05, -0.05, -0.05, 0.1, 0.1, 0.1)
        cyl = gmsh.model.occ.addCylinder(0, 0, -0.02, 0, 0, 0.04, 0.01)
        gmsh.model.occ.cut([(3, box)], [(3, cyl)], removeTool=False)
        gmsh.model.occ.synchronize()
        gmsh.write(str(path))
        gmsh.finalize()
        return path

    def test_split_mesh_and_coupling(self, step_file):
        from pycafe.core.model_spec import (
            AIR, CadFile, ModelSpec, aluminium, build_model,
        )

        # Tags read the same way a user would: fluid is the cut box,
        # the shell is the cylinder side surface.
        from pycafe.create_geom.external import inspect_cad

        found = inspect_cad(step_file, units="m", verbose=False)
        fluid_tag = found[3][0].tag
        side = [i.tag for i in found[2]
                if 0.019 < (i.bbox_max - i.bbox_min)[0] < 0.021]
        assert side, "the cylinder side surface was not found"

        spec = ModelSpec(
            geometry=CadFile(step_file, units="m", recombine=False,
                             fluid=[fluid_tag], structure=side,
                             independent_structure=True),
            # The unstructured mesher overshoots the target size on curved
            # boundaries by a factor close to two, so a density well above
            # the floor is asked to keep the largest element legal.
            fluid=AIR, structure=aluminium(t=1e-3),
            f_max=2000.0, elements_per_wavelength=15,
            work_dir=step_file.parent,
        )
        with pytest.warns(RuntimeWarning):
            model = build_model(spec, show=False)

        report = model["report"]
        assert report.ok
        assert "Tetrahedron 4" in report.element_counts
        assert "Quadrilateral 4" in report.element_counts

        system = model["system"]
        assert system["interface"]["conforming"] is False
        nc = system["coupling"]["nonconforming"]
        assert nc["consistency"]["relative_error"] < 1e-9
