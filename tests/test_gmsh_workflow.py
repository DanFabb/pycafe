"""
Tests for the standardized Gmsh workflow.

Covered:
- the naming contract (roles, aliases, free boundary names);
- ``GmshModel``: option handling and the group-name checks that catch a
  typo when the mesh is written instead of three functions later;
- the parametric library: every geometry loads and satisfies the
  contract it claims;
- the validator: it accepts a good mesh and reports the specific
  failures (missing role, non-conforming interface, too coarse);
- the external path: geometric tagging rules and retagging a foreign
  mesh.
"""

import pathlib

import numpy as np
import pytest

gmsh = pytest.importorskip("gmsh")

from pycafe.create_geom import (  # noqa: E402
    GmshModel,
    TagRule,
    box_cavity,
    box_with_plate,
    describe_conventions,
    duct_2d,
    duct_with_flush_plate,
    find_group,
    largest,
    list_groups,
    load_mesh_with_groups,
    on_plane,
    plate,
    rest,
    retag_mesh,
    role_of,
    validate_mesh,
)


# ---------------------------------------------------------------------------
# Conventions
# ---------------------------------------------------------------------------

class TestConventions:

    @pytest.mark.parametrize("name,role", [
        ("fluid", "fluid"), ("Acoustic", "fluid"), ("domain", "fluid"),
        ("plate", "structure"), ("SHELL", "structure"),
        ("plate_clamp", "clamp"), ("fixed", "clamp"),
        ("rigid_walls", None), ("inlet", None), ("my_wall", None),
    ])
    def test_role_of(self, name, role):
        assert role_of(name) == role

    def test_find_group_is_case_insensitive(self):
        groups = {"Fluid": {}, "my_wall": {}}
        assert find_group(groups, "fluid") == "Fluid"
        assert find_group(groups, "structure") is None

    def test_conventions_are_printable(self):
        text = describe_conventions()
        assert "fluid" in text and "plate_clamp" in text


# ---------------------------------------------------------------------------
# GmshModel
# ---------------------------------------------------------------------------

class TestGmshModel:

    def test_unusable_outside_its_block(self):
        model = GmshModel()
        with pytest.raises(RuntimeError, match="'with' block"):
            _ = model.geo

    def test_duplicate_name_is_refused(self, tmp_path):
        with GmshModel("dup") as m:
            s = m.occ.addRectangle(0, 0, 0, 1, 1)
            m.occ.synchronize()
            m.physical(2, [s], "fluid")
            with pytest.raises(ValueError, match="already exists"):
                m.physical(2, [s], "fluid")

    def test_one_group_per_role(self):
        with GmshModel("roles") as m:
            s1 = m.occ.addRectangle(0, 0, 0, 1, 1)
            s2 = m.occ.addRectangle(2, 0, 0, 1, 1)
            m.occ.synchronize()
            m.physical(2, [s1], "fluid")
            # 'acoustic' is an alias of the same role.
            with pytest.raises(ValueError, match="role is already taken"):
                m.physical(2, [s2], "acoustic")

    def test_role_on_a_wrong_dimension_is_caught(self):
        with GmshModel("dims") as m:
            p1, p2 = m.occ.addPoint(0, 0, 0), m.occ.addPoint(1, 0, 0)
            line = m.occ.addLine(p1, p2)
            m.occ.synchronize()
            with pytest.raises(ValueError, match="dimension 2 or 3"):
                m.physical(1, [line], "fluid")
            # ... and can be overridden knowingly
            m.physical(1, [line], "fluid", strict=False)

    def test_summary_lists_roles_and_gaps(self):
        with GmshModel("summary") as m:
            s = m.occ.addRectangle(0, 0, 0, 1, 1)
            m.occ.synchronize()
            m.physical(2, [s], "fluid")
            text = m.summary()
        assert "role: fluid" in text
        assert "no group for: structure" in text

    def test_second_order_is_incomplete_by_default(self, tmp_path):
        """9-node quads would have no element in pyCAFE."""
        with GmshModel("order2", order=2, recombine=True) as m:
            s = m.occ.addRectangle(0, 0, 0, 1, 1)
            m.occ.synchronize()
            m.physical(2, [s], "fluid")
            m.generate(2, kernel="occ")
            path = m.write(tmp_path / "o2.msh")
        _, elements, _, _ = load_mesh_with_groups(str(path), verbose=False)
        assert any("8" in name for name in elements if "Quad" in name)
        assert not any("9" in name for name in elements if "Quad" in name)


# ---------------------------------------------------------------------------
# Geometry library
# ---------------------------------------------------------------------------

class TestLibrary:

    def test_box_cavity(self, tmp_path):
        path = box_cavity(0.4, 0.3, 0.5, 4, 3, 5,
                          output_path=tmp_path / "cav.msh")
        report = validate_mesh(str(path), analysis="acoustic")
        assert report.ok, report
        assert report.roles["fluid"] == "fluid"
        _, elements, _, groups = load_mesh_with_groups(str(path), verbose=False)
        assert elements["Hexahedron 8"].shape[0] == 4 * 3 * 5
        assert {"x_min", "z_max", "rigid_walls"} <= set(groups)

    def test_box_cavity_with_an_opening(self, tmp_path):
        path = box_cavity(0.4, 0.3, 0.5, 3, 3, 3, open_faces=("x_max",),
                          output_path=tmp_path / "open.msh")
        _, _, _, groups = load_mesh_with_groups(str(path), verbose=False)
        assert "opening" in groups
        assert validate_mesh(str(path), analysis="acoustic").ok

    def test_box_cavity_rejects_a_wrong_face_name(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown face name"):
            box_cavity(open_faces=("top",), output_path=tmp_path / "x.msh")

    def test_box_with_plate_is_conforming(self, tmp_path):
        path = box_with_plate(0.4, 0.3, 0.5, 4, 3, 4,
                              output_path=tmp_path / "bp.msh")
        report = validate_mesh(str(path), f_max=200.0)
        assert report.ok, report
        assert report.analysis == "vibroacoustic"
        assert any("interface conforming" in n for n in report.notes)

    def test_duct_2d(self, tmp_path):
        path = duct_2d(1.0, 0.1, 10, 2, output_path=tmp_path / "duct.msh")
        _, elements, _, groups = load_mesh_with_groups(str(path), verbose=False)
        assert elements["Quadrilateral 4"].shape[0] == 20
        assert {"inlet", "outlet", "top", "bottom"} <= set(groups)
        assert validate_mesh(str(path), analysis="acoustic").ok

    def test_plate_alone(self, tmp_path):
        path = plate(0.4, 0.3, 4, 3, output_path=tmp_path / "plate.msh")
        report = validate_mesh(str(path), analysis="structural")
        assert report.ok, report
        assert report.roles["clamp"] == "plate_clamp"

    def test_load_returns_the_mesh(self, tmp_path):
        nodes, elements, boundaries, groups = box_cavity(
            0.2, 0.2, 0.2, 2, 2, 2, output_path=tmp_path / "l.msh", load=True,
        )
        assert nodes.shape == (27, 3)
        assert "fluid" in groups


# ---------------------------------------------------------------------------
# Flush-mounted panel, from CAD or from dimensions
# ---------------------------------------------------------------------------

class TestFlushPlate:

    @pytest.fixture(scope="class")
    def duct(self, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("flush")
        return duct_with_flush_plate(
            lengths=(3.0, 1.0, 1.0), plate_length=1.0, element_size=0.25,
            output_path=tmp / "duct.msh",
        )

    def test_mesh_is_usable_and_conforming(self, duct):
        _, report = duct
        assert report.ok, report
        assert any("interface conforming" in n for n in report.notes)

    def test_panel_sits_where_asked(self, duct):
        path, _ = duct
        nodes, _, _, groups = load_mesh_with_groups(str(path), verbose=False)
        plate_nodes = np.asarray(groups["plate"]["nodes"], dtype=int) - 1
        x = nodes[plate_nodes, 0]
        z = nodes[plate_nodes, 2]
        assert np.allclose(z, z[0])                 # flush: all on one wall
        assert 1.0 - 1e-6 <= x.min() and x.max() <= 2.0 + 1e-6

    def test_clamp_is_the_border_of_the_panel(self, duct):
        path, _ = duct
        nodes, _, _, groups = load_mesh_with_groups(str(path), verbose=False)
        plate_nodes = set(np.asarray(groups["plate"]["nodes"], dtype=int))
        clamp_nodes = set(np.asarray(groups["plate_clamp"]["nodes"], dtype=int))
        assert clamp_nodes < plate_nodes            # strictly inside the panel
        idx = np.array(sorted(clamp_nodes)) - 1
        x, y = nodes[idx, 0], nodes[idx, 1]
        on_border = ((np.abs(x - 1.0) < 1e-6) | (np.abs(x - 2.0) < 1e-6)
                     | (np.abs(y - 0.0) < 1e-6) | (np.abs(y - 1.0) < 1e-6))
        assert on_border.all()

    def test_a_narrow_panel_is_cut_out_of_the_wall(self, tmp_path):
        path, report = duct_with_flush_plate(
            lengths=(3.0, 1.0, 1.0), plate_length=1.0, plate_width=0.5,
            element_size=0.25, output_path=tmp_path / "narrow.msh",
        )
        assert report.ok, report
        nodes, _, _, groups = load_mesh_with_groups(str(path), verbose=False)
        y = nodes[np.asarray(groups["plate"]["nodes"], dtype=int) - 1, 1]
        assert 0.25 - 1e-6 <= y.min() and y.max() <= 0.75 + 1e-6

    def test_panel_must_fit(self, tmp_path):
        with pytest.raises(ValueError, match="does not fit"):
            duct_with_flush_plate(
                lengths=(1.0, 1.0, 1.0), plate_length=2.0,
                element_size=0.25, output_path=tmp_path / "big.msh",
            )

    def test_panel_cannot_sit_on_an_end(self, tmp_path):
        with pytest.raises(ValueError, match="end of the duct"):
            duct_with_flush_plate(
                lengths=(3.0, 1.0, 1.0), plate_face="x_min",
                element_size=0.25, output_path=tmp_path / "end.msh",
            )

    def test_size_from_frequency(self, tmp_path):
        path, report = duct_with_flush_plate(
            lengths=(3.0, 1.0, 1.0), plate_length=1.0, f_max=200.0,
            output_path=tmp_path / "f.msh",
        )
        # c0 / (200 * 13) = 132 mm, rounded to fit the geometry.
        assert any("elements per wavelength" in n for n in report.notes)
        assert report.ok, report


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:

    def test_missing_fluid_is_reported_with_a_suggestion(self, tmp_path):
        with GmshModel("typo") as m:
            box = m.occ.addBox(0, 0, 0, 1, 1, 1)
            m.occ.synchronize()
            m.physical(3, [box], "fluids")          # a typo, not a role
            m.generate(3, kernel="occ")
            path = m.write(tmp_path / "typo.msh")
        report = validate_mesh(str(path), analysis="acoustic")
        assert not report.ok
        assert any("fluids" in e for e in report.errors)

    def test_linear_tetrahedra_are_accepted(self, tmp_path):
        """CTETRA4 has a kernel, so a plain tetrahedral mesh validates."""
        with GmshModel("tets") as m:
            box = m.occ.addBox(0, 0, 0, 1, 1, 1)
            m.occ.synchronize()
            m.size(0.2)
            m.physical(3, [box], "fluid")
            m.generate(3, kernel="occ")
            path = m.write(tmp_path / "tets.msh")
        report = validate_mesh(str(path), analysis="acoustic", f_max=100.0)
        assert report.ok, report.errors

    def test_unusable_element_type_is_reported(self, tmp_path):
        """
        Second-order tetrahedra have no kernel: the linear CTETRA4 is
        the only tetrahedron pyCAFE assembles.
        """
        with GmshModel("tets10") as m:
            box = m.occ.addBox(0, 0, 0, 1, 1, 1)
            m.occ.synchronize()
            m.size(0.5)
            m.physical(3, [box], "fluid")
            gmsh.option.setNumber("Mesh.ElementOrder", 2)
            m.generate(3, kernel="occ")
            path = m.write(tmp_path / "tets10.msh")
        report = validate_mesh(str(path), analysis="acoustic")
        assert not report.ok
        assert any("no element type pyCAFE can assemble" in e
                   for e in report.errors)

    def test_too_coarse_for_the_band(self, tmp_path):
        path = box_cavity(1.0, 1.0, 1.0, 2, 2, 2,
                          output_path=tmp_path / "coarse.msh")
        report = validate_mesh(str(path), analysis="acoustic", f_max=1000.0)
        assert not report.ok
        assert any("elements per wavelength" in e for e in report.errors)

    def test_warning_between_6_and_13_elements(self, tmp_path):
        path = box_cavity(1.0, 1.0, 1.0, 10, 10, 10,
                          output_path=tmp_path / "ok.msh")
        report = validate_mesh(str(path), analysis="acoustic", f_max=400.0)
        assert report.ok
        assert any("elements per wavelength" in w for w in report.warnings)

    def test_non_conforming_interface_is_caught(self, tmp_path):
        """A plate meshed on its own does not share the fluid nodes."""
        with GmshModel("split") as m:
            box = m.occ.addBox(0, 0, 0, 1, 1, 1)
            m.occ.synchronize()
            m.size(0.5)
            m.physical(3, [box], "fluid")
            faces = [t for _, t in m.model.getEntities(2)]
            m.physical(2, [faces[0]], "plate")
            m.generate(3, kernel="occ")
            path = m.write(tmp_path / "split.msh")
        report = validate_mesh(str(path))
        assert not report.ok
        assert report.analysis == "vibroacoustic"

    def test_report_raises_on_demand(self, tmp_path):
        path = box_cavity(1.0, 1.0, 1.0, 2, 2, 2,
                          output_path=tmp_path / "raise.msh")
        report = validate_mesh(str(path), analysis="acoustic", f_max=2000.0)
        with pytest.raises(ValueError, match="cannot be used"):
            report.raise_if_invalid()

    def test_accepts_an_already_loaded_mesh(self, tmp_path):
        mesh = box_cavity(0.3, 0.3, 0.3, 3, 3, 3,
                          output_path=tmp_path / "mem.msh", load=True)
        report = validate_mesh(mesh, analysis="acoustic")
        assert report.ok
        assert report.source == "<mesh in memory>"


# ---------------------------------------------------------------------------
# External geometry
# ---------------------------------------------------------------------------

class TestExternal:

    def test_rules_classify_by_geometry(self, tmp_path):
        with GmshModel("rules") as m:
            box = m.occ.addBox(0, 0, 0, 1, 1, 1)
            m.occ.synchronize()
            m.size(0.5)
            from pycafe.create_geom import assign_groups

            assigned = assign_groups(m, [
                largest(3, "fluid"),
                on_plane("z", 0.0, dim=2, name="plate"),
                rest(2, "rigid_walls"),
            ])
            m.generate(3, kernel="occ")
            path = m.write(tmp_path / "rules.msh")

        assert assigned["fluid"] == [box]
        assert len(assigned["plate"]) == 1
        assert len(assigned["rigid_walls"]) == 5
        groups = list_groups(str(path), verbose=False)
        assert {"fluid", "plate", "rigid_walls"} <= set(groups)

    def test_rule_that_matches_nothing_warns(self, tmp_path):
        from pycafe.create_geom import assign_groups

        with GmshModel("empty_rule") as m:
            m.occ.addBox(0, 0, 0, 1, 1, 1)
            m.occ.synchronize()
            with pytest.warns(RuntimeWarning, match="matched no entity"):
                assign_groups(m, [on_plane("z", 99.0, dim=2, name="plate")])

    def test_retag_renames_and_drops(self, tmp_path):
        path = box_cavity(0.3, 0.3, 0.3, 2, 2, 2,
                          output_path=tmp_path / "src.msh")
        out, report = retag_mesh(
            path, rename={"fluid": "acoustic"}, drop=["x_min"],
            output_path=tmp_path / "retagged.msh",
        )
        groups = list_groups(str(out), verbose=False)
        assert "acoustic" in groups and "fluid" not in groups
        assert "x_min" not in groups
        assert report.roles["fluid"] == "acoustic"   # the alias still works

    def test_retag_refuses_an_unknown_name(self, tmp_path):
        path = box_cavity(0.3, 0.3, 0.3, 2, 2, 2,
                          output_path=tmp_path / "src2.msh")
        with pytest.raises(KeyError, match="no group named"):
            retag_mesh(path, rename={"nope": "fluid"},
                       output_path=tmp_path / "out.msh")

    def test_cad_import_needs_a_file(self, tmp_path):
        from pycafe.create_geom import cad_to_mesh

        with pytest.raises(FileNotFoundError):
            cad_to_mesh(tmp_path / "missing.step", tmp_path / "o.msh")


# ---------------------------------------------------------------------------
# The STEP file shipped with the repository, when it is there
# ---------------------------------------------------------------------------

STEP = pathlib.Path(__file__).parent.parent / "Geom" / "Tubo_1m_1m.stp"


@pytest.mark.skipif(not STEP.exists(), reason="Geom/Tubo_1m_1m.stp not present")
class TestStepDuct:

    def test_units_and_dimensions_are_recovered(self, tmp_path):
        """The file is in millimetres; the model must come out in metres."""
        path, report = duct_with_flush_plate(
            cad_path=STEP, plate_length=1.0, f_max=200.0,
            output_path=tmp_path / "tubo.msh",
        )
        assert report.ok, report
        nodes, _, _, groups = load_mesh_with_groups(str(path), verbose=False)
        extent = nodes.max(axis=0) - nodes.min(axis=0)
        assert np.allclose(extent, [3.0, 1.0, 1.0], atol=1e-3)

        plate_nodes = np.asarray(groups["plate"]["nodes"], dtype=int) - 1
        x = nodes[plate_nodes, 0]
        assert np.isclose(x.min(), 1.0, atol=1e-6)
        assert np.isclose(x.max(), 2.0, atol=1e-6)

    def test_non_box_cad_is_refused(self, tmp_path):
        """A sphere would mesh tetrahedral, which has no acoustic kernel."""
        sphere = tmp_path / "ball.step"
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.model.add("ball")
            gmsh.model.occ.addSphere(0, 0, 0, 1)
            gmsh.model.occ.synchronize()
            gmsh.write(str(sphere))
        finally:
            gmsh.finalize()

        with pytest.raises(ValueError, match="not box-shaped"):
            duct_with_flush_plate(cad_path=sphere, element_size=0.3,
                                  output_path=tmp_path / "ball.msh")
