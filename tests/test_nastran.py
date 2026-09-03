"""
Tests for reading a Nastran bulk data deck.

Covered:
- the card reader: small, large and free field, continuations, the
  implicit exponent, ``THRU`` in an ``SPC1``;
- the cards Gmsh discards (``PSHELL``, ``PSOLID``, ``MAT1``, ``MAT10``,
  ``SPC1``) and the roles they suggest;
- the overfull free field line the Gmsh reader keeps reading as nodes,
  rejected before it can produce wrong elements;
- ``bdf_to_mesh``: property ids become physical groups, the free faces
  of the fluid become named boundaries, the constrained grids become
  the support;
- ``NastranFile`` inside a :class:`ModelSpec`, up to an assembled
  coupled system with the thickness and the materials taken from the
  deck.

The deck is written by the tests themselves: a 2 x 2 x 2 block of
hexahedra with a shell on one face, so every check is against numbers
that can be counted by hand.
"""

import textwrap
import warnings

import numpy as np
import pytest

gmsh = pytest.importorskip("gmsh")

from pycafe.core.model_spec import (  # noqa: E402
    AIR,
    ModelSpec,
    NastranFile,
    build_mesh,
    build_model,
    materials_from_deck,
)
from pycafe_vibro.structure import aluminium  # noqa: E402
from pycafe.create_geom import (  # noqa: E402
    bdf_to_mesh,
    inspect_bdf,
    load_mesh_with_groups,
    on_plane,
    read_bdf,
)
from pycafe.create_geom.nastran import _to_float  # noqa: E402

# The block: 3 x 3 x 3 grids over a 0.4 x 0.3 x 0.5 box, 8 hexahedra,
# and the 4 quadrilaterals of the z = 0 face as a shell.
NX = NY = NZ = 3
LX, LY, LZ = 0.4, 0.3, 0.5

PID_FLUID = 101
PID_PLATE = 202
MID_FLUID = 10
MID_SHELL = 20


def _grid_id(i, j, k):
    return 1 + i + NX * (j + NY * k)


def _write_deck(path, *, fields="small", with_shell=True, with_spc=True):
    """A cavity deck in the requested field format, written by hand."""
    lines = ["$ block of hexahedra, for the tests", "BEGIN BULK"]

    def card(name, values, free):
        # Both formats hold eight data fields per line and continue on
        # the next one, which is what a bulk data card is allowed to do
        # and what the Gmsh reader handles.
        text = [f"{v}" for v in values]
        out, row = [], (name if free else f"{name:<8s}")
        for n, value in enumerate(text):
            if n and n % 8 == 0:
                out.append(row)
                row = "" if free else " " * 8
            row += ("," + value) if free else f"{value:<8s}"
        out.append(row)
        return out

    free = fields == "free"
    for k in range(NZ):
        for j in range(NY):
            for i in range(NX):
                lines += card("GRID", [
                    _grid_id(i, j, k), "",
                    f"{i * LX / (NX - 1):<8.4f}"[:8],
                    f"{j * LY / (NY - 1):<8.4f}"[:8],
                    f"{k * LZ / (NZ - 1):<8.4f}"[:8],
                ], free)

    eid = 1
    for k in range(NZ - 1):
        for j in range(NY - 1):
            for i in range(NX - 1):
                corners = [
                    _grid_id(i, j, k), _grid_id(i + 1, j, k),
                    _grid_id(i + 1, j + 1, k), _grid_id(i, j + 1, k),
                    _grid_id(i, j, k + 1), _grid_id(i + 1, j, k + 1),
                    _grid_id(i + 1, j + 1, k + 1), _grid_id(i, j + 1, k + 1),
                ]
                lines += card("CHEXA", [eid, PID_FLUID] + corners, free)
                eid += 1

    if with_shell:
        for j in range(NY - 1):
            for i in range(NX - 1):
                lines += card("CQUAD4", [
                    eid, PID_PLATE,
                    _grid_id(i, j, 0), _grid_id(i + 1, j, 0),
                    _grid_id(i + 1, j + 1, 0), _grid_id(i, j + 1, 0),
                ], free)
                eid += 1

    lines += card("PSOLID", [PID_FLUID, MID_FLUID, "", "", "", "", "PFLUID"],
                  free)
    lines += card("MAT10", [MID_FLUID, "", "1.204", "343."], free)
    if with_shell:
        lines += card("PSHELL", [PID_PLATE, MID_SHELL, ".002"], free)
        lines += card("MAT1", [MID_SHELL, "70.+9", "", ".33", "2700."], free)

    if with_spc:
        edge = sorted({
            _grid_id(i, j, 0)
            for i in range(NX) for j in range(NY)
            if i in (0, NX - 1) or j in (0, NY - 1)
        })
        lines += card("SPC1", [10, "123456"] + edge, free)

    lines.append("ENDDATA")
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture
def deck_path(tmp_path):
    return _write_deck(tmp_path / "cavity.bdf")


@pytest.fixture
def fluid_deck_path(tmp_path):
    """The same block with no shell on it: an acoustic deck."""
    return _write_deck(tmp_path / "fluid.bdf", with_shell=False)


# The card reader

class TestCardReader:

    @pytest.mark.parametrize("text,value", [
        ("1.5", 1.5), ("70.+9", 70e9), ("1.5-3", 1.5e-3),
        ("-2.5+2", -250.0), ("1.5E+3", 1500.0), ("", None),
    ])
    def test_implicit_exponent(self, text, value):
        assert _to_float(text) == value

    def test_small_field(self, deck_path):
        deck = read_bdf(deck_path)
        assert deck.n_grids == NX * NY * NZ
        assert dict(deck.elements[PID_FLUID]) == {"CHEXA": 8}
        assert dict(deck.elements[PID_PLATE]) == {"CQUAD4": 4}

    def test_free_field_without_continuation_reads(self, tmp_path):
        # A CQUAD4 fits on one line, so this deck is fine as it is.
        path = tmp_path / "free.bdf"
        path.write_text(textwrap.dedent("""\
            BEGIN BULK
            GRID,1,,0.,0.,0.
            GRID,2,,1.,0.,0.
            GRID,3,,1.,1.,0.
            GRID,4,,0.,1.,0.
            CQUAD4,1,202,1,2,3,4
            PSHELL,202,20,.002
            ENDDATA
            """))
        deck = read_bdf(path)
        assert dict(deck.elements[202]) == {"CQUAD4": 1}
        assert deck.properties[202].thickness == pytest.approx(2e-3)

    def test_free_field_continuation_reads(self, tmp_path):
        # Eight data fields then a continuation is the legal way to
        # write a CHEXA in free field, and Gmsh reads it.
        path = _write_deck(tmp_path / "free.bdf", fields="free")
        deck = read_bdf(path)
        assert dict(deck.elements[PID_FLUID]) == {"CHEXA": 8}

        out, report = bdf_to_mesh(path, tmp_path / "free.msh",
                                  fluid=[PID_FLUID], structure=[PID_PLATE])
        assert report.ok
        _n, _e, _b, groups = load_mesh_with_groups(str(out), verbose=False)
        assert groups["fluid"]["elements"]["Hexahedron 8"].shape == (8, 8)

    def test_an_overfull_free_field_line_is_rejected(self, tmp_path):
        # Nine values before the continuation: Gmsh reads the ninth as a
        # node and stops on 'Wrong node index'.
        path = tmp_path / "overfull.bdf"
        path.write_text(textwrap.dedent("""\
            BEGIN BULK
            GRID,1,,0.,0.,0.
            GRID,2,,1.,0.,0.
            GRID,3,,1.,1.,0.
            GRID,4,,0.,1.,0.
            GRID,5,,0.,0.,1.
            GRID,6,,1.,0.,1.
            GRID,7,,1.,1.,1.
            GRID,8,,0.,1.,1.
            CHEXA,1,101,1,2,3,4,5,6,7,
            ,8
            ENDDATA
            """))
        with pytest.raises(ValueError, match="eight data fields"):
            read_bdf(path)
        # ...and can still be looked at, which is the point of strict.
        with pytest.warns(RuntimeWarning):
            deck = read_bdf(path, strict=False)
        assert 101 in deck.elements
        with pytest.raises(Exception):
            bdf_to_mesh(path, tmp_path / "overfull.msh", fluid=[101],
                        deck=deck)

    def test_large_field_grid(self, tmp_path):
        path = tmp_path / "large.bdf"
        path.write_text(textwrap.dedent("""\
            BEGIN BULK
            GRID*   1                               2.0             0.0
            *       0.5
            ENDDATA
            """))
        assert read_bdf(path).n_grids == 1

    def test_executive_control_is_not_read_as_bulk(self, tmp_path):
        path = tmp_path / "run.bdf"
        path.write_text(textwrap.dedent("""\
            SOL 111
            CEND
            SPC = 10
            BEGIN BULK
            GRID    1               0.      0.      0.
            ENDDATA
            """))
        deck = read_bdf(path)
        assert deck.n_grids == 1
        assert deck.constraints == {}


# What the deck says about the model

class TestDeck:

    def test_properties_and_materials(self, deck_path):
        deck = read_bdf(deck_path)
        assert deck.properties[PID_FLUID].card == "PSOLID"
        assert deck.properties[PID_FLUID].fluid_flag
        assert deck.properties[PID_PLATE].thickness == pytest.approx(2e-3)

        fluid = deck.material_of(PID_FLUID)
        assert fluid.is_fluid and fluid.rho == pytest.approx(1.204)
        # bulk is completed from rho and c0
        assert fluid.bulk == pytest.approx(1.204 * 343.0 ** 2)

        shell = deck.material_of(PID_PLATE)
        assert (shell.E, shell.nu, shell.rho) == (70e9, 0.33, 2700.0)

    def test_spc1_thru_expands(self, tmp_path):
        path = tmp_path / "spc.bdf"
        path.write_text(textwrap.dedent("""\
            BEGIN BULK
            SPC1    10      123456  4       THRU    9
            ENDDATA
            """))
        assert read_bdf(path).constraints[10].nodes == (4, 5, 6, 7, 8, 9)

    def test_suggested_roles(self, deck_path):
        roles = read_bdf(deck_path).suggest_roles()
        assert [pid for pid, _why in roles["fluid"]] == [PID_FLUID]
        assert [pid for pid, _why in roles["structure"]] == [PID_PLATE]
        assert roles["unusable"] == []

    def test_elastic_solid_is_reported_as_unusable(self, tmp_path):
        path = tmp_path / "solid.bdf"
        path.write_text(textwrap.dedent("""\
            BEGIN BULK
            GRID    1               0.      0.      0.
            GRID    2               1.      0.      0.
            GRID    3               0.      1.      0.
            GRID    4               0.      0.      1.
            CTETRA  1       7       1       2       3       4
            PSOLID  7       8
            MAT1    8       210.+9          .3      7850.
            ENDDATA
            """))
        roles = read_bdf(path).suggest_roles()
        assert roles["fluid"] == []
        assert roles["unusable"][0][0] == 7
        assert "shells only" in roles["unusable"][0][1]

    def test_inspect_survives_a_deck_gmsh_refuses(self, tmp_path, capsys):
        path = tmp_path / "overfull.bdf"
        path.write_text(textwrap.dedent("""\
            BEGIN BULK
            GRID,1,,0.,0.,0.
            GRID,2,,1.,0.,0.
            GRID,3,,1.,1.,0.
            GRID,4,,0.,1.,0.
            GRID,5,,0.,0.,1.
            GRID,6,,1.,0.,1.
            GRID,7,,1.,1.,1.
            GRID,8,,0.,1.,1.
            CHEXA,1,101,1,2,3,4,5,6,7,
            ,8
            ENDDATA
            """))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            deck, entities = inspect_bdf(path)
        assert 101 in deck.elements and entities == {}
        assert "could not read the mesh" in capsys.readouterr().out

    def test_inspect_prints_the_tags(self, deck_path, capsys):
        deck, entities = inspect_bdf(deck_path)
        assert set(entities) == {PID_FLUID, PID_PLATE}
        assert entities[PID_FLUID].dim == 3
        text = capsys.readouterr().out
        assert f"tag {PID_FLUID}" in text and "looks like fluid" in text


# From the deck to a pyCAFE mesh

class TestBdfToMesh:

    def test_property_ids_become_groups(self, deck_path, tmp_path):
        out, report = bdf_to_mesh(
            deck_path, tmp_path / "cavity.msh",
            fluid=[PID_FLUID], structure=[PID_PLATE],
        )
        assert report.ok
        assert report.roles == {"fluid": "fluid", "structure": "plate",
                                "clamp": "plate_clamp"}

        _nodes, _elements, _boundaries, groups = load_mesh_with_groups(
            str(out), verbose=False
        )
        assert groups["fluid"]["elements"]["Hexahedron 8"].shape == (8, 8)
        assert groups["plate"]["elements"]["Quadrilateral 4"].shape == (4, 4)

    def test_free_faces_become_named_boundaries(self, deck_path, tmp_path):
        out, report = bdf_to_mesh(
            deck_path, tmp_path / "cavity.msh", fluid=[PID_FLUID],
            structure=[PID_PLATE],
            surface_rules=[on_plane("z", LZ, name="opening")],
        )
        _n, _e, _b, groups = load_mesh_with_groups(str(out), verbose=False)
        # 6 faces of 4 quadrilaterals; the z = 0 one is the shell, so it
        # is not offered again, and 4 go to the opening.
        assert groups["opening"]["elements"]["Quadrilateral 4"].shape == (4, 4)
        assert groups["rigid_walls"]["elements"]["Quadrilateral 4"].shape \
            == (16, 4)
        assert "conforming" in " ".join(report.notes)

    def test_faces_are_wound_outwards(self, fluid_deck_path, tmp_path):
        out, _report = bdf_to_mesh(
            fluid_deck_path, tmp_path / "cavity.msh", fluid=[PID_FLUID],
            surface_rules=[on_plane("z", LZ, name="opening")],
            clamp=None,
        )
        nodes, _e, _b, groups = load_mesh_with_groups(str(out), verbose=False)
        conn = groups["opening"]["elements"]["Quadrilateral 4"] - 1
        for row in conn:
            x = nodes[row]
            normal = np.cross(x[1] - x[0], x[2] - x[0])
            assert normal[2] > 0.0          # out of the fluid, upwards

    def test_walls_can_be_left_out(self, fluid_deck_path, tmp_path):
        out, _report = bdf_to_mesh(
            fluid_deck_path, tmp_path / "cavity.msh", fluid=[PID_FLUID],
            walls=None, clamp=None,
        )
        _n, _e, _b, groups = load_mesh_with_groups(str(out), verbose=False)
        assert "rigid_walls" not in groups

    def test_constrained_grids_become_the_support(self, deck_path, tmp_path):
        out, _report = bdf_to_mesh(
            deck_path, tmp_path / "cavity.msh",
            fluid=[PID_FLUID], structure=[PID_PLATE],
        )
        _n, _e, _b, groups = load_mesh_with_groups(str(out), verbose=False)
        # The edge of a 3 x 3 face: every node but the middle one.
        assert len(groups["plate_clamp"]["nodes"]) == 8

    def test_no_support_without_a_structure(self, fluid_deck_path, tmp_path):
        out, _report = bdf_to_mesh(
            fluid_deck_path, tmp_path / "acoustic.msh", fluid=[PID_FLUID],
            walls="rigid_walls",
        )
        _n, _e, _b, groups = load_mesh_with_groups(str(out), verbose=False)
        assert "plate_clamp" not in groups

    def test_unknown_property_id_is_refused(self, deck_path, tmp_path):
        with pytest.raises(ValueError, match="no dim 3 element"):
            bdf_to_mesh(deck_path, tmp_path / "x.msh", fluid=[999])

    def test_no_role_at_all_is_refused(self, deck_path, tmp_path):
        with pytest.raises(ValueError, match="carries no roles"):
            bdf_to_mesh(deck_path, tmp_path / "x.msh")

    def test_units_are_applied(self, fluid_deck_path, tmp_path):
        out, _report = bdf_to_mesh(
            fluid_deck_path, tmp_path / "mm.msh", fluid=[PID_FLUID],
            units="mm", clamp=None,
        )
        nodes, _e, _b, _g = load_mesh_with_groups(str(out), verbose=False)
        assert nodes[:, 0].max() == pytest.approx(LX * 1e-3)

    def test_an_unnamed_surface_property_is_reported(self, deck_path,
                                                     tmp_path):
        with pytest.warns(RuntimeWarning, match="given no name"):
            bdf_to_mesh(deck_path, tmp_path / "cavity.msh",
                        fluid=[PID_FLUID], clamp=None)


# The deck as a model input

class TestNastranFile:

    def test_roles_are_required(self, deck_path):
        with pytest.raises(ValueError, match="carries no roles"):
            NastranFile(deck_path)

    def test_mesh_path_is_derived(self, deck_path, tmp_path):
        spec = ModelSpec(
            geometry=NastranFile(deck_path, fluid=[PID_FLUID]),
            f_max=200.0, work_dir=tmp_path,
        )
        assert spec.mesh_path == tmp_path / "cavity.msh"

    def test_build_mesh_validates(self, deck_path, tmp_path):
        spec = ModelSpec(
            geometry=NastranFile(deck_path, fluid=[PID_FLUID],
                                 structure=[PID_PLATE]),
            f_max=200.0, work_dir=tmp_path,
        )
        path, report = build_mesh(spec)
        assert path.exists() and report.ok
        assert report.analysis == "vibroacoustic"

    def test_materials_come_from_the_deck(self, deck_path, tmp_path):
        spec = ModelSpec(
            geometry=NastranFile(deck_path, fluid=[PID_FLUID],
                                 structure=[PID_PLATE]),
            f_max=200.0, work_dir=tmp_path,
        )
        fluid, structure, notes = materials_from_deck(spec)
        assert (fluid.rho0, fluid.c0) == (1.204, 343.0)
        assert structure.t == pytest.approx(2e-3)
        assert (structure.E, structure.nu, structure.rho_s) \
            == (70e9, 0.33, 2700.0)
        assert len(notes) == 2

    def test_the_spec_wins_over_the_deck(self, deck_path, tmp_path):
        spec = ModelSpec(
            geometry=NastranFile(deck_path, fluid=[PID_FLUID],
                                 structure=[PID_PLATE]),
            f_max=200.0, work_dir=tmp_path,
            structure=aluminium(t=5e-3),
        )
        _fluid, structure, _notes = materials_from_deck(spec)
        assert structure.t == pytest.approx(5e-3)

    def test_a_disagreeing_fluid_is_reported(self, deck_path, tmp_path):
        from pycafe.core.model_spec import WATER

        spec = ModelSpec(
            geometry=NastranFile(deck_path, fluid=[PID_FLUID]),
            f_max=200.0, work_dir=tmp_path, fluid=WATER,
        )
        with pytest.warns(RuntimeWarning, match="The spec is used"):
            fluid, _structure, _notes = materials_from_deck(spec)
        assert fluid is WATER

    def test_build_model_assembles_the_coupled_system(self, deck_path,
                                                      tmp_path):
        spec = ModelSpec(
            geometry=NastranFile(
                deck_path, fluid=[PID_FLUID], structure=[PID_PLATE],
                surface_rules=[on_plane("z", LZ, name="opening")],
            ),
            f_max=200.0, work_dir=tmp_path,
        )
        model = build_model(spec, show=False)
        assert model["analysis"] == "vibroacoustic"
        assert model["fluid"].rho0 == pytest.approx(1.204)
        assert model["structure"].t == pytest.approx(2e-3)

        n = model["nodes"].shape[0]
        assert model["system"]["acoustic"]["K"].shape == (n, n)
        assert len(model["system"]["structural"]["clamped_nodes0"]) == 8
        assert model["system"]["coupling"]["Kc"].shape == (6 * n, n)

    def test_the_deck_is_read_once(self, deck_path):
        source = NastranFile(deck_path, fluid=[PID_FLUID])
        assert source.deck is source.deck

    def test_an_acoustic_deck_assembles_with_a_boundary_condition(
        self, deck_path, tmp_path
    ):
        from pycafe.boundary_condition.acoustic_bc import AcousticBC

        bc = AcousticBC()
        bc.add_impedance("opening", AIR.Z0)
        spec = ModelSpec(
            geometry=NastranFile(
                deck_path, fluid=[PID_FLUID],
                boundaries={"panel_face": [PID_PLATE]},
                surface_rules=[on_plane("z", LZ, name="opening")],
            ),
            f_max=200.0, work_dir=tmp_path, analysis="acoustic",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            model = build_model(spec, bc=bc, show=False)
        assert model["analysis"] == "acoustic"
        assert model["system"]["C"] is not None
