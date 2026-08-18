"""
Tests for the standardized model input.

Covered:
- the sizing rule: the mesh follows from ``f_max`` and the element
  density, and a density below the practical minimum is refused;
- ``Library``: the element counts derived from the size actually reach
  the geometry, and an explicit count still wins;
- ``build_model``: a cavity assembles as an acoustic system, a cavity
  with a plate as a coupled one, and a mesh whose structural group has
  no material is refused instead of assembled;
- ``MeshFile``: an existing mesh is used as it is, and a foreign one is
  renamed on the way in;
- ``CadFile``: the roles asked for by tag end up in the written mesh;
- the geometry preview: every role in the mesh is drawn, and
  ``build_model`` draws it before assembling unless told not to.
"""

import pathlib

import pytest

gmsh = pytest.importorskip("gmsh")

from pycafe.core.model_spec import (  # noqa: E402
    AIR,
    DEFAULT_ELEMENTS_PER_WAVELENGTH,
    WATER,
    CadFile,
    Fluid,
    Library,
    MeshFile,
    ModelSpec,
    Structure,
    aluminium,
    build_mesh,
    build_model,
    describe_domains,
    element_size_for,
    steel,
)
from pycafe.create_geom import inspect_cad, list_groups  # noqa: E402


# sizing
def test_element_size_follows_from_frequency():
    spec = ModelSpec(geometry=Library("box_cavity"), f_max=500.0)
    assert spec.elements_per_wavelength == DEFAULT_ELEMENTS_PER_WAVELENGTH
    assert spec.element_size == pytest.approx(343.0 / (500.0 * 10))
    # Twice the frequency, half the element.
    assert spec.at(f_max=1000.0).element_size == pytest.approx(
        spec.element_size / 2
    )


def test_size_uses_the_fluid_of_the_spec():
    in_water = ModelSpec(geometry=Library("box_cavity"), f_max=500.0,
                         fluid=WATER)
    assert in_water.element_size == pytest.approx(1480.0 / (500.0 * 10))
    assert element_size_for(WATER.c0, 500.0, 10) == in_water.element_size


def test_too_coarse_is_refused_and_slightly_coarse_warns():
    with pytest.raises(ValueError, match="below the practical minimum"):
        ModelSpec(geometry=Library("box_cavity"), f_max=500.0,
                  elements_per_wavelength=4)
    with pytest.warns(RuntimeWarning, match="per wavelength"):
        ModelSpec(geometry=Library("box_cavity"), f_max=500.0,
                  elements_per_wavelength=8)


def test_bad_frequency_and_bad_geometry_name():
    with pytest.raises(ValueError, match="f_max must be positive"):
        ModelSpec(geometry=Library("box_cavity"), f_max=0.0)
    with pytest.raises(ValueError, match="Unknown geometry"):
        Library("cavity_shaped_like_a_duck")


def test_describe_mentions_the_size_and_the_bending_wavelength():
    spec = ModelSpec(geometry=Library("box_with_plate"), f_max=400.0,
                     structure=aluminium(t=2e-3))
    text = spec.describe()
    assert "mesh size" in text and "elements per wavelength" in text
    assert "lambda_b" in text
    # A 2 mm aluminium panel at 400 Hz is well below coincidence, so its
    # bending wave is the shorter one and the panel mesh needs a look.
    assert spec.structure.bending_wavelength(400.0) < AIR.wavelength(400.0)


def test_material_helpers():
    assert steel(1e-3).E == 210e9
    assert aluminium(2e-3, support="simply_supported").support == \
        "simply_supported"
    plate_ = Structure(t=1e-3, E=200e9, nu=0.3, rho_s=7800.0)
    assert plate_.D == pytest.approx(200e9 * 1e-9 / (12 * (1 - 0.09)))
    assert Fluid(1.2, 340.0).Z0 == pytest.approx(408.0)


# library geometries
def test_library_counts_come_from_the_size(tmp_path):
    spec = ModelSpec(
        geometry=Library("box_cavity", Lx=0.4, Ly=0.3, Lz=0.5),
        f_max=343.0,                       # lambda = 1 m, h = 0.1 m
        work_dir=tmp_path,
    )
    assert spec.element_size == pytest.approx(0.1)
    path, report = build_mesh(spec)

    assert path.exists() and report.ok
    nodes, elements, _b, groups = report_mesh(path)
    # 4 x 3 x 5 hexahedra from ceil(L / h).
    assert groups["fluid"]["elements"]["Hexahedron 8"].shape[0] == 4 * 3 * 5


def test_explicit_count_overrides_the_derived_one(tmp_path):
    spec = ModelSpec(
        # Finer than the rule asks for, which is what a convergence
        # study does: 8 x 6 x 10 instead of the derived 4 x 3 x 5.
        geometry=Library("box_cavity", Lx=0.4, Ly=0.3, Lz=0.5, nx=8, ny=6,
                         nz=10),
        f_max=343.0,
        work_dir=tmp_path,
    )
    path, _report = build_mesh(spec)
    _n, _e, _b, groups = report_mesh(path)
    assert groups["fluid"]["elements"]["Hexahedron 8"].shape[0] == 8 * 6 * 10


def test_acoustic_model_assembles(tmp_path):
    spec = ModelSpec(
        geometry=Library("box_cavity", Lx=0.4, Ly=0.3, Lz=0.5),
        f_max=343.0,
        work_dir=tmp_path,
    )
    model = build_model(spec, show=False)

    assert model["analysis"] == "acoustic"
    n = model["nodes"].shape[0]
    assert model["system"]["K"].shape[0] <= n
    assert model["mesh_path"].exists()


def test_vibroacoustic_model_assembles_and_couples(tmp_path):
    spec = ModelSpec(
        geometry=Library("box_with_plate", Lx=0.4, Ly=0.3, Lz=0.5),
        f_max=343.0,
        structure=aluminium(t=2e-3),
        work_dir=tmp_path,
    )
    model = build_model(spec, show=False)

    assert model["analysis"] == "vibroacoustic"
    system = model["system"]
    assert system["coupling"] is not None
    assert system["props"]["E"] == 70e9
    assert system["props"]["c0"] == AIR.c0
    assert system["structural"]["support"] == "clamped"


def test_structural_group_without_material_is_refused(tmp_path):
    spec = ModelSpec(
        geometry=Library("box_with_plate", Lx=0.4, Ly=0.3, Lz=0.5),
        f_max=343.0,
        work_dir=tmp_path,
    )
    with pytest.raises(ValueError, match="no Structure"):
        build_model(spec, show=False)


# meshes brought in
def test_mesh_file_is_used_as_it_is(tmp_path):
    from pycafe.create_geom import box_cavity

    existing = box_cavity(Lx=0.4, Ly=0.3, Lz=0.5, nx=4, ny=3, nz=5,
                          output_path=tmp_path / "given.msh")
    spec = ModelSpec(geometry=MeshFile(existing), f_max=343.0,
                     work_dir=tmp_path)
    path, report = build_mesh(spec)

    assert path == pathlib.Path(existing)
    assert report.ok and report.analysis == "acoustic"


def test_mesh_file_renames_foreign_groups(tmp_path):
    from pycafe.create_geom import box_cavity, retag_mesh

    original = box_cavity(Lx=0.4, Ly=0.3, Lz=0.5, nx=4, ny=3, nz=5,
                          face_groups=False,
                          output_path=tmp_path / "original.msh")
    foreign = retag_mesh(original, rename={"fluid": "AIR"},
                         output_path=tmp_path / "foreign.msh",
                         validate=False)

    spec = ModelSpec(
        geometry=MeshFile(foreign, rename={"AIR": "fluid"}),
        f_max=343.0,
        output_path=tmp_path / "fixed.msh",
    )
    path, report = build_mesh(spec)

    assert path == tmp_path / "fixed.msh"
    assert report.roles["fluid"] == "fluid"


def test_retagging_never_overwrites_the_mesh_given(tmp_path):
    """The mesh the user brought is an input: retagging writes elsewhere."""
    from pycafe.create_geom import box_cavity, list_groups, retag_mesh

    original = box_cavity(Lx=0.4, Ly=0.3, Lz=0.5, nx=4, ny=3, nz=5,
                          face_groups=False,
                          output_path=tmp_path / "original.msh")
    foreign = pathlib.Path(retag_mesh(original, rename={"fluid": "AIR"},
                                      output_path=tmp_path / "foreign.msh",
                                      validate=False))

    # work_dir is where the mesh lies, and no output_path is given: the
    # naive default would land back on foreign.msh.
    spec = ModelSpec(
        geometry=MeshFile(foreign, rename={"AIR": "fluid"}),
        f_max=343.0,
        work_dir=tmp_path,
    )
    path, report = build_mesh(spec)

    assert path != foreign
    assert report.roles["fluid"] == "fluid"
    # the original still carries its own name, untouched
    assert "AIR" in list_groups(foreign, verbose=False)


def test_mesh_that_breaks_the_contract_is_refused(tmp_path):
    from pycafe.create_geom import box_cavity, retag_mesh

    original = box_cavity(Lx=0.4, Ly=0.3, Lz=0.5, nx=4, ny=3, nz=5,
                          face_groups=False,
                          output_path=tmp_path / "original.msh")
    nameless = retag_mesh(original, drop=["fluid"],
                          output_path=tmp_path / "nameless.msh",
                          validate=False)

    spec = ModelSpec(geometry=MeshFile(nameless), f_max=343.0,
                     work_dir=tmp_path)
    with pytest.raises(RuntimeError, match="contract"):
        build_mesh(spec)


# CAD
@pytest.fixture
def step_box(tmp_path):
    """A 0.4 x 0.3 x 0.5 m box written as a STEP file, in millimetres."""
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("box")
        gmsh.model.occ.addBox(0, 0, 0, 400, 300, 500)
        gmsh.model.occ.synchronize()
        path = tmp_path / "box.step"
        gmsh.write(str(path))
    finally:
        gmsh.finalize()
    return path


def test_inspect_cad_reports_entities_in_metres(step_box):
    found = inspect_cad(step_box, verbose=False)

    assert len(found[3]) == 1
    volume = found[3][0]
    assert volume.size == pytest.approx(0.4 * 0.3 * 0.5, rel=1e-6)
    assert len(found[2]) == 6


def test_cad_file_tags_the_roles_it_is_given(step_box, tmp_path):
    found = inspect_cad(step_box, verbose=False)
    fluid_tag = found[3][0].tag
    # The face at z = 0.5 m is the panel.
    top = max(found[2], key=lambda i: i.com[2])

    spec = ModelSpec(
        geometry=CadFile(step_box, fluid=[fluid_tag], structure=[top.tag]),
        f_max=343.0,
        structure=aluminium(t=2e-3),
        work_dir=tmp_path,
    )
    model = build_model(spec, show=False)

    assert model["mesh_path"] == tmp_path / "box.msh"
    groups = list_groups(model["mesh_path"], verbose=False)
    assert groups["fluid"]["dim"] == 3
    assert groups["plate"]["dim"] == 2
    assert groups["rigid_walls"]["dim"] == 2
    assert groups["plate_clamp"]["dim"] == 1
    # A box is meshed structurally, so the panel quadrilaterals are
    # faces of the fluid hexahedra and the coupling can be assembled.
    assert "Hexahedron 8" in model["elements"]
    assert model["system"]["coupling"] is not None


VIBRO_STEP = pathlib.Path(__file__).resolve().parent.parent / "Library" \
    / "Vibro_ac1.step"


@pytest.mark.skipif(not VIBRO_STEP.exists(),
                    reason="Library/Vibro_ac1.step is not in the checkout")
def test_the_solid_with_no_role_is_not_meshed(tmp_path):
    """
    A body pyCAFE models by its surface must not be meshed inside.

    ``Library/Vibro_ac1.step`` is the shape of a STEP assembly: a sphere of
    fluid with a cylinder cut out of it, and the cylinder itself, each
    solid carrying its own copy of the surface where they touch. Only
    the sphere is declared here, so the cylinder has no role at all:
    meshing its interior would put nodes in the file with no element to
    give them an equation, and the acoustic matrices would be singular
    before any solve. Its private copy of the contact surface has to go
    with it, for the same reason.
    """
    import numpy as np

    spec = ModelSpec(
        geometry=CadFile(VIBRO_STEP, units="mm", fluid=[1]),
        f_max=1000.0,
        work_dir=tmp_path,
    )
    model = build_model(spec, show=False)

    used = set()
    for conn in model["elements"].values():
        used |= set(np.asarray(conn).ravel().tolist())
    orphans = set(range(1, model["nodes"].shape[0] + 1)) - used
    assert not orphans

    # And the assembled system says the same thing, on the fluid alone.
    covered = np.diff(model["system"]["K"].tocsr().indptr) > 0
    assert covered.all()


def test_cad_units_are_detected(step_box, tmp_path):
    """The STEP box is written in millimetres and must come back in metres."""
    spec = ModelSpec(
        geometry=CadFile(step_box, fluid=[1]),
        f_max=343.0,
        work_dir=tmp_path,
    )
    model = build_model(spec, show=False)
    extent = model["nodes"].max(axis=0) - model["nodes"].min(axis=0)
    assert extent == pytest.approx([0.4, 0.3, 0.5])


def test_cad_file_without_fluid_tags_is_refused(step_box):
    with pytest.raises(ValueError, match="tags of the fluid"):
        CadFile(step_box)


def test_missing_files_are_reported_at_once(tmp_path):
    with pytest.raises(FileNotFoundError):
        CadFile(tmp_path / "nowhere.step", fluid=[1])
    with pytest.raises(FileNotFoundError):
        MeshFile(tmp_path / "nowhere.msh")


# looking at the geometry first
def test_preview_draws_every_role(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    from pycafe.core.model_spec import preview

    spec = ModelSpec(
        geometry=Library("box_with_plate", Lx=0.4, Ly=0.3, Lz=0.5),
        f_max=343.0,
        structure=aluminium(t=2e-3),
        work_dir=tmp_path,
    )
    path, report = preview(spec, show=False)

    assert path.exists() and report.ok
    ax = matplotlib.pyplot.gca()
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    # fluid, plate and the clamp are the roles; rigid_walls is a free name.
    assert any("fluid" in t for t in labels)
    assert any("structure" in t for t in labels)
    assert any("clamp" in t for t in labels)
    assert any("rigid_walls" in t for t in labels)
    matplotlib.pyplot.close("all")


def test_build_model_shows_the_geometry_before_assembling(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.close("all")
    spec = ModelSpec(
        geometry=Library("box_cavity", Lx=0.4, Ly=0.3, Lz=0.5),
        f_max=343.0,
        work_dir=tmp_path,
    )
    build_model(spec, plot_kwargs={"show": False})
    assert plt.get_fignums(), "build_model must draw the geometry by default"

    plt.close("all")
    build_model(spec, show=False)
    assert not plt.get_fignums(), "show=False must draw nothing"


def test_plot_geometry_accepts_a_2d_mesh(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pycafe.create_geom import duct_2d, load_mesh_with_groups, plot_geometry

    path = duct_2d(L=1.0, H=0.1, nx=10, ny=2,
                   output_path=tmp_path / "duct.msh")
    nodes, _elements, _boundaries, groups = load_mesh_with_groups(
        str(path), verbose=False
    )
    ax = plot_geometry(nodes, groups, show=False)

    assert ax.get_legend() is not None
    plt.close("all")


# domains and their materials
class TestDescribeDomains:
    """Which group is which domain, and what fills it."""

    def _report(self, tmp_path, analysis, structure):
        spec = ModelSpec(
            geometry=Library("box_with_plate", Lx=0.4, Ly=0.3, Lz=0.5,
                             nx=4, ny=3, nz=4),
            fluid=AIR, structure=structure, f_max=200.0,
            analysis=analysis, work_dir=tmp_path,
        )
        _path, report = build_mesh(spec)
        return report

    def test_each_domain_is_paired_with_its_material(self, tmp_path):
        report = self._report(tmp_path, "vibroacoustic",
                              aluminium(t=2e-3, support="clamped"))
        text = describe_domains(report, AIR, aluminium(t=2e-3,
                                                       support="clamped"))
        assert "fluid" in text and "'fluid'" in text and "air" in text
        assert "structure" in text and "'plate'" in text
        assert "2.00 mm" in text
        # The mesh says which nodes are held, the material says how.
        assert "'plate_clamp'" in text and "clamped" in text

    def test_a_missing_material_is_named(self, tmp_path):
        report = self._report(tmp_path, "vibroacoustic", None)
        text = describe_domains(report, AIR, None)
        assert "no structure given" in text

    def test_an_acoustic_run_has_one_domain(self, tmp_path):
        spec = ModelSpec(geometry=Library("box_cavity", nx=4, ny=3, nz=4),
                         fluid=AIR, f_max=200.0, analysis="acoustic",
                         work_dir=tmp_path)
        _path, report = build_mesh(spec)
        text = describe_domains(report, AIR, None)
        assert "structure" not in text

    def test_vibroacoustic_on_a_fluid_only_mesh_is_refused(self, tmp_path):
        # The pairing cannot even be reached: validate_mesh stops it.
        spec = ModelSpec(geometry=Library("box_cavity", nx=4, ny=3, nz=4),
                         fluid=AIR, structure=aluminium(t=2e-3), f_max=200.0,
                         analysis="vibroacoustic", work_dir=tmp_path)
        with pytest.raises(RuntimeError, match="no structure domain"):
            build_mesh(spec)


# helpers
def report_mesh(path):
    from pycafe.create_geom import load_mesh_with_groups

    return load_mesh_with_groups(str(path), verbose=False)
