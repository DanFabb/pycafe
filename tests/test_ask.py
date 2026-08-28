"""
Tests for "where is the file?" — the dialog and what it dispatches to.

Covered:
- ``ask_for_file``: the kinds it knows, the directory it opens on in a
  pyCAFE tree, the typed path it accepts without a display, the
  ``PYCAFE_FILE_*`` answer a run with nobody at the keyboard gives, and
  the refusal to substitute a file when there is no answer at all;
- ``geometry_from_file``: extension -> geometry source, the roles read
  from a deck, the roles guessed on a CAD file, and the refusal to
  guess anything at all from an unknown extension;
- ``ask_for_geometry``: the two together, i.e. what every example calls.

No dialog is ever opened here: the Tk layer is monkeypatched, which is
also how the "no display" branch is exercised on a machine that has one.
"""

import pathlib

import pytest

from pycafe.create_geom import ask as ask_module
from pycafe.create_geom.ask import (
    FILE_KINDS,
    _default_directory,
    ask_for_file,
    ask_for_geometry,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
STEP = ROOT / "Library" / "WIP" / "Tubo_1m_1m.stp"
BDF = ROOT / "Library" / "WIP" / "demo_cavity.bdf"
MSH = ROOT / "Library" / "box_cavity.msh"


@pytest.fixture(autouse=True)
def no_mac_panel(monkeypatch):
    """
    Never open the real macOS panel.

    It is tried before Tk on a Mac, so a test that only silences Tk would
    otherwise sit there with an open panel waiting for a human.
    """
    monkeypatch.setattr(ask_module, "_osascript_available", lambda: False)


@pytest.fixture
def no_dialog(monkeypatch):
    """A machine with no display: the dialog layer declines to open."""
    monkeypatch.setattr(ask_module, "_ask_with_dialog",
                        lambda *a, **k: None)


@pytest.fixture
def cancelled(monkeypatch):
    """A machine with a display, and Cancel pressed in the dialog."""
    monkeypatch.setattr(ask_module, "_ask_with_dialog", lambda *a, **k: "")


class TestAskForFile:
    def test_every_kind_has_patterns(self):
        for kind, (label, patterns) in FILE_KINDS.items():
            assert label and patterns, kind
            assert all(p.startswith("*") for p in patterns), kind

    def test_unknown_kind_is_refused(self):
        with pytest.raises(ValueError, match="kind must be one of"):
            ask_for_file("solid_model")

    def test_the_dialog_opens_on_the_library(self):
        # One folder holds everything a run can be pointed at: the meshes
        # pyCAFE builds and the CAD files and decks it is given.
        for kind in ("cad", "nastran", "mesh", "geometry"):
            assert _default_directory(kind, start=ROOT) == ROOT / "Library"
        # Found from a subdirectory too, which is where the examples run.
        assert (_default_directory("geometry", start=ROOT / "examples")
                == ROOT / "Library")

    def test_patterns_go_in_as_a_tuple_and_in_both_cases(self):
        # A space-joined string greys out every file in the macOS panel,
        # and a lower-case-only pattern hides DUCT.STP on X11.
        types = ask_module._filetypes(("*.stp", "*.step"), "CAD geometry")
        assert types[0][0] == "CAD geometry"
        assert isinstance(types[0][1], tuple)
        assert "*.STP" in types[0][1] and "*.stp" in types[0][1]
        assert types[-1] == ("All files", "*")

    def test_typed_path_is_accepted_without_a_display(self, no_dialog,
                                                      monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _prompt: f' "{STEP}" ')
        assert ask_for_file("cad") == STEP

    def test_an_empty_answer_raises_rather_than_substituting_a_file(
            self, no_dialog, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _prompt: "")
        with pytest.raises(RuntimeError, match="no file selected"):
            ask_for_file("cad")

    def test_nothing_to_ask_raises(self, no_dialog, monkeypatch):
        def no_console(_prompt):
            raise EOFError

        monkeypatch.setattr("builtins.input", no_console)
        with pytest.raises(RuntimeError, match="nothing could be asked"):
            ask_for_file("cad")

    def test_the_environment_answers_for_a_run_with_nobody_at_the_keyboard(
            self, monkeypatch):
        def fail(*_a, **_k):
            raise AssertionError("the dialog must not be opened")

        monkeypatch.setattr(ask_module, "_ask_with_dialog", fail)
        monkeypatch.setenv("PYCAFE_FILE_CAD", str(STEP))
        assert ask_for_file("cad") == STEP
        # The kind-specific variable wins over the general one.
        monkeypatch.setenv("PYCAFE_FILE", str(MSH))
        assert ask_for_file("cad") == STEP
        monkeypatch.delenv("PYCAFE_FILE_CAD")
        assert ask_for_file("mesh") == MSH

    def test_cancel_raises_instead_of_guessing(self, cancelled):
        with pytest.raises(RuntimeError, match="no file selected"):
            ask_for_file("cad")

    def test_a_path_that_is_not_there_raises(self, no_dialog, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda _prompt: "nowhere.step")
        with pytest.raises(FileNotFoundError):
            ask_for_file("cad")


class TestTheMacPanel:
    """On a Mac the panel is a separate process, so Tk is never reached."""

    def test_the_script_names_the_folder_and_no_uti_filter(self, monkeypatch):
        seen = {}

        class Done:
            returncode = 0
            stdout = "/tmp/duct.step\n"
            stderr = ""

        def fake_run(cmd, **kwargs):
            seen["script"] = cmd[-1]
            return Done()

        monkeypatch.setattr(ask_module.subprocess, "run", fake_run)
        out = ask_module._ask_with_osascript("pick one", "/tmp/Library",
                                             ("*.msh", "*.step"))
        assert out == "/tmp/duct.step"
        assert "choose file with prompt" in seen["script"]
        assert "/tmp/Library" in seen["script"]
        # "of type" filters by UTI, which a .msh has none of: the panel
        # would grey out the file the run needs.
        assert "of type" not in seen["script"]
        assert "msh" in seen["script"] and "step" in seen["script"]

    def test_cancel_and_failure_are_told_apart(self, monkeypatch):
        class Done:
            def __init__(self, rc, err):
                self.returncode, self.stdout, self.stderr = rc, "", err

        monkeypatch.setattr(ask_module.subprocess, "run",
                            lambda *a, **k: Done(1, "User canceled. (-128)"))
        # Cancelled: an answer, and an empty one -> the caller raises.
        assert ask_module._ask_with_osascript("t", "/tmp", ("*.msh",)) == ""

        monkeypatch.setattr(ask_module.subprocess, "run",
                            lambda *a, **k: Done(1, "syntax error"))
        # Refused: no answer at all -> Tk gets its turn.
        assert ask_module._ask_with_osascript("t", "/tmp", ("*.msh",)) is None


class TestTheDialogWindow:
    """The empty ``tk`` window the panel hangs from must not survive it."""

    def test_a_root_we_made_is_closed_and_unmapped(self):
        tkinter = pytest.importorskip("tkinter")
        try:
            root = tkinter.Tk()
        except Exception:                      # no display on this machine
            pytest.skip("no display")
        root.deiconify()
        root.update()
        assert root.winfo_viewable()

        ask_module._close_root(root)
        # Destroying alone only schedules the unmapping; the window has to
        # be gone by the time the call returns, since no event loop is
        # left to run afterwards in a script or a Jupyter kernel. Once it
        # is, the interpreter behind it is gone too, so even asking is an
        # error — which is the check.
        with pytest.raises(tkinter.TclError):
            root.winfo_exists()

    def test_an_existing_root_is_reused_and_left_alive(self, monkeypatch):
        tkinter = pytest.importorskip("tkinter")
        filedialog = pytest.importorskip("tkinter.filedialog")
        try:
            existing = tkinter.Tk()
        except Exception:
            pytest.skip("no display")
        existing.withdraw()
        seen = {}

        def fake_dialog(**kwargs):
            seen.update(kwargs)
            return str(MSH)

        monkeypatch.setattr(filedialog, "askopenfilename", fake_dialog)
        chosen = ask_module._ask_with_dialog("t", MSH.parent, ("*.msh",),
                                             "Gmsh mesh")

        assert chosen == str(MSH)
        # Matplotlib's TkAgg backend keeps a root: killing it would take
        # the figure windows with it.
        assert existing.winfo_exists()
        # No parent: on macOS it turns the panel into a sheet attached to
        # the hidden root, which is not where anyone is looking.
        assert "parent" not in seen
        existing.destroy()


gmsh = pytest.importorskip("gmsh")

from pycafe.core.model_spec import (  # noqa: E402
    CadFile,
    MeshFile,
    NastranFile,
    geometry_from_file,
)


class TestGeometryFromFile:
    def test_mesh_is_used_as_it_is(self):
        source = geometry_from_file(MSH, verbose=False)
        assert isinstance(source, MeshFile)
        assert source.path == MSH

    def test_deck_roles_come_from_the_cards(self):
        source = geometry_from_file(BDF, verbose=False)
        assert isinstance(source, NastranFile)
        # PSOLID with FCTN = PFLUID is the fluid, PSHELL the structure.
        assert tuple(source.fluid) == (101,)
        assert tuple(source.structure) == (202,)

    def test_stated_roles_win_over_the_guess(self):
        source = geometry_from_file(BDF, fluid=[101], structure=[],
                                    verbose=False)
        assert tuple(source.structure) == ()

    def test_cad_falls_back_to_every_solid(self, capsys):
        source = geometry_from_file(STEP)
        assert isinstance(source, CadFile)
        assert tuple(source.fluid) == (1,)
        assert tuple(source.structure) == ()
        # A guess about which solid is the fluid must not pass silently.
        assert "no roles" in capsys.readouterr().out

    def test_unknown_extension_is_refused(self, tmp_path):
        odd = tmp_path / "geometry.sldprt"
        odd.write_text("")
        with pytest.raises(ValueError, match="not '.sldprt'"):
            geometry_from_file(odd)

    def test_missing_file_is_refused(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            geometry_from_file(tmp_path / "absent.step")


class TestAskForGeometry:
    @pytest.mark.parametrize("picked, expected", [
        (MSH, MeshFile),
        (BDF, NastranFile),
        (STEP, CadFile),
    ])
    def test_the_extension_picks_the_source(self, no_dialog, monkeypatch,
                                            picked, expected):
        monkeypatch.setattr("builtins.input", lambda _prompt: str(picked))
        source = ask_for_geometry(verbose=False)
        assert isinstance(source, expected)
        assert pathlib.Path(source.path) == picked
