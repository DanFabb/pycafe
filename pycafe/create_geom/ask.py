r"""
Asking the user where a file is, the way MATLAB's ``uigetfile`` does.

A geometry pyCAFE builds itself is fully described by its parameters, so
a script can state it and be done. A geometry the user *brings* has one
piece of information no script can hold: where the file is on that
machine. Hard-coding the path is what every example used to do, and it
breaks as soon as the notebook is opened anywhere else.

:func:`ask_for_file` asks instead. It opens the native file dialog when
one can be opened, falls back to a typed path when it cannot, and
returns an absolute :class:`pathlib.Path` that exists:

.. code-block:: python

    from pycafe.create_geom import ask_for_file

    STEP = ask_for_file("cad", title="Pick the duct geometry")
    inspect_cad(STEP)

The dialog starts in ``Library/``, the directory where pyCAFE keeps both
the meshes it builds and the CAD files and decks it is given, and
filters on the extensions of the kind asked for.

**There is no fallback file.** No answer — a cancelled dialog, an empty
line, no display and no console — raises. Silently loading some other
geometry than the one meant is the one outcome worth ruling out, since
every number downstream would still look perfectly reasonable.

For a run with nobody at the keyboard (CI, a notebook executed by
``nbconvert``) the answer is given by the environment instead, one
variable per kind of file:

.. code-block:: bash

    export PYCAFE_FILE_GEOMETRY=Library/box_with_plate.msh
    export PYCAFE_FILE_CAD=Library/Tubo_1m_1m.stp
    export PYCAFE_FILE_NASTRAN=Library/demo_cavity.bdf
    export PYCAFE_FILE=...          # any kind, when one file is enough

That is a deliberate act by whoever starts the run, which is exactly
what a default argument is not.
"""

import os
import pathlib
import shutil
import subprocess
import sys

# What each kind of file is called and which extensions it covers. The
# order matters: the first entry is the filter the dialog opens on.
FILE_KINDS = {
    "geometry": ("Geometry (CAD, Nastran deck or mesh)",
                 ("*.step", "*.stp", "*.iges", "*.igs", "*.brep",
                  "*.bdf", "*.nas", "*.dat", "*.msh")),
    "cad": ("CAD geometry", ("*.step", "*.stp", "*.iges", "*.igs", "*.brep")),
    "mesh": ("Gmsh mesh", ("*.msh",)),
    "nastran": ("Nastran bulk data", ("*.bdf", "*.nas", "*.dat")),
    "any": ("All files", ("*",)),
}

# Where pyCAFE keeps the files a run can be pointed at: the meshes it
# builds itself and the CAD files and decks it is given, in one place, so
# that the dialog opens where everything is.
LIBRARY_DIRECTORY = "Library"


def _env_answer(kind):
    """The path given by the environment, for a run with nobody to ask."""
    for name in (f"PYCAFE_FILE_{kind.upper()}", "PYCAFE_FILE"):
        value = os.environ.get(name)
        if value:
            return value, name
    return None, None


def _default_directory(kind="geometry", start=None):
    """
    Where to open the dialog: the nearest ``Library/`` above the cwd.

    The examples run from ``examples/``, one level below it, so walking
    up finds it from either place. A tree without a ``Library`` opens on
    the working directory.
    """
    here = pathlib.Path(start or os.getcwd()).resolve()
    for folder in (here, *here.parents):
        candidate = folder / LIBRARY_DIRECTORY
        if candidate.is_dir():
            return candidate
    return here


def _filetypes(patterns, label):
    """
    The ``filetypes`` list, in the form every Tk build understands.

    Two things this gets right that the obvious version does not. The
    patterns go in as a **tuple**, not as one space-joined string: the
    macOS panel takes a string as a single extension and greys out every
    file in the folder. And each pattern is repeated in upper case,
    because the pattern match is case sensitive on X11 — a file saved as
    ``DUCT.STP`` is otherwise invisible.
    """
    extensions = []
    for pattern in patterns:
        extensions.append(pattern)
        if pattern != "*" and pattern.upper() != pattern:
            extensions.append(pattern.upper())
    return [(label, tuple(extensions)), ("All files", "*")]


def _osascript_available():
    """Is this a Mac with the AppleScript runner on it?"""
    return sys.platform == "darwin" and shutil.which("osascript") is not None


def _ask_with_osascript(title, initial_dir, patterns):
    """
    The macOS open panel, asked for through AppleScript.

    Tried before Tk on a Mac, and the reason is what Tk does inside a
    Jupyter kernel: the panel runs its own nested event loop, the kernel
    has no Tk loop to hand back to, and the cell can sit there with an
    empty ``tk`` window on screen after the file has been picked. The
    AppleScript panel is a separate process — it blocks until the user
    answers, prints the path, and leaves nothing behind.

    Returns the chosen path, ``""`` when the panel was cancelled, or
    None when AppleScript could not be used at all (so the caller falls
    back to Tk).
    """
    def literal(text):
        return '"' + str(text).replace("\\", "\\\\").replace('"', '\\"') + '"'

    # No "of type" clause. It filters by UTI, and a .msh or a .bdf has
    # none registered — the panel would then grey out the very file the
    # run needs, with no way to say so. The extensions go in the prompt
    # instead, and an answer that is none of them is refused later, by
    # name, in geometry_from_file.
    extensions = sorted({p.lstrip("*.").lower() for p in patterns
                         if p not in ("*", "*.*")})
    prompt = title if not extensions else f"{title}  ({', '.join(extensions)})"
    script = (
        'set chosen to choose file with prompt ' + literal(prompt)
        + ' default location POSIX file ' + literal(str(initial_dir))
        + '\nPOSIX path of chosen'
    )

    try:
        done = subprocess.run(["osascript", "-e", script],
                              capture_output=True, text=True)
    except Exception:
        return None

    if done.returncode == 0:
        return done.stdout.strip()
    # -128 is "the user cancelled"; anything else means the script itself
    # was refused, and Tk should get its turn.
    return "" if "-128" in (done.stderr or "") else None


def _close_root(root):
    """
    Take the empty ``tk`` window off the screen, and mean it.

    Destroying a Tk root only *schedules* the unmapping: the window is
    gone once the event loop runs again, and inside a Jupyter kernel or a
    plain script there is no loop left to run it, so the empty window
    stays painted until the process exits. Withdrawing it and pumping
    the queue once — before the interpreter is torn down — is what
    actually removes it. ``quit`` first, in case the native panel left a
    nested loop behind.
    """
    for step in (root.withdraw, root.update, root.quit, root.destroy):
        try:
            step()
        except Exception:
            pass


def _ask_with_dialog(title, initial_dir, patterns, label):
    """
    The native file dialog, or None if this machine cannot show one.

    Tk is in the standard library but needs a display; over SSH, in a
    container or in a plain batch run there is none, and the import or
    the window creation raises. That is not an error here — it is the
    signal to ask on the console instead.

    The panel is asked for **without** a ``parent``: on macOS a parent
    turns it into a sheet attached to that window, and the window it
    would hang from is the hidden root, which is not where anyone is
    looking. A root that was already there (Matplotlib's ``TkAgg``
    backend keeps one) is reused and left alone; only a root created
    here is closed here.

    Returns the chosen path, or ``""`` when the dialog was cancelled.
    """
    try:
        import tkinter
        from tkinter import filedialog
    except ImportError:
        return None

    root = getattr(tkinter, "_default_root", None)
    ours = root is None
    if ours:
        try:
            root = tkinter.Tk()
        except Exception:
            return None
        root.withdraw()

    try:
        # Keeps the panel above the terminal or the browser; cosmetic, so
        # a window manager that refuses is not a failure.
        try:
            root.call("wm", "attributes", ".", "-topmost", True)
        except Exception:
            pass
        chosen = filedialog.askopenfilename(
            title=title,
            initialdir=str(initial_dir),
            filetypes=_filetypes(patterns, label),
        )
    except Exception:
        return None
    finally:
        if ours:
            _close_root(root)

    return chosen or ""


def _ask_on_console(title, initial_dir, patterns):
    """The same question as text, for a machine with no display."""
    prompt = (f"{title}\n  looking in {initial_dir}"
              f"\n  ({' '.join(patterns)})\n  path")
    try:
        answer = input(f"{prompt}: ").strip()
    except (EOFError, KeyboardInterrupt, OSError):
        return None
    return answer.strip('"').strip("'")


def ask_for_file(kind="geometry", *, title=None, initial_dir=None,
                 must_exist=True):
    """
    Ask the user for a file and return its path.

    The equivalent of MATLAB's ``uigetfile``: the native dialog first, a
    typed path when there is no display, and an error — never a
    substitute file — when there is no answer.

    Parameters
    ----------
    kind : {"geometry", "cad", "mesh", "nastran", "any"}, optional
        Which extensions the dialog offers. Default ``"geometry"``, i.e.
        all three inputs pyCAFE reads.
    title : str, optional
        Text of the dialog. Default follows the kind.
    initial_dir : str or Path, optional
        Where the dialog opens. Default: the nearest ``Library/``
        directory above the working directory, else the working
        directory.
    must_exist : bool, optional
        Check that the answer names an existing file. Default True.

    Returns
    -------
    pathlib.Path
        Absolute path.

    Raises
    ------
    ValueError
        For an unknown ``kind``.
    FileNotFoundError
        If the answer names no file.
    RuntimeError
        If the dialog was cancelled, or nothing could be asked and the
        environment gives no answer either (see the module docstring for
        ``PYCAFE_FILE_*``).

    Examples
    --------
    >>> STEP = ask_for_file("cad")                     # doctest: +SKIP
    using /home/me/pycafe/Library/Tubo_1m_1m.stp
    """
    if kind not in FILE_KINDS:
        raise ValueError(
            f"kind must be one of {sorted(FILE_KINDS)}, not {kind!r}."
        )
    label, patterns = FILE_KINDS[kind]
    title = title or f"pyCAFE — select a {label.lower()} file"
    initial_dir = pathlib.Path(initial_dir) if initial_dir is not None \
        else _default_directory(kind)
    if not initial_dir.is_dir():
        initial_dir = pathlib.Path.cwd()

    chosen, source = _env_answer(kind)
    if chosen is None:
        # macOS first: its own panel is a separate process, so nothing is
        # left running (or painted) in the kernel afterwards.
        chosen = (_ask_with_osascript(title, initial_dir, patterns)
                  if _osascript_available() else None)
        if chosen is None:
            chosen = _ask_with_dialog(title, initial_dir, patterns, label)
        source = "the file dialog"
        if chosen is None:                      # no display: ask in text
            chosen = _ask_on_console(title, initial_dir, patterns)
            source = "the console"
            if chosen is None:
                raise RuntimeError(
                    f"{title}: nothing could be asked — no display for a "
                    "file dialog and no console to type in. Set "
                    f"PYCAFE_FILE_{kind.upper()}=<path> to answer from the "
                    "environment."
                )
        if not chosen:
            raise RuntimeError(
                f"{title}: no file selected. pyCAFE does not fall back on "
                "another geometry — pick one, or set "
                f"PYCAFE_FILE_{kind.upper()}=<path>."
            )

    path = pathlib.Path(chosen).expanduser()
    if not path.is_absolute():
        path = (pathlib.Path.cwd() / path).resolve()
    if must_exist and not path.is_file():
        raise FileNotFoundError(
            f"No file at {path} (answer taken from {source})."
        )
    print(f"using {path}")
    return path


def ask_for_geometry(*, title=None, initial_dir=None, fluid=None,
                     structure=None, verbose=True, **kwargs):
    """
    Ask where the geometry is, and read whatever kind it turns out to be.

    The opening move of every pyCAFE run that does not build its own
    geometry: one dialog, three possible answers, and the file itself
    says which one it is —

    * a **CAD** file (``.step``, ``.stp``, ``.iges``, ``.brep``) is
      meshed here, from the ``f_max`` of the spec it goes into;
    * a **Nastran deck** (``.bdf``, ``.nas``) is already meshed and is
      read as it stands, roles taken from its property cards;
    * a **mesh** (``.msh``) — one built by ``Library/``, or any other —
      is used as it is.

    Parameters
    ----------
    title : str, optional
    initial_dir : str or Path, optional
        Default: ``Library/``, which holds both the meshes pyCAFE builds
        and the CAD files and decks it is given.
    fluid, structure : sequence of int, optional
        Roles, when they are known: entity tags for a CAD file,
        property ids for a deck. Left out, they are guessed and the
        guess is printed.
    verbose : bool, optional
    **kwargs
        Passed on to the geometry source (``units``, ``boundaries``,
        ``recombine``, ``rename``, ...).

    Returns
    -------
    CadFile, NastranFile or MeshFile
        Ready to go into
        :class:`~pycafe.core.model_spec.ModelSpec`.

    Examples
    --------
    >>> geometry = ask_for_geometry()                    # doctest: +SKIP
    >>> spec = ModelSpec(geometry=geometry, f_max=500.0)  # doctest: +SKIP
    """
    from pycafe.core.model_spec import geometry_from_file

    path = ask_for_file(
        "geometry",
        title=title or "pyCAFE — select a geometry (CAD, Nastran deck or mesh)",
        initial_dir=initial_dir,
    )
    return geometry_from_file(path, fluid=fluid, structure=structure,
                              verbose=verbose, **kwargs)
