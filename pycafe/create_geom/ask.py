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
    export PYCAFE_FILE_CAD=Library/WIP/Tubo_1m_1m.stp
    export PYCAFE_FILE_NASTRAN=Library/WIP/demo_cavity.bdf
    export PYCAFE_FILE=...          # any kind, when one file is enough

That is a deliberate act by whoever starts the run, which is exactly
what a default argument is not.

The same three-way contract covers the questions that are not paths —
:func:`ask_for_choice`, :func:`ask_for_number` and
:func:`ask_for_structure`, for which physical group plays which role and
what the structure is made of. Their environment answers are named after
the question:

.. code-block:: bash

    export PYCAFE_ANSWER_STRUCTURE_GROUP=plate
    export PYCAFE_ANSWER_MATERIAL=aluminium
    export PYCAFE_ANSWER_THICKNESS=1.0        # mm
    export PYCAFE_ANSWER_SUPPORT=clamped
"""

import os
import pathlib
import shutil
import subprocess
import sys

# What each kind of file is called and which extensions it covers. The
# order matters: the first entry is the filter the dialog opens on.
FILE_KINDS = {
    # The first release reads meshes, and only meshes: a model pyCAFE
    # can be pointed at is one gmsh has already cut. Reading a CAD file
    # or a Nastran deck is written and works (``CadFile``,
    # ``NastranFile``), but it is not part of this release and the
    # dialog does not offer it; those inputs live in ``Library/WIP``.
    "geometry": ("Gmsh mesh", ("*.msh",)),
    "cad": ("CAD geometry", ("*.step", "*.stp", "*.iges", "*.igs", "*.brep")),
    "mesh": ("Gmsh mesh", ("*.msh",)),
    "nastran": ("Nastran bulk data", ("*.bdf", "*.nas", "*.dat")),
    "any": ("All files", ("*",)),
}

# The prefix of the environment variables that answer a question when
# there is nobody at the keyboard: PYCAFE_ANSWER_<NAME>.
ANSWER_PREFIX = "PYCAFE_ANSWER"

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
    using /home/me/pycafe/Library/WIP/Tubo_1m_1m.stp
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
    geometry: one dialog on ``Library/``, and the mesh it answers with.

    **This release reads gmsh meshes only.** A ``.msh`` carries its own
    element sizes and its own physical groups, so loading it is the last
    question about the geometry: nothing is remeshed and nothing is
    guessed. Meshing a CAD file and reading a Nastran deck are written
    (:class:`~pycafe.core.model_spec.CadFile`,
    :class:`~pycafe.core.model_spec.NastranFile`) and still work when
    called directly, but they are not part of this release: the dialog
    does not offer them, and the files they would read live in
    ``Library/WIP``.

    Parameters
    ----------
    title : str, optional
    initial_dir : str or Path, optional
        Default: ``Library/``, which holds the meshes.
    fluid, structure : sequence of int, optional
        Roles, when they are known. Meshes carry their own, so this is
        for the WIP inputs only.
    verbose : bool, optional
    **kwargs
        Passed on to the geometry source (``units``, ``boundaries``,
        ``recombine``, ``rename``, ...).

    Returns
    -------
    MeshFile
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
        title=title or "pyCAFE — select a gmsh mesh (.msh)",
        initial_dir=initial_dir,
    )
    return geometry_from_file(path, fluid=fluid, structure=structure,
                              verbose=verbose, **kwargs)


def _env_value(name):
    """The answer given by the environment, for a run with nobody to ask."""
    for variable in (f"{ANSWER_PREFIX}_{name.upper()}", ANSWER_PREFIX):
        value = os.environ.get(variable)
        if value:
            return value, variable
    return None, None


def _choose_with_osascript(question, options, default=None):
    """
    The macOS list panel, asked for through AppleScript.

    Same reason as the file panel: a separate process, so nothing is
    left running inside a Jupyter kernel. Returns the chosen item,
    ``""`` when the panel was cancelled, or None when AppleScript could
    not be used at all.
    """
    def literal(text):
        return '"' + str(text).replace("\\", "\\\\").replace('"', '\\"') + '"'

    items = "{" + ", ".join(literal(o) for o in options) + "}"
    script = (
        "set chosen to choose from list " + items
        + " with prompt " + literal(question)
        + " default items {" + literal(default or options[0]) + "}"
        + "\nif chosen is false then return \"\"\nitem 1 of chosen"
    )
    try:
        done = subprocess.run(["osascript", "-e", script],
                              capture_output=True, text=True)
    except Exception:
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def _ask_text_with_osascript(question, default=None):
    """The macOS text dialog. Same contract as :func:`_choose_with_osascript`."""
    def literal(text):
        return '"' + str(text).replace("\\", "\\\\").replace('"', '\\"') + '"'

    script = (
        "set answer to display dialog " + literal(question)
        + " default answer " + literal("" if default is None else default)
        + "\ntext returned of answer"
    )
    try:
        done = subprocess.run(["osascript", "-e", script],
                              capture_output=True, text=True)
    except Exception:
        return None
    if done.returncode == 0:
        return done.stdout.strip()
    return "" if "-128" in (done.stderr or "") else None


def _ask_text_with_dialog(question, default=None):
    """The Tk text dialog, or None if this machine cannot show one."""
    try:
        import tkinter
        from tkinter import simpledialog
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
        answer = simpledialog.askstring("pyCAFE", question,
                                        initialvalue=default or "")
    except Exception:
        return None
    finally:
        if ours:
            _close_root(root)
    return "" if answer is None else answer.strip()


def _ask_text_on_console(question, default=None):
    """The same question as text, for a machine with no display."""
    suffix = "" if default is None else f" [{default}]"
    try:
        answer = input(f"{question}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt, OSError):
        return None
    return answer or ("" if default is None else str(default))


def ask_for_choice(question, options, *, name="choice", default=None):
    """
    Ask the user to pick one of a few answers, and return it.

    The counterpart of :func:`ask_for_file` for everything that is not a
    path: which physical group plays which role, which material, how an
    edge is held. Same contract throughout — the native panel when one
    can be shown, a typed answer when it cannot, the environment when
    there is nobody at the keyboard, and an error rather than a silent
    default when there is no answer at all.

    Parameters
    ----------
    question : str
        What is being asked, in full: it is the whole of the prompt.
    options : sequence of str
        The answers offered. An answer outside the list is refused.
    name : str, optional
        Short name of the question. It is what the environment variable
        is called, ``PYCAFE_ANSWER_<NAME>``.
    default : str, optional
        Which option the dialog opens on. It is *not* a fallback: an
        unanswered question still raises.

    Returns
    -------
    str
        One of ``options``.

    Raises
    ------
    ValueError
        For an empty ``options``, or a ``default`` outside it.
    RuntimeError
        If the question was cancelled or could not be asked.

    Examples
    --------
    >>> face = ask_for_choice("Which surface is the plate?",       # doctest: +SKIP
    ...                       ["plate", "rigid_walls"],
    ...                       name="structure_group")
    """
    options = [str(o) for o in options]
    if not options:
        raise ValueError(f"{question}: no options to choose from.")
    if default is not None and str(default) not in options:
        raise ValueError(
            f"{question}: default {default!r} is not one of {options}."
        )

    chosen, source = _env_value(name)
    if chosen is None:
        prompt = f"pyCAFE — {question}"
        chosen = (_choose_with_osascript(prompt, options, default)
                  if _osascript_available() else None)
        if chosen is None:
            listing = ", ".join(options)
            chosen = _ask_text_with_dialog(f"{prompt}\n({listing})", default)
            source = "the dialog"
            if chosen is None:
                print(f"{prompt}")
                for i, option in enumerate(options, 1):
                    print(f"  {i}. {option}")
                chosen = _ask_text_on_console("  answer", default)
                source = "the console"
                if chosen is None:
                    raise RuntimeError(
                        f"{question}: nothing could be asked — no display "
                        "and no console. Set "
                        f"{ANSWER_PREFIX}_{name.upper()}=<answer> to answer "
                        "from the environment."
                    )
        else:
            source = "the dialog"
        if not chosen:
            raise RuntimeError(
                f"{question}: no answer. pyCAFE does not pick one for you — "
                f"answer, or set {ANSWER_PREFIX}_{name.upper()}=<answer>."
            )

    if chosen.isdigit() and 1 <= int(chosen) <= len(options):
        chosen = options[int(chosen) - 1]
    if chosen not in options:
        raise RuntimeError(
            f"{question}: {chosen!r} (from {source}) is not one of {options}."
        )
    print(f"{name}: {chosen}")
    return chosen


def ask_for_number(question, *, name="value", default=None, unit=None,
                   minimum=None, maximum=None):
    """
    Ask the user for one number, and return it as a float.

    Same contract as :func:`ask_for_choice`, with the range checked
    here rather than by the caller.

    Parameters
    ----------
    question : str
    name : str, optional
        Name of the environment variable, ``PYCAFE_ANSWER_<NAME>``.
    default : float, optional
        What the dialog opens on; not a fallback.
    unit : str, optional
        Added to the prompt, so that the number asked for is
        unambiguous.
    minimum, maximum : float, optional
        Bounds the answer has to be within.

    Returns
    -------
    float

    Raises
    ------
    RuntimeError
        If the question was cancelled, could not be asked, or the answer
        is not a number in range.
    """
    prompt = f"pyCAFE — {question}" + (f" [{unit}]" if unit else "")
    shown = None if default is None else f"{default:g}"

    answer, source = _env_value(name)
    if answer is None:
        answer = (_ask_text_with_osascript(prompt, shown)
                  if _osascript_available() else None)
        source = "the dialog"
        if answer is None:
            answer = _ask_text_with_dialog(prompt, shown)
            if answer is None:
                answer = _ask_text_on_console(prompt, shown)
                source = "the console"
                if answer is None:
                    raise RuntimeError(
                        f"{question}: nothing could be asked — no display "
                        "and no console. Set "
                        f"{ANSWER_PREFIX}_{name.upper()}=<number> to answer "
                        "from the environment."
                    )
        if not answer:
            raise RuntimeError(
                f"{question}: no answer. pyCAFE does not pick one for you — "
                f"answer, or set {ANSWER_PREFIX}_{name.upper()}=<number>."
            )

    try:
        value = float(str(answer).replace(",", "."))
    except ValueError:
        raise RuntimeError(
            f"{question}: {answer!r} (from {source}) is not a number."
        ) from None
    if minimum is not None and value < minimum:
        raise RuntimeError(f"{question}: {value:g} is below {minimum:g}.")
    if maximum is not None and value > maximum:
        raise RuntimeError(f"{question}: {value:g} is above {maximum:g}.")
    print(f"{name}: {value:g}" + (f" {unit}" if unit else ""))
    return value


def ask_for_structure(*, materials=("aluminium", "steel"),
                      supports=("clamped", "simply_supported", "free"),
                      default_material="aluminium", default_thickness=1.0,
                      default_support="clamped"):
    """
    Ask what the structural domain is made of, and how it is held.

    The mesh says which elements are structural; it does not say what
    they are made of, nor whether their edge is clamped or hinged. That
    is asked here, once the mesh is on the table, in the same way its
    path was.

    Parameters
    ----------
    materials : sequence of str, optional
        Which materials are offered. Each name must be a factory of
        :mod:`pycafe.core.model_spec` (``aluminium``, ``steel``).
    supports : sequence of str, optional
        Which supports are offered; see
        :func:`~pycafe.core.prepare_vibroacoustic_system.prepare_vibroacoustic_system`
        for what each one blocks.
    default_material, default_support : str, optional
    default_thickness : float, optional
        In **millimetres**, the unit the thickness is asked in.

    Returns
    -------
    Structure
        Ready to go into :class:`~pycafe.core.model_spec.ModelSpec`.

    Examples
    --------
    >>> structure = ask_for_structure()                # doctest: +SKIP
    material: aluminium
    thickness: 1 mm
    support: clamped
    """
    from pycafe.core import model_spec

    material = ask_for_choice("what is the structure made of?", materials,
                              name="material", default=default_material)
    thickness = ask_for_number("how thick is it?", name="thickness",
                               default=default_thickness, unit="mm",
                               minimum=0.0)
    support = ask_for_choice("how is its edge held?", supports,
                             name="support", default=default_support)
    factory = getattr(model_spec, material, None)
    if factory is None:
        raise RuntimeError(
            f"{material!r} is not a material of pycafe.core.model_spec; "
            f"the ones it defines are {sorted(materials)}."
        )
    return factory(t=thickness * 1e-3, support=support)


def ask_for_point(question, nodes, *, name="point", default=None,
                  default_is=None, candidates=None, unit="m"):
    """
    Ask where a point is, and return the nearest node of the mesh.

    Anything applied or measured at a point — a force on a panel, a
    microphone in a cavity — is a coordinate the user has in mind and a
    node the model actually has. This asks for the first and returns the
    second, printing how far apart they turned out to be.

    Parameters
    ----------
    question : str
        What the point is for, in full.
    nodes : ndarray (N, 3) or (N, 2)
        Mesh node coordinates.
    name : str, optional
        Name of the environment variable, ``PYCAFE_ANSWER_<NAME>``,
        answered as ``"0.4, 0.3, 0.5"``.
    default : array-like, optional
        Coordinates the dialog opens on — a sensible place to put it,
        not a fallback: an unanswered question still raises.
    default_is : str, optional
        What that place is ("the centre of the panel"), so that the
        number offered means something.
    candidates : array-like of int, optional
        0-based node indices the answer may snap to. Default: every
        node. Pass the panel nodes for a force on the panel, the fluid
        nodes for a microphone.
    unit : str, optional
        Added to the prompt.

    Returns
    -------
    node0 : int
        0-based index of the nearest node.
    coordinates : ndarray (3,)
        Where that node is.

    Raises
    ------
    RuntimeError
        If the question was cancelled, could not be asked, or the answer
        is not two or three numbers.

    Examples
    --------
    >>> node, xyz = ask_for_point("where is the force?", nodes,  # doctest: +SKIP
    ...                           name="drive_point",
    ...                           default=panel.centre(),
    ...                           candidates=panel.nodes0)
    """
    import numpy as np

    nodes = np.asarray(nodes, dtype=float)
    if nodes.shape[1] == 2:                     # a 2D mesh lives on z = 0
        nodes = np.column_stack([nodes, np.zeros(len(nodes))])

    pool = (np.arange(len(nodes)) if candidates is None
            else np.unique(np.asarray(candidates, dtype=int)))
    if pool.size == 0:
        raise RuntimeError(f"{question}: there are no nodes to choose from.")

    shown = None
    if default is not None:
        shown = ", ".join(f"{float(v):g}" for v in np.asarray(default).ravel())

    # The prompt carries the box the answer has to fall in: a coordinate is
    # meaningless without the extent of what it is a coordinate of.
    lo, hi = nodes[pool].min(axis=0), nodes[pool].max(axis=0)
    extent = ", ".join(f"{axis} {a:g}..{b:g}"
                       for axis, a, b in zip("xyz", lo, hi))
    prompt = f"pyCAFE — {question}\n({extent} [{unit}])"
    if shown is not None:
        prompt += f"\ndefault: {shown}" + (f" — {default_is}" if default_is
                                           else "")
    answer, source = _env_value(name)
    if answer is None:
        answer = (_ask_text_with_osascript(prompt, shown)
                  if _osascript_available() else None)
        source = "the dialog"
        if answer is None:
            answer = _ask_text_with_dialog(prompt, shown)
            if answer is None:
                answer = _ask_text_on_console(prompt, shown)
                source = "the console"
                if answer is None:
                    raise RuntimeError(
                        f"{question}: nothing could be asked — no display "
                        "and no console. Set "
                        f"{ANSWER_PREFIX}_{name.upper()}=\"x, y, z\" to "
                        "answer from the environment."
                    )
        if not answer:
            raise RuntimeError(
                f"{question}: no answer. pyCAFE does not pick a point for "
                f"you — answer, or set {ANSWER_PREFIX}_{name.upper()}="
                "\"x, y, z\"."
            )

    parts = [p for p in str(answer).replace(",", " ").split() if p]
    try:
        wanted = np.array([float(p) for p in parts], dtype=float)
    except ValueError:
        raise RuntimeError(
            f"{question}: {answer!r} (from {source}) is not a list of numbers."
        ) from None
    if wanted.size == 2:
        wanted = np.append(wanted, 0.0)
    if wanted.size != 3:
        raise RuntimeError(
            f"{question}: {answer!r} (from {source}) is {wanted.size} "
            "numbers, not the two or three a point takes."
        )

    distances = np.linalg.norm(nodes[pool] - wanted, axis=1)
    node0 = int(pool[np.argmin(distances)])
    gap = float(distances.min())
    print(f"{name}: asked ({', '.join(f'{v:g}' for v in wanted)}) "
          f"-> node {node0} at ({', '.join(f'{v:.4g}' for v in nodes[node0])}), "
          f"{gap * 1e3:.1f} mm away")
    return node0, nodes[node0]



# What a face of an acoustic model can be told to do. The catalogue is
# the physics: nothing at all (a hard wall), a pressure (zero, which is a
# soft wall, or a value), an impedance, or a prescribed motion. An
# anechoic end is the impedance zeta = 1, the value the dialog opens on.
ACOUSTIC_BOUNDARY_CONDITIONS = (
    "hard wall",
    "soft wall",
    "imposed pressure",
    "impedance",
    "velocity",
)

# The names the same conditions used to go by, still accepted so that a
# script or an environment answer written earlier keeps working.
BC_ALIASES = {
    "rigid": "hard wall",
    "rigid wall": "hard wall",
    "pressure release": "soft wall",
    "p=0": "soft wall",
}

# What can be applied at a single point instead of over a face.
ACOUSTIC_POINT_SOURCES = ("none", "monopole", "pressure at a point")


def ask_for_many(question, options, *, name="selection", default=None,
                 allow_none=True):
    """
    Ask for several of the answers at once, and return them as a list.

    The counterpart of :func:`ask_for_choice` for a question whose answer
    is a subset: which faces carry a boundary condition, which groups to
    plot. The answer is typed or given as a comma separated list, and
    two words stand for the obvious extremes: ``none`` and ``all``.

    Parameters
    ----------
    question : str
    options : sequence of str
        What may be picked. Anything outside the list is refused, so a
        typo cannot quietly select nothing.
    name : str, optional
        Environment variable, ``PYCAFE_ANSWER_<NAME>``, answered as
        ``"inlet, outlet"``, ``"all"`` or ``"none"``.
    default : str, optional
        What the dialog opens on.
    allow_none : bool, optional
        Whether an empty selection is a valid answer. Default True.

    Returns
    -------
    list of str
        In the order of ``options``, each name once.

    Raises
    ------
    RuntimeError
        For an unanswered question, a name outside ``options``, or an
        empty answer where ``allow_none`` is False.

    Examples
    --------
    >>> ask_for_many("which faces carry a condition?",        # doctest: +SKIP
    ...              ["inlet", "outlet", "wall"], name="bc_faces")
    """
    options = [str(o) for o in options]
    if not options:
        return []

    prompt = (f"pyCAFE — {question}\n"
              f"one of: {', '.join(options)}\n"
              "several are separated by commas; 'all' or 'none' also work")
    shown = "none" if default is None else str(default)

    answer, source = _env_value(name)
    if answer is None:
        answer = (_ask_text_with_osascript(prompt, shown)
                  if _osascript_available() else None)
        source = "the dialog"
        if answer is None:
            answer = _ask_text_with_dialog(prompt, shown)
            if answer is None:
                answer = _ask_text_on_console(prompt, shown)
                source = "the console"
                if answer is None:
                    raise RuntimeError(
                        f"{question}: nothing could be asked — no display "
                        "and no console. Set "
                        f"{ANSWER_PREFIX}_{name.upper()}=<names> to answer "
                        "from the environment."
                    )
        if answer is None or answer == "":
            raise RuntimeError(
                f"{question}: no answer. pyCAFE does not pick one for you — "
                f"answer, or set {ANSWER_PREFIX}_{name.upper()}."
            )

    text = str(answer).strip().lower()
    if text in ("none", "-", "no"):
        picked = []
    elif text == "all":
        picked = list(options)
    else:
        wanted = [part.strip() for part in str(answer).split(",")
                  if part.strip()]
        lower = {o.lower(): o for o in options}
        picked = []
        for item in wanted:
            if item.lower() not in lower:
                raise RuntimeError(
                    f"{question}: '{item}' (from {source}) is not one of "
                    f"{options}."
                )
            if lower[item.lower()] not in picked:
                picked.append(lower[item.lower()])
        picked = [o for o in options if o in picked]

    if not picked and not allow_none:
        raise RuntimeError(f"{question}: nothing was picked, and this "
                           "question needs at least one answer.")
    print(f"{name}: {', '.join(picked) if picked else 'none'}")
    return picked


def ask_for_acoustic_bc(faces, *, default="impedance",
                        options=ACOUSTIC_BOUNDARY_CONDITIONS,
                        default_zeta=1.0, default_velocity=1.0,
                        default_pressure=1.0):
    """
    Ask which faces carry a condition, and what each of them does.

    Two questions rather than one per face: **most faces of most models
    are hard walls**, and answering that a dozen times is not a
    modelling decision. So the first question is which faces are not
    walls, and only those are asked about. Everything unnamed keeps
    :math:`\partial p/\partial n = 0`, which costs no matrix and is the
    natural condition of the acoustic problem.

    ====================  =====================================================
    ``hard wall``         nothing to assemble; the default for every face
    ``soft wall``         :math:`p = 0`, those DOFs leave the problem
    ``imposed pressure``  a prescribed pressure [Pa]
    ``impedance``         :math:`\zeta = Z/(\rho_0 c_0)`; :math:`\zeta = 1`
                          is the plane-wave anechoic end
    ``velocity``          a prescribed outward normal velocity [m/s]
    ====================  =====================================================

    With nobody at the keyboard, ``PYCAFE_ANSWER_BC_FACES`` picks the
    faces (``"inlet, outlet"``, ``"all"``, ``"none"``) and
    ``PYCAFE_ANSWER_BC_<FACE>`` says what each does, with
    ``PYCAFE_ANSWER_ZETA_<FACE>``, ``PYCAFE_ANSWER_PRESSURE_<FACE>`` or
    ``PYCAFE_ANSWER_VELOCITY_<FACE>`` for the number it needs. The names
    this menu used to carry, ``rigid`` and ``pressure release``, are
    still understood.

    Parameters
    ----------
    faces : sequence of str
        Every face of the fluid, whether or not it ends up carrying
        anything.
    default : str, optional
        Which condition the per-face dialog opens on.
    options : sequence of str, optional
    default_zeta, default_velocity, default_pressure : float, optional

    Returns
    -------
    bc : AcousticBC
    answers : dict
        What each face was told to do, the hard walls included, for
        printing and for the title of a figure.

    Examples
    --------
    >>> bc, answers = ask_for_acoustic_bc(["inlet", "outlet"])  # doctest: +SKIP
    """
    from pycafe.boundary_condition.acoustic_bc import AcousticBC

    faces = [str(f) for f in faces]
    bc = AcousticBC()
    answers = {name: "hard wall" for name in faces}
    if not faces:
        return bc, answers

    chosen = ask_for_many(
        "which faces carry a boundary condition? "
        "(everything else stays a hard wall)",
        faces, name="bc_faces", default="none",
    )

    for name in chosen:
        answer = ask_for_choice(
            f"what does the face '{name}' do?", options,
            name=f"bc_{name}", default=default,
        )
        answer = BC_ALIASES.get(answer.strip().lower(), answer)
        answers[name] = answer

        if answer == "hard wall":
            bc.add_rigid_wall(name)
        elif answer == "soft wall":
            bc.add_pressure(name, 0.0)
            answers[name] = "soft wall, p = 0"
        elif answer == "imposed pressure":
            value = ask_for_number(f"pressure imposed on '{name}'?", unit="Pa",
                                   name=f"pressure_{name}",
                                   default=default_pressure)
            bc.add_pressure(name, value)
            answers[name] = f"p = {value:g} Pa"
        elif answer == "impedance":
            zeta = ask_for_number(
                f"normalized impedance zeta = Z / (rho0 c0) on '{name}'? "
                "(1 is the anechoic end)",
                name=f"zeta_{name}", default=default_zeta, minimum=0.0,
            )
            bc.add_impedance(name, zeta)
            answers[name] = (f"impedance zeta = {zeta:g}"
                             + (" (anechoic)" if zeta == 1.0 else ""))
        elif answer == "velocity":
            v_n = ask_for_number(
                f"outward normal velocity on '{name}'?", unit="m/s",
                name=f"velocity_{name}", default=default_velocity,
            )
            bc.add_velocity(name, v_n)
            answers[name] = f"v_n = {v_n:g} m/s"
        else:
            raise RuntimeError(
                f"'{answer}' is not a boundary condition pyCAFE assembles "
                f"here; the ones it does are {list(options)}."
            )
    return bc, answers


def ask_for_point_source(bc, nodes, *, candidates=None, default="none",
                         options=ACOUSTIC_POINT_SOURCES,
                         default_position=None, default_is=None,
                         default_strength=1e-3, default_pressure=1.0):
    """
    Ask for the one thing an acoustic model can be given at a point.

    A face carries a condition; a point carries a source. Two of them:
    a **monopole**, a volume velocity injected into the fluid, and a
    **prescribed pressure at a node**, which drives the model by holding
    one DOF at a value instead of by pushing on it.

    Answered from the environment with ``PYCAFE_ANSWER_POINT_SOURCE``,
    then ``PYCAFE_ANSWER_SOURCE_POINT`` for where it is and
    ``PYCAFE_ANSWER_MONOPOLE_STRENGTH`` or
    ``PYCAFE_ANSWER_POINT_PRESSURE`` for how strong.

    Parameters
    ----------
    bc : AcousticBC
        The set the source is added to, in place.
    nodes : ndarray (N, 3)
        Mesh node coordinates.
    candidates : array-like of int, optional
        0-based node indices the point may snap to; pass the fluid
        nodes, since a source belongs in the fluid.
    default : str, optional
        Which answer the first dialog opens on.
    options : sequence of str, optional
        Which sources are offered. A coupled run leaves out
        ``"pressure at a point"``, which its two-block solver has no way
        of imposing.
    default_position : array-like, optional
        Where the point dialog opens.
    default_is : str, optional
        What that place is ("the middle of the fluid").
    default_strength : float, optional
        Volume velocity the monopole dialog opens on [m3/s].
    default_pressure : float, optional
        Pressure the point-pressure dialog opens on [Pa].

    Returns
    -------
    kind : str
        ``"none"``, ``"monopole"`` or ``"pressure at a point"``.
    node0 : int or None
        0-based node the source sits on.
    value : float or None
        Its volume velocity [m3/s] or its pressure [Pa].

    Examples
    --------
    >>> kind, node, value = ask_for_point_source(bc, nodes)  # doctest: +SKIP
    """
    kind = ask_for_choice("anything applied at a point?", options,
                          name="point_source", default=default)
    if kind == "none":
        return kind, None, None

    node0, _ = ask_for_point(
        f"where is the {kind}?", nodes, name="source_point",
        default=default_position, default_is=default_is,
        candidates=candidates,
    )
    if kind == "monopole":
        value = ask_for_number("volume flow of the monopole?", unit="m3/s",
                               name="monopole_strength",
                               default=default_strength)
        bc.add_monopole(value, node=node0)
    else:
        value = ask_for_number("pressure held at that node?", unit="Pa",
                               name="point_pressure",
                               default=default_pressure)
        bc.add_point_pressure(node0, value)
    return kind, node0, value


def ask_for_points(question, nodes, *, name="points", default=None,
                   default_is=None, candidates=None, unit="m", maximum=None):
    """
    Ask where several points are, and return the nearest node of each.

    The plural of :func:`ask_for_point`: one microphone is a special
    case, not the rule, and a sweep costs the same whether it is read at
    one node or at six. The answer is a list of coordinates separated by
    semicolons, ``"0.1, 0.2, 0.3;  0.7, 0.2, 0.3"``.

    Parameters
    ----------
    question : str
    nodes : ndarray (N, 3) or (N, 2)
    name : str, optional
        Environment variable, ``PYCAFE_ANSWER_<NAME>``, answered the
        same way.
    default : array-like, optional
        One point the dialog opens on, or a list of them.
    default_is : str, optional
        What that place is ("the middle of the fluid").
    candidates : array-like of int, optional
        0-based node indices the answers may snap to.
    unit : str, optional
    maximum : int, optional
        How many points are accepted at most.

    Returns
    -------
    list of (int, ndarray)
        ``(node0, coordinates)`` for each point asked for, in the order
        given.

    Raises
    ------
    RuntimeError
        For an unanswered question, an answer that is not a list of
        coordinates, or more points than ``maximum``.

    Examples
    --------
    >>> mics = ask_for_points("where are the microphones?", nodes,  # doctest: +SKIP
    ...                       name="microphones", candidates=fluid_nodes0)
    """
    import numpy as np

    nodes = np.asarray(nodes, dtype=float)
    if nodes.shape[1] == 2:
        nodes = np.column_stack([nodes, np.zeros(len(nodes))])
    pool = (np.arange(nodes.shape[0]) if candidates is None
            else np.unique(np.asarray(candidates, dtype=int)))

    shown = None
    if default is not None:
        first = np.asarray(default, dtype=float)
        if first.ndim == 1:
            shown = ", ".join(f"{v:g}" for v in first)
        else:
            shown = ";  ".join(", ".join(f"{v:g}" for v in row)
                               for row in first)
    prompt = (f"pyCAFE — {question}"
              + (f" [{unit}]" if unit else "")
              + "\nas \"x, y, z\"; several are separated by semicolons"
              + (f"\n({default_is})" if default_is else ""))

    answer, source = _env_value(name)
    if answer is None:
        answer = (_ask_text_with_osascript(prompt, shown)
                  if _osascript_available() else None)
        source = "the dialog"
        if answer is None:
            answer = _ask_text_with_dialog(prompt, shown)
            if answer is None:
                answer = _ask_text_on_console(prompt, shown)
                source = "the console"
                if answer is None:
                    raise RuntimeError(
                        f"{question}: nothing could be asked — no display "
                        "and no console. Set "
                        f"{ANSWER_PREFIX}_{name.upper()}=\"x, y, z; ...\" "
                        "to answer from the environment."
                    )
        if not answer:
            raise RuntimeError(
                f"{question}: no answer. pyCAFE does not pick one for you — "
                f"answer, or set {ANSWER_PREFIX}_{name.upper()}."
            )

    groups = [g for g in str(answer).split(";") if g.strip()]
    if maximum is not None and len(groups) > maximum:
        raise RuntimeError(
            f"{question}: {len(groups)} points asked for, at most {maximum} "
            "are accepted here."
        )

    picked = []
    for k, group in enumerate(groups, start=1):
        parts = [v for v in group.replace(",", " ").split() if v]
        try:
            wanted = np.array([float(v) for v in parts], dtype=float)
        except ValueError:
            raise RuntimeError(
                f"{question}: {group!r} (from {source}) is not a list of "
                "numbers."
            ) from None
        if wanted.size == 2:
            wanted = np.append(wanted, 0.0)
        if wanted.size != 3:
            raise RuntimeError(
                f"{question}: {group!r} (from {source}) is {wanted.size} "
                "numbers, not the two or three a point takes."
            )
        distances = np.linalg.norm(nodes[pool] - wanted, axis=1)
        node0 = int(pool[np.argmin(distances)])
        gap = float(distances.min())
        print(f"{name} {k}: asked ({', '.join(f'{v:g}' for v in wanted)}) "
              f"-> node {node0} at "
              f"({', '.join(f'{v:.4g}' for v in nodes[node0])}), "
              f"{gap * 1e3:.1f} mm away")
        picked.append((node0, nodes[node0]))
    return picked
