r"""
One declarative description of a model, from geometry to assembled system.

Every pyCAFE run needs the same four things — a geometry, a mesh fine
enough for the frequencies asked of it, material data, and the physical
groups that say which part is fluid and which is structure. Scripts used
to spell those out in a different order each time, and the element count
was picked by hand. :class:`ModelSpec` collects them in one object and
derives what can be derived:

.. code-block:: python

    from pycafe.core.model_spec import ModelSpec, Library, AIR, aluminium

    spec = ModelSpec(
        geometry=Library("box_with_plate", Lx=0.4, Ly=0.3, Lz=0.5),
        fluid=AIR,
        structure=aluminium(t=2e-3),
        f_max=500.0,                    # the one number that sizes the mesh
    )
    model = build_model(spec)           # mesh, validate, assemble

The mesh size is **not** an input: it follows from ``f_max`` and
``elements_per_wavelength`` (10 by default) as

.. math:: h = \frac{c_0}{f_{max} \, n_\lambda},

so asking for a higher frequency refines the mesh instead of quietly
running an under-resolved one. The criterion is the acoustic wavelength
only; a thin panel driven above its coincidence frequency has a shorter
bending wavelength than the sound in the fluid, and there the plate mesh
has to be checked separately.

Four kinds of geometry are accepted, and they are the answer to "where
does the model come from":

=====================  ==============================================
:class:`Library`       we build it, from parameters (cavity, duct, ...)
:class:`CadFile`       the user brings a STEP/IGES file
:class:`NastranFile`   the user brings a Nastran bulk data deck
:class:`MeshFile`      the user brings a mesh already
=====================  ==============================================

A CAD file carries no roles, so :class:`CadFile` asks for them
explicitly: which entity tags are the fluid, which are the structure.
Get the tags from
:func:`~pycafe.create_geom.external.inspect_cad`, which prints what the
file contains. A ``.bdf`` names its parts by property id instead, and
:func:`~pycafe.create_geom.nastran.inspect_bdf` prints those together
with the property and material cards behind them.
"""

import math
import pathlib
import warnings
from dataclasses import dataclass, field, replace
from typing import Dict, Optional, Sequence, Union

from pycafe.create_geom.external import (
    assign_groups,
    by_tag,
    import_cad,
    inspect_cad,
    rest,
    retag_mesh,
)
from pycafe.create_geom.gmsh_workflow import GmshModel
from pycafe.create_geom.library import GEOMETRIES
from pycafe.create_geom.validation import validate_mesh

# Below this many elements per wavelength the trilinear acoustic element
# is not resolving a wave at all (the dispersion error goes as
# 164 / n_lambda^2, i.e. ~4.5% at 6 elements); pyCAFE refuses rather than
# produce a number that looks like a result.
MIN_ELEMENTS_PER_WAVELENGTH = 6
DEFAULT_ELEMENTS_PER_WAVELENGTH = 10


# materials
@dataclass(frozen=True)
class Fluid:
    """
    Acoustic medium.

    Parameters
    ----------
    rho0 : float
        Density [kg/m^3].
    c0 : float
        Speed of sound [m/s].
    name : str, optional
        Label, for reporting only.
    """

    rho0: float = 1.204
    c0: float = 343.0
    name: str = "air"

    @property
    def Z0(self):
        """Characteristic impedance ``rho0 * c0`` [Pa s/m]."""
        return self.rho0 * self.c0

    def wavelength(self, f):
        """Wavelength at frequency ``f`` [m]."""
        return self.c0 / float(f)


AIR = Fluid(1.204, 343.0, "air")
WATER = Fluid(998.0, 1480.0, "water")


@dataclass(frozen=True)
class Structure:
    """
    Shell material and thickness, plus how its support behaves.

    Parameters
    ----------
    t : float
        Thickness [m].
    E : float
        Young's modulus [Pa].
    nu : float
        Poisson ratio.
    rho_s : float
        Density [kg/m^3].
    nsm : float, optional
        Non-structural mass per unit area [kg/m^2].
    support : {"clamped", "simply_supported", "free"}, optional
        What is blocked on the nodes of the clamp group. The mesh only
        says *which* nodes are held; this says how.
    name : str, optional
        Label, for reporting only.
    """

    t: float
    E: float = 70e9
    nu: float = 0.33
    rho_s: float = 2700.0
    nsm: float = 0.0
    support: str = "clamped"
    name: str = "custom"

    @property
    def D(self):
        """Bending stiffness ``E t^3 / (12 (1 - nu^2))`` [N m]."""
        return self.E * self.t ** 3 / (12.0 * (1.0 - self.nu ** 2))

    def bending_wavelength(self, f):
        """
        Free bending wavelength of the plate at ``f`` [m].

        ``lambda_b = 2 pi / (rho_s t omega^2 / D)^(1/4)``. Not used to
        size the mesh — the sizing rule is acoustic — but reported, so
        that a model run above coincidence shows it.
        """
        omega = 2.0 * math.pi * float(f)
        k_b = (self.rho_s * self.t * omega ** 2 / self.D) ** 0.25
        return 2.0 * math.pi / k_b


def aluminium(t, **kwargs):
    """Aluminium shell of thickness ``t`` [m]: E = 70 GPa, nu = 0.33, 2700 kg/m^3."""
    return Structure(t=t, E=70e9, nu=0.33, rho_s=2700.0, name="aluminium",
                     **kwargs)


def steel(t, **kwargs):
    """Steel shell of thickness ``t`` [m]: E = 210 GPa, nu = 0.30, 7850 kg/m^3."""
    return Structure(t=t, E=210e9, nu=0.30, rho_s=7850.0, name="steel",
                     **kwargs)


# geometry sources
class Library:
    """
    A geometry pyCAFE builds itself, from its parameters.

    The element counts are **not** given here: they come from the mesh
    size of the spec, i.e. from ``f_max``. Passing one anyway (``nx=12``)
    overrides the derived value for that direction, which is what a
    convergence study needs.

    Parameters
    ----------
    name : str
        One of :data:`pycafe.create_geom.library.GEOMETRIES`:
        ``box_cavity``, ``box_with_plate``, ``duct_2d``, ``plate``,
        ``duct_with_flush_plate``.
    **params
        Passed to the builder — dimensions, ``open_faces``,
        ``plate_length``, ...

    Examples
    --------
    >>> Library("box_cavity", Lx=0.4, Ly=0.3, Lz=0.5).name
    'box_cavity'
    """

    def __init__(self, name, **params):
        if name not in GEOMETRIES:
            raise ValueError(
                f"Unknown geometry '{name}'. Available: {sorted(GEOMETRIES)}."
            )
        self.name = str(name)
        self.params = dict(params)

    def __repr__(self):
        inside = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
        return f"Library({self.name!r}{', ' if inside else ''}{inside})"


@dataclass
class CadFile:
    """
    Geometry the user brings in: a STEP/IGES/BREP file, plus its roles.

    A CAD file says nothing about physics, so the two questions pyCAFE
    cannot answer on its own are asked here: **where is the file** and
    **which entity is the fluid, which the structure**. Run
    :func:`~pycafe.create_geom.external.inspect_cad` on the file first —
    it prints every volume and surface with its tag, size and bounding
    box — then quote those tags below.

    Parameters
    ----------
    path : str or Path
        The CAD file.
    fluid : sequence of int
        Tags of the entities holding the fluid: volumes in 3D
        (``dim=3``), surfaces in 2D.
    structure : sequence of int, optional
        Tags of the surfaces (3D) or curves (2D) that are the flexible
        structure. Leave empty for a purely acoustic model.
    clamp : sequence of int or "auto", optional
        Tags of the entities holding the supported nodes, one dimension
        below the structure. ``"auto"`` (default) takes the whole
        boundary of the structure entities, which is the usual case: a
        panel held all around its edge.
    boundaries : dict, optional
        ``{group_name: [tags]}`` for the free names used later in the
        boundary conditions (``{"inlet": [3], "outlet": [8]}``).
    walls : str or None, optional
        Name given to every remaining boundary entity. Default
        ``"rigid_walls"``; ``None`` leaves them untagged.
    units : {"auto", "m", "mm", "cm"}, optional
        Unit of the file. ``"auto"`` reads anything larger than 100
        units as millimetres.
    dim : int, optional
        Dimension of the model, 3 or 2.
    recombine : bool, optional
        Ask Gmsh for quads/hexes. It only succeeds where the geometry
        allows it; otherwise the fluid comes out tetrahedral (CTETRA4),
        and a tetrahedral fluid has triangular faces, which no shell
        kernel of pyCAFE can use. The validator reports that.
    independent_structure : bool, optional
        Mesh the fluid and the structure in two separate passes and
        merge the results, instead of asking one mesh to serve both.
        The fluid then gets the element its shape allows (tetrahedra on
        a curved volume) and the panel keeps the quadrilaterals CQUAD4F
        needs, at the price of an interface that shares no node: the
        coupling is built by interpolation
        (:mod:`pycafe.build_matrices.coupling_nonconforming`). This is
        the way to couple a geometry that cannot be cut into six-sided
        blocks.
    structure_element_size : float, optional
        Element size of the structural pass [m]. Default: the same as
        the fluid. Only used with ``independent_structure``.

    Raises
    ------
    ValueError
        If no fluid tag is given.
    """

    path: Union[str, pathlib.Path]
    fluid: Sequence[int] = ()
    structure: Sequence[int] = ()
    clamp: Union[str, Sequence[int]] = "auto"
    boundaries: Dict[str, Sequence[int]] = field(default_factory=dict)
    walls: Optional[str] = "rigid_walls"
    units: str = "auto"
    dim: int = 3
    recombine: bool = True
    independent_structure: bool = False
    structure_element_size: Optional[float] = None

    def __post_init__(self):
        self.path = pathlib.Path(self.path)
        if not self.path.exists():
            raise FileNotFoundError(f"No CAD file at {self.path}.")
        if len(tuple(self.fluid)) == 0:
            raise ValueError(
                f"CadFile({self.path.name}) needs the tags of the fluid "
                "entities: a CAD file carries no roles. Run "
                "inspect_cad(path) to see what is in it."
            )

    def inspect(self, **kwargs):
        """Shortcut for :func:`~pycafe.create_geom.external.inspect_cad`."""
        return inspect_cad(self.path, units=self.units, **kwargs)


@dataclass
class NastranFile:
    """
    A Nastran bulk data deck the user brings in, plus its roles.

    The deck already holds a mesh, so nothing is remeshed and ``f_max``
    no longer sizes anything: it is only checked against what the mesh
    can resolve. What the deck does *not* hold is physical groups —
    the format has none — so the roles are stated here by **property
    id**, which is what Gmsh turns each entity tag into. Run
    :func:`~pycafe.create_geom.nastran.inspect_bdf` on the file first:
    it lists every property id with its element count, its ``PSHELL`` /
    ``PSOLID`` and material cards, and the role it looks like.

    Parameters
    ----------
    path : str or Path
        The ``.bdf`` or ``.nas`` file.
    fluid : sequence of int
        Property ids of the acoustic domain.
    structure : sequence of int, optional
        Property ids of the shell elements that are the structure.
    boundaries : dict, optional
        ``{group_name: [pids]}`` for surface property ids the deck
        already carries, given the free names the boundary conditions
        will select them by.
    surface_rules : sequence of TagRule, optional
        Names given to the **free faces of the fluid**, the ones no
        surface element covers. An acoustic deck usually holds volume
        elements only, so without this there is no face for an
        impedance or a prescribed velocity to act on. Same rules as
        everywhere else: ``on_plane("x", 0.0, name="inlet")``,
        ``in_box(...)``.
    walls : str or None, optional
        Name for the free faces no rule claimed. Default
        ``"rigid_walls"``, which is the natural boundary condition
        anyway; ``None`` leaves them out of the mesh.
    clamp : {"spc"}, int, sequence of int or None, optional
        The support of the structure. ``"spc"`` (default) reads the
        single ``SPC``/``SPC1`` set of the deck, an integer picks one
        by id, a sequence gives the grid ids. Which degrees of freedom
        are blocked is not read from the card — that is
        ``Structure.support``.
    units : {"m", "auto", "mm", "cm", "in"}, optional
        Unit of the deck. A bulk data file records none, so the default
        is to take the coordinates as written.

    Notes
    -----
    The thickness and the material of the deck are used only for the
    fields the spec left out: pass a :class:`Structure` or a
    :class:`Fluid` and the spec wins, with the disagreement reported.
    See :func:`build_model`.

    Raises
    ------
    ValueError
        If neither a fluid nor a structure property id is given.
    """

    path: Union[str, pathlib.Path]
    fluid: Sequence[int] = ()
    structure: Sequence[int] = ()
    boundaries: Dict[str, Sequence[int]] = field(default_factory=dict)
    surface_rules: Sequence = ()
    walls: Optional[str] = "rigid_walls"
    clamp: Union[str, int, Sequence[int], None] = "spc"
    units: str = "m"

    def __post_init__(self):
        self.path = pathlib.Path(self.path)
        if not self.path.exists():
            raise FileNotFoundError(f"No bulk data file at {self.path}.")
        if not tuple(self.fluid) and not tuple(self.structure):
            raise ValueError(
                f"NastranFile({self.path.name}) needs the property ids of "
                "the fluid, of the structure, or of both: a bulk data file "
                "carries no roles. Run inspect_bdf(path) to see what is in "
                "it."
            )
        self._deck = None

    @property
    def deck(self):
        """
        The cards Gmsh discards (:class:`~pycafe.create_geom.nastran.BdfDeck`).

        Read once and kept, since both the mesh and the materials are
        taken from it.
        """
        if self._deck is None:
            from pycafe.create_geom.nastran import read_bdf

            self._deck = read_bdf(self.path)
        return self._deck

    def inspect(self, **kwargs):
        """Shortcut for :func:`~pycafe.create_geom.nastran.inspect_bdf`."""
        from pycafe.create_geom.nastran import inspect_bdf

        return inspect_bdf(self.path, units=self.units, **kwargs)


@dataclass
class MeshFile:
    """
    A mesh the user brings in, optionally with its groups renamed.

    Nothing is remeshed here, so ``f_max`` no longer sizes anything: it
    is only checked against what the mesh can resolve.

    Parameters
    ----------
    path : str or Path
        The ``.msh`` file.
    rename : dict, optional
        ``{old_group: new_group}``, to map foreign names onto the pyCAFE
        roles (``{"AIR": "fluid", "SKIN": "plate"}``).
    drop : sequence of str, optional
        Groups to remove.
    """

    path: Union[str, pathlib.Path]
    rename: Dict[str, str] = field(default_factory=dict)
    drop: Sequence[str] = ()

    def __post_init__(self):
        self.path = pathlib.Path(self.path)
        if not self.path.exists():
            raise FileNotFoundError(f"No mesh at {self.path}.")


GeometrySource = Union[Library, CadFile, NastranFile, MeshFile]

# Which extension is which kind of input. A file the user picks says what
# it is by its name, and that is the whole of the dispatch below.
GEOMETRY_SUFFIXES = {
    ".step": "cad", ".stp": "cad", ".iges": "cad", ".igs": "cad",
    ".brep": "cad",
    ".msh": "mesh",
    ".bdf": "nastran", ".nas": "nastran", ".dat": "nastran",
}


def geometry_from_file(path, *, fluid=None, structure=None, verbose=True,
                       **kwargs):
    """
    Turn a file the user picked into the geometry source that reads it.

    The three inputs pyCAFE accepts from outside are told apart by their
    extension, and each one answers the role question differently:

    ==================  ==========================================
    ``.step`` ``.stp``  :class:`CadFile` — meshed here; a CAD file
    ``.iges`` ``.brep``   carries no roles, so without ``fluid``
                          every solid is taken as the fluid
    ``.bdf`` ``.nas``   :class:`NastranFile` — already meshed; the
                          roles are read from the property and
                          material cards
                          (:meth:`~pycafe.create_geom.nastran.BdfDeck.suggest_roles`)
    ``.msh``            :class:`MeshFile` — already meshed *and*
                          already named, nothing to guess
    ==================  ==========================================

    What is guessed is printed, because a guess about which solid is the
    fluid is exactly the kind of thing that must not pass silently. Pass
    ``fluid=`` / ``structure=`` to state it instead — CAD entity tags
    from :func:`~pycafe.create_geom.external.inspect_cad`, property ids
    from :func:`~pycafe.create_geom.nastran.inspect_bdf`.

    Parameters
    ----------
    path : str or Path
    fluid : sequence of int, optional
        Entity tags (CAD) or property ids (Nastran) of the fluid.
    structure : sequence of int, optional
        Same, for the flexible structure.
    verbose : bool, optional
        Print the source that was built and what was assumed.
    **kwargs
        Passed to the source: ``units``, ``boundaries``, ``recombine``,
        ``independent_structure``, ``rename``, ``drop``, ...

    Returns
    -------
    CadFile, NastranFile or MeshFile

    Raises
    ------
    FileNotFoundError
        If there is no such file.
    ValueError
        For an extension that is none of the above.

    Examples
    --------
    >>> geometry_from_file("Library/box_cavity.msh")     # doctest: +SKIP
    MeshFile(path=PosixPath('Library/box_cavity.msh'), rename={}, drop=())
    """
    path = pathlib.Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"No file at {path}.")

    kind = GEOMETRY_SUFFIXES.get(path.suffix.lower())
    if kind is None:
        raise ValueError(
            f"{path.name}: pyCAFE reads CAD files "
            f"({', '.join(s for s, k in GEOMETRY_SUFFIXES.items() if k == 'cad')}), "
            "Nastran decks (.bdf, .nas, .dat) and Gmsh meshes (.msh), not "
            f"'{path.suffix}'."
        )

    if kind == "mesh":
        source = MeshFile(path, **kwargs)
        if verbose:
            print(f"{path.name}: a mesh, used as it is (its physical groups "
                  "already carry the roles).")
        return source

    if kind == "nastran":
        from pycafe.create_geom.nastran import read_bdf

        if fluid is None or structure is None:
            roles = read_bdf(path).suggest_roles()
            if fluid is None:
                fluid = [pid for pid, _why in roles["fluid"]]
            if structure is None:
                structure = [pid for pid, _why in roles["structure"]]
            if verbose:
                print(f"{path.name}: a Nastran deck. Roles read from the "
                      f"cards — fluid PID {list(fluid)}, structure PID "
                      f"{list(structure)}. Check them with inspect_bdf.")
        return NastranFile(path, fluid=tuple(fluid), structure=tuple(structure),
                           **kwargs)

    units = kwargs.get("units", "auto")
    dim = int(kwargs.get("dim", 3))
    if fluid is None:
        found = inspect_cad(path, units=units, dims=(dim,), verbose=False)
        fluid = [info.tag for info in found[dim]]
        if verbose:
            print(f"{path.name}: a CAD file, meshed here. It carries no "
                  f"roles, so its {len(fluid)} solid(s) {fluid} are taken as "
                  "the fluid and nothing as the structure. Run inspect_cad "
                  "and pass fluid=/structure= to say otherwise.")
    return CadFile(path, fluid=tuple(fluid),
                   structure=tuple(structure or ()), **kwargs)


def _same_file(a, b):
    """
    Do two paths name the same file?

    Compared after resolving, so ``mesh.msh`` and ``./meshes/../mesh.msh``
    are recognised as one file; a path that does not exist yet resolves
    lexically, which is what a not-yet-written output needs.
    """
    return pathlib.Path(a).resolve() == pathlib.Path(b).resolve()


# the spec
@dataclass
class ModelSpec:
    """
    Everything a run needs, in one object.

    Parameters
    ----------
    geometry : Library, CadFile or MeshFile
        Where the geometry comes from.
    f_max : float
        Highest frequency of interest [Hz]. It sizes the mesh; nothing
        else in the spec does.
    fluid : Fluid, optional
        Default :data:`AIR`.
    structure : Structure, optional
        Required as soon as the mesh has a structural group.
    elements_per_wavelength : int, optional
        Elements across the shortest acoustic wavelength. Default
        :data:`DEFAULT_ELEMENTS_PER_WAVELENGTH` (10), roughly 1.6%
        dispersion error; below
        :data:`MIN_ELEMENTS_PER_WAVELENGTH` (6) the spec is rejected.
    analysis : {"auto", "acoustic", "vibroacoustic", "structural"}, optional
        What to assemble. ``"auto"`` reads it from the groups found in
        the mesh.
    output_path : str or Path, optional
        Where to write the mesh. Default ``<work_dir>/<name>.msh``; see
        :attr:`mesh_path` for what a :class:`MeshFile` does instead, and
        set this to the source file to ask for an overwrite.
    work_dir : str or Path, optional
        Directory for the generated mesh. Default ``"."``.
    pml : dict or False, optional
        Passed to :func:`~pycafe.core.prepare_acoustic_system.prepare_acoustic_system`;
        the layer itself comes from a physical group named ``pml``.
    build_coupling : bool, optional
        Assemble the fluid-structure coupling matrix. Default True.
    verbose : bool, optional

    Raises
    ------
    ValueError
        For a non-positive ``f_max`` or a mesh density below the
        practical minimum.

    Examples
    --------
    >>> spec = ModelSpec(geometry=Library("box_cavity"), f_max=500.0)
    >>> round(spec.element_size, 4)
    0.0686
    >>> spec.elements_per_wavelength
    10
    """

    geometry: GeometrySource
    f_max: float
    fluid: Fluid = AIR
    structure: Optional[Structure] = None
    elements_per_wavelength: int = DEFAULT_ELEMENTS_PER_WAVELENGTH
    analysis: str = "auto"
    output_path: Union[str, pathlib.Path, None] = None
    work_dir: Union[str, pathlib.Path] = "."
    pml: Union[dict, bool, None] = None
    build_coupling: bool = True
    verbose: bool = False

    def __post_init__(self):
        if float(self.f_max) <= 0.0:
            raise ValueError(f"f_max must be positive, got {self.f_max}.")
        n = int(self.elements_per_wavelength)
        if n < MIN_ELEMENTS_PER_WAVELENGTH:
            raise ValueError(
                f"{n} elements per wavelength is below the practical minimum "
                f"of {MIN_ELEMENTS_PER_WAVELENGTH}: the dispersion error "
                f"would be around {164.0 / n ** 2:.0f}%. Ask for a lower "
                "f_max instead."
            )
        if n < DEFAULT_ELEMENTS_PER_WAVELENGTH:
            warnings.warn(
                f"{n} elements per wavelength (default is "
                f"{DEFAULT_ELEMENTS_PER_WAVELENGTH}): expect about "
                f"{164.0 / n ** 2:.1f}% error on the highest frequencies.",
                RuntimeWarning,
            )
        self.elements_per_wavelength = n
        if self.analysis not in ("auto", "acoustic", "vibroacoustic",
                                 "structural"):
            raise ValueError(
                "analysis must be 'auto', 'acoustic', 'vibroacoustic' or "
                f"'structural', not {self.analysis!r}."
            )

    @property
    def element_size(self):
        """Target element size [m]: ``c0 / (f_max * elements_per_wavelength)``."""
        return element_size_for(self.fluid.c0, self.f_max,
                                self.elements_per_wavelength)

    @property
    def mesh_path(self):
        """
        Where the mesh is (or will be) written.

        A mesh the user brought is an **input**: with no retagging asked
        for it is used where it lies, and when ``rename``/``drop`` do
        apply, the retagged copy goes to a ``_retagged`` name instead of
        over the original. An explicit ``output_path`` overrides both —
        pointing it at the source file is how one asks for an overwrite.
        """
        if self.output_path is not None:
            return pathlib.Path(self.output_path)
        if isinstance(self.geometry, Library):
            return pathlib.Path(self.work_dir) / f"{self.geometry.name}.msh"

        src = pathlib.Path(self.geometry.path)
        if isinstance(self.geometry, MeshFile):
            if not (self.geometry.rename or self.geometry.drop):
                return src
            out = pathlib.Path(self.work_dir) / f"{src.stem}_retagged.msh"
            while _same_file(out, src):
                out = out.with_name(f"{out.stem}_retagged.msh")
            return out
        return pathlib.Path(self.work_dir) / f"{src.stem}.msh"

    def at(self, **changes):
        """A copy of the spec with some fields replaced (parameter studies)."""
        return replace(self, **changes)

    def describe(self):
        """Printable summary: what will be built, and how finely."""
        h = self.element_size
        lines = [
            "ModelSpec",
            f"  geometry     {self.geometry!r}",
            f"  fluid        {self.fluid.name}: rho0 = {self.fluid.rho0} "
            f"kg/m3, c0 = {self.fluid.c0} m/s",
        ]
        if self.structure is not None:
            s = self.structure
            lines.append(
                f"  structure    {s.name}: t = {s.t * 1e3:.2f} mm, "
                f"E = {s.E / 1e9:.0f} GPa, nu = {s.nu}, "
                f"rho = {s.rho_s} kg/m3, {s.support}"
            )
        lines += [
            f"  f_max        {float(self.f_max):.1f} Hz "
            f"(lambda = {self.fluid.wavelength(self.f_max) * 1e3:.0f} mm)",
            f"  mesh size    {h * 1e3:.1f} mm "
            f"({self.elements_per_wavelength} elements per wavelength)",
            f"  mesh file    {self.mesh_path}",
        ]
        if self.structure is not None:
            lam_b = self.structure.bending_wavelength(self.f_max)
            lines.append(
                f"  bending      lambda_b = {lam_b * 1e3:.0f} mm at f_max"
                + ("  (shorter than the acoustic one: check the panel mesh)"
                   if lam_b < self.fluid.wavelength(self.f_max) else "")
            )
        return "\n".join(lines)

    def __str__(self):
        return self.describe()


def element_size_for(c0, f_max, elements_per_wavelength=DEFAULT_ELEMENTS_PER_WAVELENGTH):
    """
    Element size resolving ``f_max``: ``c0 / (f_max * n)``.

    Parameters
    ----------
    c0 : float
        Speed of sound [m/s].
    f_max : float
        Highest frequency of interest [Hz].
    elements_per_wavelength : int, optional

    Returns
    -------
    float
        Element size [m].
    """
    return float(c0) / (float(f_max) * float(elements_per_wavelength))


# meshing
# Which builder argument counts the elements along which length. The
# spec gives a size in metres, the builders want a number of elements.
_DIVISIONS = {
    "box_cavity": (("Lx", "nx"), ("Ly", "ny"), ("Lz", "nz")),
    "box_with_plate": (("Lx", "nx"), ("Ly", "ny"), ("Lz", "nz")),
    "duct_2d": (("L", "nx"), ("H", "ny")),
    "plate": (("Lx", "nx"), ("Ly", "ny")),
}


def _builder_default(name, argument):
    """Default value of a builder argument, for lengths the user left out."""
    import inspect

    return inspect.signature(GEOMETRIES[name]).parameters[argument].default


def _library_kwargs(source, h):
    """Builder arguments: the user's, plus the element counts from ``h``."""
    kwargs = dict(source.params)
    if source.name == "duct_with_flush_plate":
        # This one takes the size directly, and cuts the duct itself.
        if "element_size" not in kwargs and "f_max" not in kwargs:
            kwargs["element_size"] = h
        return kwargs

    for length_arg, count_arg in _DIVISIONS.get(source.name, ()):
        if count_arg in kwargs:
            continue                    # explicit count wins, for convergence
        length = kwargs.get(length_arg, _builder_default(source.name,
                                                         length_arg))
        kwargs[count_arg] = max(1, int(math.ceil(float(length) / h)))
    return kwargs


def _try_structured(model, dim, h):
    """
    Mesh the imported solid structurally when its shape allows it.

    pyCAFE has kernels for hexahedra and quadrilaterals only, and a
    coupled model needs the panel elements to *be* faces of the fluid
    elements. An unstructured import gives neither: tetrahedra for the
    fluid, quadrilaterals on the surface that share no face with them.
    A six-sided volume can instead be meshed transfinitely, which gives
    hexahedra whose faces are exactly the surface quadrilaterals.

    Returns
    -------
    bool
        Whether the structured setup was applied. When it is not, the
        caller falls back to the unstructured mesh, which is usable for
        a purely acoustic model (CTETRA4) but not for a coupled one.
    """
    import numpy as np

    volumes = model.model.getEntities(dim)
    for _d, tag in volumes:
        faces = model.model.getBoundary([(dim, tag)], oriented=False,
                                        combined=False)
        if len(faces) != 6:
            return False
        for _fd, face in faces:
            curves = model.model.getBoundary([(dim - 1, abs(face))],
                                             oriented=False, combined=False)
            if len(curves) != 4:
                return False

    for _d, curve in model.model.getEntities(1):
        bbox = np.array(model.model.getBoundingBox(1, curve), dtype=float)
        length = float(np.linalg.norm(bbox[3:] - bbox[:3]))
        model.mesh.setTransfiniteCurve(curve, max(2, int(math.ceil(length / h)) + 1))
    for _d, surface in model.model.getEntities(2):
        model.mesh.setTransfiniteSurface(surface)
        model.mesh.setRecombine(2, surface)
    for _d, volume in volumes:
        model.mesh.setTransfiniteVolume(volume)
    return True


def _clamp_tags(model, source, dim, structure):
    """Entities holding the supported nodes: the user's, or the edge."""
    if source.clamp == "auto":
        return sorted({
            abs(t) for _d, t in model.model.getBoundary(
                [(dim - 1, t) for t in structure],
                oriented=False, combined=True,
            )
        })
    return [int(t) for t in source.clamp]


def _mesh_structure_alone(source, spec, out_path):
    """
    Mesh the structural surfaces on their own, in quadrilaterals.

    The second pass of :func:`_mesh_from_cad_split`. Nothing here knows
    about the fluid: the surfaces are recombined into quadrilaterals —
    transfinitely where they have four sides, so the panel comes out
    structured — and only the structural groups are tagged. The nodes
    are new, and stay new: merging them with the fluid ones is what the
    interpolation coupling is there to avoid needing.
    """
    import numpy as np

    dim = int(source.dim)
    h = float(source.structure_element_size or spec.element_size)
    structure = tuple(source.structure)

    with GmshModel(out_path.stem, verbose=spec.verbose, recombine=True) as m:
        import_cad(m, source.path, source.units, verbose=spec.verbose)
        m.size(h)

        for tag in structure:
            # Transfinite needs a four-sided patch bounded by a single
            # curve loop: a periodic surface (a full cylinder side) also
            # shows four boundary curves, but two of them close a second
            # loop, and Gmsh rejects it.
            try:
                loops, loop_curves = m.model.getCurveLoops(tag)
            except Exception:
                continue
            if len(loops) != 1 or len(loop_curves[0]) != 4:
                continue
            curves = [(1, c) for c in np.asarray(loop_curves[0]).ravel()]
            for _d, curve in curves:
                bbox = np.array(m.model.getBoundingBox(1, abs(curve)),
                                dtype=float)
                length = float(np.linalg.norm(bbox[3:] - bbox[:3]))
                m.mesh.setTransfiniteCurve(abs(curve),
                                           max(2, int(math.ceil(length / h)) + 1))
            m.mesh.setTransfiniteSurface(tag)
            m.mesh.setRecombine(dim - 1, tag)

        m.physical(dim - 1, list(structure), "plate")
        if spec.structure is not None and spec.structure.support != "free":
            clamp = _clamp_tags(m, source, dim, structure)
            if clamp:
                m.physical(dim - 2, clamp, "plate_clamp")

        m.generate(dim - 1, kernel="occ")
        out = m.write(out_path)
        if spec.verbose:
            print(m.summary())
    return out


def _mesh_from_cad_split(source, spec, out_path):
    """
    Two meshing passes, one per domain, merged into a single file.

    The fluid is meshed without the structure in it, the structure
    without the fluid around it, and
    :func:`~pycafe.create_geom.merge.merge_meshes` puts the two in one
    file with disjoint node numbering. The interface is non-conforming
    by construction — that is the point — and the coupling is assembled
    by interpolation.
    """
    from pycafe.create_geom.merge import merge_meshes

    fluid_path = out_path.with_name(out_path.stem + "_fluid.msh")
    plate_path = out_path.with_name(out_path.stem + "_plate.msh")

    fluid_only = replace(source, structure=(), independent_structure=False)
    _mesh_from_cad(fluid_only, spec, fluid_path)
    _mesh_structure_alone(source, spec, plate_path)

    return merge_meshes([fluid_path, plate_path], out_path,
                        verbose=spec.verbose)


def _drop_unused_volumes(model, dim, source, verbose=False):
    """
    Remove the solids that carry no role, before they are meshed.

    A STEP file of a coupled problem usually holds both bodies: the
    fluid, and the thing sitting in it. pyCAFE assembles the second one
    as a shell on its **surface**, so its interior is meshed for nothing
    -- and worse than nothing, since those nodes then appear in the file
    with no element to give them an equation, and the acoustic system
    comes out singular.

    The surfaces that still bound the fluid stay, and so do the ones the
    user named -- the wet side of a shell is usually one of them. What
    goes with the solid is the rest: a STEP assembly describes the
    contact surface once per body, and the copy that belonged to the
    removed solid would otherwise be meshed on its own, a second set of
    nodes on the same geometry attached to nothing.

    Parameters
    ----------
    model : GmshModel
        Open model, with the CAD already imported.
    dim : int
        Dimension of the fluid domain.
    source : CadFile
        The declaration; only the solids named in ``fluid`` are kept,
        and only the surfaces named in ``structure`` or ``boundaries``
        survive the loss of their solid.
    verbose : bool, optional

    Returns
    -------
    tuple of (list of int, list of int)
        Tags of the removed solids and of the removed surfaces.
    """
    keep = {int(t) for t in source.fluid}
    present = [tag for _d, tag in model.model.getEntities(dim)]
    unused = [tag for tag in present if tag not in keep]
    if not unused or len(present) == len(unused):
        return [], []

    # The geometry lives in the OCC kernel, so it is OCC that has to
    # forget it: removing it from the model alone is undone by the next
    # synchronize.
    model.occ.remove([(dim, tag) for tag in unused], recursive=False)
    model.occ.synchronize()

    # A surface survives if it still bounds a solid, or if it was named.
    bounding = {abs(tag) for _d, tag in model.model.getBoundary(
        [(dim, tag) for tag in sorted(keep)], oriented=False, combined=False,
    )}
    named = {int(t) for t in source.structure}
    for tags in source.boundaries.values():
        named |= {int(t) for t in tags}

    loose = [tag for _d, tag in model.model.getEntities(dim - 1)
             if tag not in bounding and tag not in named]
    if loose:
        model.occ.remove([(dim - 1, tag) for tag in loose], recursive=True)
        model.occ.synchronize()

    if verbose:
        print(f"solids {unused} have no role in the model and are not "
              f"meshed; {len(loose)} surface(s) left without a solid went "
              f"with them")
    return unused, loose


def _mesh_from_cad(source, spec, out_path):
    """Import, size, tag by the tags the user gave, mesh, write."""
    dim = int(source.dim)
    h = spec.element_size

    if source.independent_structure and tuple(source.structure):
        return _mesh_from_cad_split(source, spec, out_path)

    with GmshModel(out_path.stem, verbose=spec.verbose,
                   recombine=source.recombine) as m:
        import_cad(m, source.path, source.units, verbose=spec.verbose)
        _drop_unused_volumes(m, dim, source, verbose=spec.verbose)
        m.size(h)

        structured = _try_structured(m, dim, h) if source.recombine else False
        if not structured and tuple(source.structure):
            warnings.warn(
                f"{pathlib.Path(source.path).name} is not a six-sided solid, "
                "so it is meshed unstructured: the fluid comes out "
                "tetrahedral and its faces do not match the panel "
                "quadrilaterals. Cut the geometry into boxes (see "
                "duct_with_flush_plate) for a coupled model.",
                RuntimeWarning,
            )

        rules = [by_tag(dim, tuple(source.fluid), "fluid")]
        structure = tuple(source.structure)
        if structure:
            rules.append(by_tag(dim - 1, structure, "plate"))
        for name, tags in source.boundaries.items():
            rules.append(by_tag(dim - 1, tuple(tags), name))
        if source.walls:
            rules.append(rest(dim - 1, source.walls))

        assign_groups(m, rules)

        # The support: the boundary of the structure entities, unless the
        # user named it. Only tagged when there is something to hold.
        if structure and spec.structure is not None \
                and spec.structure.support != "free":
            if source.clamp == "auto":
                clamp_tags = sorted({
                    abs(t) for _d, t in m.model.getBoundary(
                        [(dim - 1, t) for t in structure],
                        oriented=False, combined=True,
                    )
                })
            else:
                clamp_tags = [int(t) for t in source.clamp]
            if clamp_tags:
                m.physical(dim - 2, clamp_tags, "plate_clamp")

        m.generate(dim, kernel="occ")
        out = m.write(out_path)
        if spec.verbose:
            print(m.summary())
    return out


def _mesh_from_bdf(source, spec, out_path):
    """Read the deck, name its property ids, write the pyCAFE mesh."""
    from pycafe.create_geom.nastran import bdf_to_mesh

    return bdf_to_mesh(
        source.path, out_path,
        fluid=source.fluid, structure=source.structure,
        boundaries=source.boundaries, surface_rules=source.surface_rules,
        walls=source.walls, clamp=source.clamp, units=source.units,
        deck=source.deck, validate=False, verbose=spec.verbose,
    )


def materials_from_deck(spec):
    """
    Fill the materials the spec left out with what the deck states.

    A bulk data file carries the thickness and the material cards, so a
    :class:`NastranFile` model can state them once, in the deck, instead
    of twice. The rule is the one that keeps the spec authoritative:

    * a field the spec **gives** wins, and a deck that disagrees with it
      is reported as a warning rather than silently overridden;
    * a field the spec **leaves out** — no :class:`Structure`, or the
      default :data:`AIR` still in place — is taken from the deck.

    Parameters
    ----------
    spec : ModelSpec
        Its geometry must be a :class:`NastranFile`.

    Returns
    -------
    fluid : Fluid
    structure : Structure or None
    notes : list of str
        What was taken from the deck, one line each, for reporting.

    Warns
    -----
    RuntimeWarning
        When a material given in the spec and one written in the deck
        describe different matter.
    """
    source = spec.geometry
    deck = source.deck
    notes = []
    fluid, structure = spec.fluid, spec.structure

    deck_fluid = next(
        (m for m in (deck.material_of(pid) for pid in source.fluid)
         if m is not None and m.is_fluid and m.rho and m.c0), None
    )
    if deck_fluid is not None:
        if fluid is AIR:
            fluid = Fluid(rho0=float(deck_fluid.rho), c0=float(deck_fluid.c0),
                          name=f"{deck_fluid.card} {deck_fluid.mid}")
            notes.append(
                f"fluid from the deck: rho0 = {fluid.rho0} kg/m3, "
                f"c0 = {fluid.c0} m/s ({fluid.name})"
            )
        elif (abs(deck_fluid.rho - fluid.rho0) > 1e-3 * fluid.rho0
                or abs(deck_fluid.c0 - fluid.c0) > 1e-3 * fluid.c0):
            warnings.warn(
                f"the spec asks for {fluid.name} (rho0 = {fluid.rho0}, "
                f"c0 = {fluid.c0}) while {deck.path.name} states "
                f"rho = {deck_fluid.rho}, c0 = {deck_fluid.c0} on "
                f"{deck_fluid.card} {deck_fluid.mid}. The spec is used.",
                RuntimeWarning,
            )

    if structure is None:
        for pid in source.structure:
            prop = deck.properties.get(int(pid))
            mat = deck.material_of(pid)
            if prop is None or prop.thickness is None:
                continue
            kwargs = dict(t=float(prop.thickness),
                          name=f"PSHELL {prop.pid}")
            if mat is not None and not mat.is_fluid:
                for key, value in (("E", mat.E), ("nu", mat.nu),
                                   ("rho_s", mat.rho)):
                    if value is not None:
                        kwargs[key] = float(value)
            structure = Structure(**kwargs)
            notes.append(
                f"structure from the deck: t = {structure.t * 1e3:.2f} mm, "
                f"E = {structure.E / 1e9:.0f} GPa, nu = {structure.nu}, "
                f"rho = {structure.rho_s} kg/m3 ({structure.name})"
            )
            break

    return fluid, structure, notes


def build_mesh(spec):
    """
    Build (or fetch) the mesh of a spec and check it.

    Parameters
    ----------
    spec : ModelSpec

    Returns
    -------
    path : pathlib.Path
        The ``.msh`` actually used.
    report : MeshReport
        Result of :func:`~pycafe.create_geom.validation.validate_mesh`,
        run at ``spec.f_max`` so the resolution note is about the
        frequencies asked for.

    Raises
    ------
    RuntimeError
        If the mesh does not satisfy the pyCAFE contract.
    """
    source = spec.geometry
    out_path = spec.mesh_path
    h = spec.element_size

    if isinstance(source, Library):
        kwargs = _library_kwargs(source, h)
        kwargs.setdefault("output_path", out_path)
        kwargs.setdefault("verbose", spec.verbose)
        result = GEOMETRIES[source.name](**kwargs)
        # duct_with_flush_plate returns (path, report) when it validates.
        path = pathlib.Path(result[0] if isinstance(result, tuple) else result)

    elif isinstance(source, CadFile):
        path = _mesh_from_cad(source, spec, out_path)

    elif isinstance(source, NastranFile):
        path = _mesh_from_bdf(source, spec, out_path)

    elif isinstance(source, MeshFile):
        if source.rename or source.drop:
            # out_path is never source.path unless the user named it
            # explicitly (see ModelSpec.mesh_path): retagging writes a
            # copy, it does not consume the file it was given.
            path = retag_mesh(source.path, rename=source.rename,
                              drop=source.drop, output_path=out_path,
                              validate=False, verbose=spec.verbose)
        else:
            path = pathlib.Path(source.path)

    else:
        raise TypeError(
            "spec.geometry must be a Library, CadFile, NastranFile or "
            f"MeshFile, not {type(source).__name__}."
        )

    report = validate_mesh(str(path), analysis=spec.analysis,
                           c0=spec.fluid.c0, f_max=spec.f_max)
    if spec.verbose:
        print(report)
    if not report.ok:
        raise RuntimeError(
            f"the mesh does not satisfy the pyCAFE contract:\n{report}"
        )
    return path, report


def describe_domains(report, fluid=None, structure=None):
    """
    Which physical group is which domain, and which material fills it.

    The check to read between "a file was loaded" and "a system was
    assembled": the mesh says *which* groups are there and what role
    each one plays, the spec says *what they are made of*, and this puts
    the two side by side. A vibroacoustic run needs both domains and a
    material for each; anything missing is named here rather than
    further down, where it turns into a KeyError or, worse, a silent
    default.

    Parameters
    ----------
    report : MeshReport
        From :func:`build_mesh` or
        :func:`~pycafe.create_geom.validation.validate_mesh`.
    fluid : Fluid, optional
        The material of the acoustic domain.
    structure : Structure, optional
        The material and thickness of the structural domain.

    Returns
    -------
    str
        Printable, one line per domain.

    Examples
    --------
    >>> path, report = build_mesh(spec)                  # doctest: +SKIP
    >>> print(describe_domains(report, spec.fluid, spec.structure))
    ... # doctest: +SKIP
    """
    roles = report.roles
    analysis = report.analysis
    lines = [f"domains in {pathlib.Path(report.source).name} "
             f"({analysis} analysis)"]

    def row(what, group, material):
        return f"  {what:<11s} group {group!r:<16s} <- {material}"

    if "fluid" in roles:
        lines.append(row(
            "fluid", roles["fluid"],
            f"{fluid.name}: rho0 = {fluid.rho0} kg/m3, c0 = {fluid.c0} m/s"
            if fluid is not None else
            "no fluid given (Fluid(rho0, c0), or the default AIR)",
        ))
    elif analysis in ("acoustic", "vibroacoustic"):
        lines.append("  fluid       missing: no group named fluid / acoustic "
                     "/ domain in this mesh")

    if "structure" in roles:
        lines.append(row(
            "structure", roles["structure"],
            f"{structure.name}: t = {structure.t * 1e3:.2f} mm, "
            f"E = {structure.E / 1e9:.0f} GPa, nu = {structure.nu}, "
            f"rho = {structure.rho_s} kg/m3"
            if structure is not None else
            "no structure given: pass structure=aluminium(t=2e-3)",
        ))
    elif analysis in ("vibroacoustic", "structural"):
        lines.append("  structure   missing: no group named plate / structure "
                     "/ shell in this mesh")

    if "clamp" in roles:
        support = structure.support if structure is not None else "unset"
        lines.append(row("support", roles["clamp"],
                         f"{support} (the mesh says which nodes are held, "
                         "Structure.support says how)"))

    for note in report.notes:
        if note.startswith("selectable boundaries"):
            lines.append(f"  {note}")
    return "\n".join(lines)


# assembling
def preview(spec, **kwargs):
    """
    Build the mesh and look at it — no matrices, no solve.

    The step to run before anything else: it answers "is this the model I
    meant?", which no number downstream answers. Same as
    :func:`build_mesh` plus the picture.

    Parameters
    ----------
    spec : ModelSpec
    **kwargs
        Passed to :func:`~pycafe.create_geom.preview.plot_geometry`
        (``elev``, ``azim``, ``figsize``, ``show_boundaries``, ...).

    Returns
    -------
    path : pathlib.Path
    report : MeshReport
    """
    from pycafe.create_geom.preview import plot_geometry
    from pycafe.create_geom.visualize_mesh import load_mesh_with_groups

    path, report = build_mesh(spec)
    nodes, _elements, _boundaries, groups = load_mesh_with_groups(
        str(path), verbose=False
    )
    kwargs.setdefault("title", f"{path.name} — {report.analysis} model")
    plot_geometry(nodes, groups, **kwargs)
    print(report)
    return path, report


def build_model(spec, bc=None, *, show=True, plot_kwargs=None):
    """
    Mesh, load, validate and assemble — the whole front end of a run.

    The geometry is drawn as soon as it exists, **before** the matrices
    are assembled: a panel on the wrong wall or a CAD file read in the
    wrong unit shows up in the picture and in none of the numbers.

    Parameters
    ----------
    spec : ModelSpec
    bc : AcousticBC, optional
        Boundary conditions for an acoustic model. Default: none at all,
        i.e. hard walls everywhere. Ignored for a vibroacoustic model,
        which is assembled without boundary conditions.
    show : bool, optional
        Draw the geometry, coloured by role, before assembling. Default
        True. Set it to ``False`` in a batch run or a test, where nobody
        is looking at the figure.
    plot_kwargs : dict, optional
        Passed to :func:`~pycafe.create_geom.preview.plot_geometry`.

    Returns
    -------
    dict
        ``spec``, ``mesh_path``, ``report``, ``analysis``, the
        materials actually used (``fluid``, ``structure`` — the same as
        the spec's, except where a :class:`NastranFile` deck filled in
        what the spec left out, see :func:`materials_from_deck`), the
        loaded mesh (``nodes``, ``elements``, ``boundaries``,
        ``groups``) and ``system``, the output of
        :func:`~pycafe.core.prepare_acoustic_system.prepare_acoustic_system`
        or
        :func:`~pycafe.core.prepare_vibroacoustic_system.prepare_vibroacoustic_system`.

    Raises
    ------
    ValueError
        If the mesh has a structural group but the spec has no
        :class:`Structure`.

    Examples
    --------
    >>> spec = ModelSpec(geometry=Library("box_cavity", Lx=0.4, Ly=0.3,
    ...                                   Lz=0.5), f_max=300.0)
    >>> model = build_model(spec)                      # doctest: +SKIP
    >>> model["system"]["K"].shape                     # doctest: +SKIP
    (1080, 1080)
    """
    from pycafe.create_geom.visualize_mesh import load_mesh_with_groups

    path, report = build_mesh(spec)
    nodes, elements, boundaries, groups = load_mesh_with_groups(
        str(path), verbose=False
    )

    if show:
        from pycafe.create_geom.preview import plot_geometry

        options = dict(plot_kwargs or {})
        options.setdefault("title", f"{path.name} — {report.analysis} model")
        plot_geometry(nodes, groups, **options)

    fluid, structure = spec.fluid, spec.structure
    if isinstance(spec.geometry, NastranFile):
        fluid, structure, notes = materials_from_deck(spec)
        for note in notes:
            print(note)

    analysis = report.analysis if spec.analysis == "auto" else spec.analysis
    if analysis in ("vibroacoustic", "structural") and structure is None:
        raise ValueError(
            f"the mesh holds a structural group ('{report.roles.get('structure')}') "
            "but the spec has no Structure: give one, e.g. "
            "structure=aluminium(t=2e-3)."
        )

    if analysis == "vibroacoustic":
        from pycafe.core.prepare_vibroacoustic_system import (
            prepare_vibroacoustic_system,
        )

        s = structure
        system = prepare_vibroacoustic_system(
            nodes=nodes, groups=groups,
            rho0=fluid.rho0, c0=fluid.c0,
            t=s.t, rho_s=s.rho_s, E=s.E, nu=s.nu, nsm=s.nsm,
            support=s.support,
            build_coupling=spec.build_coupling,
        )
    elif analysis == "acoustic":
        from pycafe.boundary_condition.acoustic_bc import AcousticBC
        from pycafe.core.prepare_acoustic_system import prepare_acoustic_system

        system = prepare_acoustic_system(
            nodes=nodes, elements=elements, boundaries=boundaries,
            rho=fluid.rho0, c0=fluid.c0,
            bc=AcousticBC() if bc is None else bc,
            groups=groups, pml=spec.pml,
        )
    else:
        raise NotImplementedError(
            f"'{analysis}' models have no preparation step yet; assemble the "
            "structural domain with build_KM_structural_domain."
        )

    return {
        "spec": spec,
        "mesh_path": path,
        "report": report,
        "analysis": analysis,
        "fluid": fluid,
        "structure": structure,
        "nodes": nodes,
        "elements": elements,
        "boundaries": boundaries,
        "groups": groups,
        "system": system,
    }
