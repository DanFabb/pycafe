r"""
Reading a Nastran bulk data file: the mesh through Gmsh, the rest here.

A ``.bdf`` is the most common way a model arrives from outside, and it is
a text file that already contains everything an analysis needs — nodes,
elements, thicknesses, materials, supports. Gmsh reads the geometry part
of it, and only that part:

* ``GRID`` and the element cards become a mesh, with **one entity per
  property id**: ``CQUAD4  2  202  ...`` lands in entity ``(2, 202)``.
  So the property ids of the deck are the tags to quote when saying what
  is fluid and what is structure — the same conversation
  :func:`~pycafe.create_geom.external.inspect_cad` starts for a STEP
  file;
* ``PSHELL``, ``PSOLID``, ``MAT1``, ``MAT10``, ``SPC``/``SPC1`` are
  dropped on the floor, and with them the thickness, the material and
  the supported nodes;
* physical groups do not exist in the format at all, so nothing in the
  file says "this face is the opening".

This module reads the second bullet itself, with a small bulk data
parser, so that :func:`inspect_bdf` can show *why* a property id looks
like a fluid (a ``PSOLID`` pointing at a ``MAT10``) rather than leaving
it to be guessed:

.. code-block:: python

    from pycafe.create_geom import inspect_bdf, bdf_to_mesh

    inspect_bdf("cabin.bdf")           # what is in it, and what it looks like
    bdf_to_mesh("cabin.bdf", "cabin.msh",
                fluid=[101], structure=[202],
                surface_rules=[on_plane("z", 0.0, name="opening")])

The roles are still stated, never inferred: the deck is read to *propose*
them and to fill in the material data, and the caller confirms.

The third bullet is what :func:`bdf_to_mesh` has to build. An acoustic
deck usually holds volume elements only, so there is no surface to name
and no face for an impedance or a prescribed velocity to act on. The
faces with a single element behind them are therefore extracted from the
fluid mesh and turned into named surface groups by the same geometric
rules used everywhere else (:func:`~pycafe.create_geom.external.on_plane`,
:func:`~pycafe.create_geom.external.in_box`); whatever is left becomes
``rigid_walls``, which is the natural boundary condition anyway.

Two things worth knowing about the Gmsh reader:

* it counts fields rather than columns on a **free field** line, and a
  line written with more than the eight data fields the format allows
  does not stop it: the extra values are read on as nodes, and the
  elements come out with the wrong connectivity or the read aborts on
  ``Wrong node index``. A card properly broken over two lines is read
  correctly, in both the blank and the ``+CONT`` continuation styles;
  it is the overfull line that :func:`read_bdf` rejects up front;
* ``GRID`` points no element refers to are dropped, which is what one
  wants: they would otherwise appear in the mesh with no equation.

See Also
--------
pycafe.create_geom.external : The same problem for CAD files and meshes.
pycafe.core.model_spec : ``NastranFile``, the deck as a model input.
"""

import pathlib
import warnings
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from .external import EntityInfo
from .gmsh_workflow import GmshModel

# Element cards Gmsh turns into elements. Anything else in the deck
# (springs, rigid elements, concentrated masses) is ignored by the
# reader, and ignoring it here too keeps the two views consistent.
ELEMENT_CARDS = (
    "CTRIA3", "CTRIA6", "CQUAD4", "CQUAD8",
    "CTETRA", "CHEXA", "CPENTA", "CPYRAM",
)

# Dimension of the domain each element card belongs to.
CARD_DIMENSION = {
    "CTRIA3": 2, "CTRIA6": 2, "CQUAD4": 2, "CQUAD8": 2,
    "CTETRA": 3, "CHEXA": 3, "CPENTA": 3, "CPYRAM": 3,
}


@dataclass(frozen=True)
class NastranProperty:
    """
    One ``PSHELL`` or ``PSOLID`` card: what a property id stands for.

    Parameters
    ----------
    pid : int
        Property id — also the Gmsh entity tag of every element that
        references it.
    card : str
        ``"PSHELL"`` or ``"PSOLID"``.
    mid : int or None
        Material id of the first material field.
    thickness : float or None
        ``PSHELL`` membrane thickness ``T``.
    fluid_flag : bool
        ``PSOLID`` with ``FCTN = PFLUID``, Nastran's own marking of an
        acoustic solid.
    """

    pid: int
    card: str
    mid: Optional[int] = None
    thickness: Optional[float] = None
    fluid_flag: bool = False


@dataclass(frozen=True)
class NastranMaterial:
    """
    One ``MAT1`` (elastic) or ``MAT10`` (acoustic fluid) card.

    ``MAT10`` states the bulk modulus, the density and the speed of
    sound, any two of which determine the third through
    ``bulk = rho c0^2``; the missing one is completed here so that the
    card can be compared with a :class:`~pycafe.core.model_spec.Fluid`
    whatever the analyst chose to write.

    Parameters
    ----------
    mid : int
    card : str
        ``"MAT1"`` or ``"MAT10"``.
    E, nu, rho : float or None
        Young's modulus [Pa], Poisson ratio, density [kg/m^3].
    bulk : float or None
        Bulk modulus [Pa], ``MAT10`` only.
    c0 : float or None
        Speed of sound [m/s], ``MAT10`` only.
    """

    mid: int
    card: str
    E: Optional[float] = None
    nu: Optional[float] = None
    rho: Optional[float] = None
    bulk: Optional[float] = None
    c0: Optional[float] = None

    @property
    def is_fluid(self):
        """True for a ``MAT10``, i.e. a material with no shear stiffness."""
        return self.card == "MAT10"


@dataclass(frozen=True)
class SpcSet:
    """
    One single-point constraint set: which nodes are held, and in what.

    Parameters
    ----------
    sid : int
        Constraint set id, the number a subcase selects with ``SPC =``.
    dofs : str
        Nastran component string, digits 1..6 (``"123456"`` for a
        clamped edge, ``"123"`` for a pinned one). pyCAFE does not read
        it: the support is chosen by ``Structure.support``, and the
        string is reported so that a disagreement is visible.
    nodes : tuple of int
        Grid ids of the set.
    """

    sid: int
    dofs: str
    nodes: Tuple[int, ...]


@dataclass
class BdfDeck:
    """
    What a bulk data file says beyond its mesh.

    Returned by :func:`read_bdf`. The mesh itself is not here — Gmsh
    reads it — but the property ids are, and they are the link between
    the two views: a Gmsh entity tag *is* a property id.

    Attributes
    ----------
    path : pathlib.Path
    properties : dict
        ``{pid: NastranProperty}``.
    materials : dict
        ``{mid: NastranMaterial}``.
    elements : dict
        ``{pid: Counter({card: count})}`` — how many elements of which
        card reference each property id.
    constraints : dict
        ``{sid: SpcSet}``.
    n_grids : int
        ``GRID`` cards in the file, before Gmsh drops the unused ones.
    unsupported : collections.Counter
        Cards Gmsh will not turn into elements, with their counts. Not
        an error; a note on what the mesh will be missing.
    """

    path: pathlib.Path
    properties: Dict[int, NastranProperty] = field(default_factory=dict)
    materials: Dict[int, NastranMaterial] = field(default_factory=dict)
    elements: Dict[int, Counter] = field(default_factory=dict)
    constraints: Dict[int, SpcSet] = field(default_factory=dict)
    n_grids: int = 0
    unsupported: Counter = field(default_factory=Counter)

    def material_of(self, pid):
        """Material behind a property id, or ``None``."""
        prop = self.properties.get(int(pid))
        if prop is None or prop.mid is None:
            return None
        return self.materials.get(prop.mid)

    def dimension_of(self, pid):
        """Domain dimension of the elements carrying ``pid``, or ``None``."""
        cards = self.elements.get(int(pid))
        if not cards:
            return None
        dims = {CARD_DIMENSION[c] for c in cards if c in CARD_DIMENSION}
        return dims.pop() if len(dims) == 1 else None

    def suggest_roles(self):
        """
        Which property id looks like which pyCAFE role, and on what evidence.

        A proposal, not a decision: nothing downstream reads it, and
        :func:`bdf_to_mesh` still asks for the ids explicitly. The
        evidence is the property and material cards, in this order:

        ===================================  =========================
        ``PSOLID`` with ``FCTN = PFLUID``    fluid, said by the deck
        ``PSOLID`` pointing at a ``MAT10``   fluid, said by the material
        3D elements, no property card        fluid, by dimension alone
        ``PSHELL``                           structure
        ``PSOLID`` pointing at a ``MAT1``    an elastic solid: pyCAFE
                                             has no kernel for it
        ===================================  =========================

        Returns
        -------
        dict
            ``{role: [(pid, reason), ...]}`` with roles ``"fluid"``,
            ``"structure"`` and ``"unusable"``.
        """
        out = {"fluid": [], "structure": [], "unusable": []}
        for pid in sorted(self.elements):
            prop = self.properties.get(pid)
            mat = self.material_of(pid)
            dim = self.dimension_of(pid)

            if prop is None:
                where = "fluid" if dim == 3 else "structure"
                out[where].append(
                    (pid, f"no property card, {dim}D elements only")
                )
            elif prop.card == "PSOLID":
                if prop.fluid_flag:
                    out["fluid"].append((pid, "PSOLID with FCTN = PFLUID"))
                elif mat is not None and mat.is_fluid:
                    out["fluid"].append(
                        (pid, f"PSOLID on {mat.card} {mat.mid} "
                              f"(rho = {mat.rho}, c0 = {mat.c0})")
                    )
                elif mat is None:
                    out["fluid"].append((pid, "PSOLID, material not in the deck"))
                else:
                    out["unusable"].append(
                        (pid, f"PSOLID on {mat.card} {mat.mid}: an elastic "
                              "solid, and pyCAFE assembles structures as "
                              "shells only")
                    )
            elif prop.card == "PSHELL":
                t = "" if prop.thickness is None else f", t = {prop.thickness}"
                out["structure"].append((pid, f"PSHELL{t}"))
        return out

    def describe(self):
        """Printable summary of the cards that were read."""
        lines = [f"{self.path.name}: {self.n_grids} GRID, "
                 f"{sum(sum(c.values()) for c in self.elements.values())} "
                 "elements"]
        for pid in sorted(self.elements):
            cards = ", ".join(f"{n} x {c}"
                              for c, n in sorted(self.elements[pid].items()))
            prop = self.properties.get(pid)
            mat = self.material_of(pid)
            what = "no property card" if prop is None else prop.card
            if prop is not None and prop.thickness is not None:
                what += f" t = {prop.thickness:g}"
            if mat is not None:
                what += f" -> {mat.card} {mat.mid}"
            lines.append(f"  pid {pid:<8d} {cards:<22} {what}")
        for sid, spc in sorted(self.constraints.items()):
            lines.append(f"  SPC set {sid}: {len(spc.nodes)} node(s), "
                         f"components {spc.dofs}")
        if self.unsupported:
            lines.append("  cards Gmsh does not mesh: " + ", ".join(
                f"{n} x {c}" for c, n in sorted(self.unsupported.items())
            ))
        return "\n".join(lines)

    def __str__(self):
        return self.describe()


def _to_float(text):
    """
    One bulk data real field, including Nastran's implicit exponent.

    ``1.5+3`` and ``1.5-3`` are legal ways of writing ``1.5e3`` and
    ``1.5e-3``: the ``E`` may be left out, so a sign after a digit is an
    exponent, not a subtraction.
    """
    text = str(text).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass
    body = text[1:] if text[0] in "+-" else text
    for i, ch in enumerate(body):
        if ch in "+-" and i > 0 and body[i - 1] not in "eEdD":
            head = text[: len(text) - len(body) + i]
            return float(f"{head}e{body[i:]}")
    return float(text.replace("D", "E").replace("d", "e"))


def _to_int(text):
    """One bulk data integer field, or ``None`` when it is blank."""
    text = str(text).strip()
    return int(text) if text else None


def _split_line(line):
    """
    Split one bulk data line into its fields.

    Returns
    -------
    fields : list of str
        The name field first, then the eight (small/free) or four
        (large) data fields of the line.
    free : bool
        Whether the line is written in free field format.
    """
    line = line.split("$", 1)[0].rstrip("\n").rstrip()
    if "," in line:
        return line.split(","), True
    name = line[:8]
    if name.rstrip().endswith("*"):
        width, count = 16, 4
    else:
        width, count = 8, 8
    return ([name] + [line[8 + width * i: 8 + width * (i + 1)]
                      for i in range(count)], False)


def _is_continuation(fields):
    """Does this line continue the previous card rather than start one?"""
    first = fields[0].strip()
    return first == "" or first[0] in "+*"


def _is_overfull(fields, free):
    """
    Does this free field line carry more values than the format allows?

    A bulk data line holds eight data fields, whatever the format; a
    ninth one is the continuation marker and nothing else. Small field
    lines are read by column, so an over-long one is simply truncated,
    but a free field line is read by counting commas: the Gmsh reader
    goes on consuming the extra values as if they were still part of
    the card, and an element ends up with a node index that does not
    exist. Written by hand, or by an exporter with an off-by-one, this
    is the one shape of deck that reads as a silent corruption.
    """
    if not free:
        return False
    data = list(fields[1:])
    while data and not data[-1].strip():
        data.pop()
    if len(data) <= 8:
        return False
    return not data[8].strip().startswith(("+", "*"))


def _cards(path):
    """
    Yield the bulk data cards of a file, continuations already joined.

    Yields
    ------
    name : str
        Card name, upper case, without the large field ``*``.
    fields : list of str
        Data fields, the name excluded, in card order.
    free : bool
        Whether any line of the card is free field.
    overfull : bool
        Whether any line of the card carries more data fields than the
        format allows; see :func:`_is_overfull`.
    """
    with open(path, "r", errors="replace") as stream:
        lines = stream.readlines()

    # A deck exported on its own is all bulk data; a full run also has
    # an executive and a case control section, which stop at BEGIN BULK
    # and hold cards (SET, SPC = 10) that would be misread as bulk ones.
    start = next((i + 1 for i, l in enumerate(lines)
                  if l.strip().upper().startswith("BEGIN BULK")), 0)

    name, fields, free, overfull = None, [], False, False

    for raw in lines[start:]:
        if raw.strip().upper().startswith("ENDDATA"):
            break
        if not raw.strip() or raw.lstrip().startswith("$"):
            continue

        line_fields, line_free = _split_line(raw)
        if not any(f.strip() for f in line_fields):
            continue
        line_overfull = _is_overfull(line_fields, line_free)

        if _is_continuation(line_fields) and name is not None:
            fields.extend(line_fields[1:])
            free = free or line_free
            overfull = overfull or line_overfull
            continue

        if name is not None:
            yield name, fields, free, overfull
        name = line_fields[0].strip().rstrip("*").upper()
        fields = list(line_fields[1:])
        free, overfull = line_free, line_overfull

    if name is not None:
        yield name, fields, free, overfull


def read_bdf(path, *, strict=True):
    """
    Read the parts of a bulk data file that Gmsh throws away.

    Parameters
    ----------
    path : str or Path
    strict : bool, optional
        Raise on a deck Gmsh cannot read (see below). ``False`` turns
        it into a warning, for looking at a file that will not mesh.

    Returns
    -------
    BdfDeck

    Raises
    ------
    ValueError
        When an element card is written in free field format with more
        than the eight data fields a bulk data line holds. The Gmsh
        reader keeps reading the extra values as nodes, so the elements
        come out with the wrong connectivity or the read aborts on
        ``Wrong node index``; both are caught here rather than after a
        wrong answer. Properly continued cards are fine.

    Examples
    --------
    >>> deck = read_bdf("cabin.bdf")            # doctest: +SKIP
    >>> deck.suggest_roles()["fluid"]           # doctest: +SKIP
    [(101, 'PSOLID on MAT10 1 (rho = 1.204, c0 = 343.0)')]
    """
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No bulk data file at {path}.")

    deck = BdfDeck(path=path)
    broken = Counter()
    spc_nodes = {}
    spc_dofs = {}

    for name, fields, free, overfull in _cards(path):
        if name == "GRID":
            deck.n_grids += 1

        elif name in ELEMENT_CARDS:
            if overfull:
                broken[name] += 1
            pid = _to_int(fields[1]) if len(fields) > 1 else None
            if pid is not None:
                deck.elements.setdefault(pid, Counter())[name] += 1

        elif name == "PSHELL":
            pid = _to_int(fields[0])
            if pid is not None:
                deck.properties[pid] = NastranProperty(
                    pid=pid, card="PSHELL",
                    mid=_to_int(fields[1]) if len(fields) > 1 else None,
                    thickness=_to_float(fields[2]) if len(fields) > 2 else None,
                )

        elif name == "PSOLID":
            pid = _to_int(fields[0])
            if pid is not None:
                fctn = fields[6].strip().upper() if len(fields) > 6 else ""
                deck.properties[pid] = NastranProperty(
                    pid=pid, card="PSOLID",
                    mid=_to_int(fields[1]) if len(fields) > 1 else None,
                    fluid_flag=(fctn == "PFLUID"),
                )

        elif name == "MAT1":
            mid = _to_int(fields[0])
            if mid is not None:
                deck.materials[mid] = NastranMaterial(
                    mid=mid, card="MAT1",
                    E=_to_float(fields[1]) if len(fields) > 1 else None,
                    nu=_to_float(fields[3]) if len(fields) > 3 else None,
                    rho=_to_float(fields[4]) if len(fields) > 4 else None,
                )

        elif name == "MAT10":
            mid = _to_int(fields[0])
            if mid is not None:
                bulk = _to_float(fields[1]) if len(fields) > 1 else None
                rho = _to_float(fields[2]) if len(fields) > 2 else None
                c0 = _to_float(fields[3]) if len(fields) > 3 else None
                # Any two of the three give the third; MSC completes the
                # card the same way, and a deck that states all three is
                # not checked for consistency here.
                if c0 is None and bulk is not None and rho:
                    c0 = (bulk / rho) ** 0.5
                elif bulk is None and rho is not None and c0 is not None:
                    bulk = rho * c0 ** 2
                elif rho is None and bulk is not None and c0:
                    rho = bulk / c0 ** 2
                deck.materials[mid] = NastranMaterial(
                    mid=mid, card="MAT10", rho=rho, bulk=bulk, c0=c0,
                )

        elif name == "SPC1":
            sid = _to_int(fields[0])
            dofs = fields[1].strip() if len(fields) > 1 else ""
            nodes = _grid_list(fields[2:])
            if sid is not None:
                spc_nodes.setdefault(sid, []).extend(nodes)
                spc_dofs[sid] = dofs

        elif name == "SPC":
            sid = _to_int(fields[0])
            if sid is None:
                continue
            # SPC packs up to two (grid, components, value) triplets.
            for start in (1, 4):
                if len(fields) > start and fields[start].strip():
                    grid = _to_int(fields[start])
                    if grid is None:
                        continue
                    spc_nodes.setdefault(sid, []).append(grid)
                    spc_dofs[sid] = fields[start + 1].strip() \
                        if len(fields) > start + 1 else ""

        elif name.startswith("C"):
            deck.unsupported[name] += 1

    for sid, nodes in spc_nodes.items():
        deck.constraints[sid] = SpcSet(
            sid=sid, dofs=spc_dofs.get(sid, ""),
            nodes=tuple(sorted(set(nodes))),
        )

    if broken:
        message = (
            f"{path.name} writes {', '.join(sorted(broken))} in free field "
            "format with more than eight data fields on one line. A bulk "
            "data line holds eight values and a continuation marker, and "
            "the Gmsh reader goes on reading the extra ones as nodes: the "
            "elements come out with the wrong connectivity, or the read "
            "stops on 'Wrong node index'. Break the card over two lines, or "
            "re-export it in small field (8 column) format."
        )
        if strict:
            raise ValueError(message)
        warnings.warn(message, RuntimeWarning)

    return deck


def _grid_list(fields):
    """
    Grid ids of an ``SPC1`` card, expanding a ``THRU`` range.

    ``G1 THRU G2`` means every id between the two, which is how a whole
    edge is usually held.
    """
    values = [f.strip().upper() for f in fields if f.strip()]
    nodes = []
    i = 0
    while i < len(values):
        if values[i] == "THRU" and nodes and i + 1 < len(values):
            nodes.extend(range(nodes[-1] + 1, int(values[i + 1]) + 1))
            i += 2
            continue
        try:
            nodes.append(int(values[i]))
        except ValueError:
            pass                        # BY, ALL and other qualifiers
        i += 1
    return nodes


def _scale_factor(gmsh, units):
    """Factor bringing the mesh to metres, from the unit or from its size."""
    if units == "auto":
        bbox = np.array(gmsh.model.getBoundingBox(-1, -1), dtype=float)
        return 1e-3 if float((bbox[3:] - bbox[:3]).max()) > 100.0 else 1.0
    try:
        return {"m": 1.0, "mm": 1e-3, "cm": 1e-2, "in": 0.0254}[units]
    except KeyError:
        raise ValueError(
            f"units must be 'auto', 'm', 'mm', 'cm' or 'in', not {units!r}."
        ) from None


def _open_deck(model, path, units, *, verbose=False):
    """Merge a bulk data file into an open model and bring it to metres."""
    model.gmsh.merge(str(path))
    scale = _scale_factor(model.gmsh, units)
    if scale != 1.0:
        # The mesh is discrete: there is no CAD entity to dilate, so the
        # node coordinates themselves are transformed.
        model.mesh.affineTransform(
            [scale, 0.0, 0.0, 0.0,
             0.0, scale, 0.0, 0.0,
             0.0, 0.0, scale, 0.0]
        )
    if verbose:
        print(f"{pathlib.Path(path).name} read, scaled by {scale:g} to metres")
    return scale


def inspect_bdf(path, *, units="m", verbose=True, strict=False):
    """
    List what a bulk data file contains, property id by property id.

    The first half of the Nastran workflow, and the counterpart of
    :func:`~pycafe.create_geom.external.inspect_cad`: it prints the
    entity tags to quote in :func:`bdf_to_mesh` or in
    :class:`~pycafe.core.model_spec.NastranFile`, together with the
    property and material cards behind them, and the role each one looks
    like. The role is a suggestion; the caller states it.

    Parameters
    ----------
    path : str or Path
    units : {"m", "auto", "mm", "cm", "in"}, optional
        Unit of the deck, applied before the sizes are reported. A bulk
        data file records no unit, so ``"m"`` (take it as written) is
        the default and ``"auto"`` calls anything larger than 100 units
        a millimetre model.
    verbose : bool, optional
        Print the table. Default True.
    strict : bool, optional
        Passed to :func:`read_bdf`. Default False here: a deck that
        will not mesh is exactly the one worth looking at.

    Returns
    -------
    deck : BdfDeck
    entities : dict
        ``{pid: EntityInfo}`` of the meshed entities, so a property id
        can also be recognised by where it sits.
    """
    path = pathlib.Path(path)
    deck = read_bdf(path, strict=strict)

    scale, entities, counts, failure = _mesh_overview(path, units)

    if verbose:
        roles = deck.suggest_roles()
        suggestion = {pid: (role, why)
                      for role, items in roles.items()
                      for pid, why in items}
        print(f"{path}  (scaled by {scale:g} to metres)")
        print(deck.describe())
        if failure is not None:
            print(f"\nGmsh could not read the mesh: {failure}")
        print("\nmeshed entities:")
        for pid in sorted(entities):
            info = entities[pid]
            extent = info.bbox_max - info.bbox_min
            role, why = suggestion.get(pid, ("?", "not in the deck"))
            print(f"  dim {info.dim}  tag {pid:<8d} {counts[pid]:6d} elem   "
                  f"bbox {extent[0]:.3f} x {extent[1]:.3f} x {extent[2]:.3f}"
                  f"   looks like {role}  ({why})")
        print("\nName them with bdf_to_mesh(..., fluid=[...], structure=[...]) "
              "or NastranFile(path, fluid=[...], structure=[...]).")
    return deck, entities


def _mesh_overview(path, units):
    """
    Size and element count of every entity Gmsh reads from a deck.

    A file Gmsh refuses is reported rather than raised: the cards were
    read anyway, and a deck that will not mesh is the one most worth
    looking at.

    Returns
    -------
    scale : float
    entities : dict
        ``{tag: EntityInfo}`` for the surface and volume entities.
    counts : dict
        ``{tag: number of elements}``.
    failure : str or None
        What Gmsh said, when it said no.
    """
    entities, counts = {}, {}
    with GmshModel(path.stem, verbose=False) as m:
        try:
            scale = _open_deck(m, path, units)
        except Exception as exc:
            return 1.0, entities, counts, str(exc)

        coords = _node_coordinates(m.gmsh)
        for dim, tag in m.model.getEntities():
            if dim < 2:
                continue
            types, _tags, node_tags = m.mesh.getElements(dim, tag)
            nodes = np.unique(np.concatenate([np.asarray(n, dtype=np.int64)
                                              for n in node_tags])) \
                if len(node_tags) else np.empty(0, dtype=np.int64)
            if nodes.size == 0:
                continue
            xyz = np.array([coords[int(t)] for t in nodes], dtype=float)
            entities[int(tag)] = EntityInfo(
                dim=dim, tag=int(tag), com=xyz.mean(axis=0),
                bbox_min=xyz.min(axis=0), bbox_max=xyz.max(axis=0),
                size=None,
            )
            counts[int(tag)] = sum(
                len(m.mesh.getElementsByType(t, tag)[0]) for t in types
            )
    return scale, entities, counts, None


def _node_coordinates(gmsh):
    """``{node tag: (x, y, z)}`` of the whole mesh."""
    tags, coords, _ = gmsh.model.mesh.getNodes()
    xyz = np.asarray(coords, dtype=float).reshape(-1, 3)
    return {int(t): xyz[i] for i, t in enumerate(np.asarray(tags, dtype=np.int64))}


def _existing_face_keys(gmsh):
    """Sorted node keys of every surface element already in the mesh."""
    keys = set()
    for dim, tag in gmsh.model.getEntities(2):
        types, _tags, node_tags = gmsh.model.mesh.getElements(dim, tag)
        for etype, nodes in zip(types, node_tags):
            _n, _d, _o, n_nodes, *_ = gmsh.model.mesh.getElementProperties(etype)
            conn = np.asarray(nodes, dtype=np.int64).reshape(-1, n_nodes)
            keys.update(tuple(sorted(row.tolist())) for row in conn)
    return keys


def free_faces(gmsh, volume_entities, *, coords=None, skip=()):
    """
    Faces of a volume mesh with a single element behind them.

    The surface an acoustic deck usually does not carry. Faces are
    returned wound **outwards**, checked against the centroid of the
    element behind each one, so the elements written from them describe
    the boundary the same way whatever order the deck listed its nodes
    in.

    Parameters
    ----------
    gmsh : module
        An initialised Gmsh session with the mesh loaded.
    volume_entities : sequence of int
        Entity tags of the volume to take the boundary of.
    coords : dict, optional
        ``{node tag: (x, y, z)}``; read from the session when omitted.
    skip : set of tuple, optional
        Sorted node keys to leave out — the faces some surface element
        already covers, which would otherwise be written twice.

    Returns
    -------
    list of tuple
        Node tags of each boundary face, outward wound.
    """
    if coords is None:
        coords = _node_coordinates(gmsh)

    seen = {}
    for tag in volume_entities:
        types, _tags, _nodes = gmsh.model.mesh.getElements(3, int(tag))
        for etype in types:
            elem_tags, elem_nodes = gmsh.model.mesh.getElementsByType(
                etype, int(tag)
            )
            if len(elem_tags) == 0:
                continue
            _n, _d, _o, n_nodes, *_ = gmsh.model.mesh.getElementProperties(etype)
            conn = np.asarray(elem_nodes, dtype=np.int64).reshape(-1, n_nodes)
            centroids = np.array([
                np.mean([coords[int(t)] for t in row], axis=0) for row in conn
            ])
            for face_type in (3, 4):
                flat = gmsh.model.mesh.getElementFaceNodes(
                    etype, face_type, tag=int(tag)
                )
                if len(flat) == 0:
                    continue
                faces = np.asarray(flat, dtype=np.int64).reshape(-1, face_type)
                per_element = faces.shape[0] // conn.shape[0]
                owner = np.repeat(np.arange(conn.shape[0]), per_element)
                for row, e in zip(faces, owner):
                    key = tuple(sorted(row.tolist()))
                    if key in seen:
                        seen[key] = None            # interior: two owners
                    else:
                        seen[key] = (tuple(row.tolist()), centroids[e])

    out = []
    for key, value in seen.items():
        if value is None or key in skip:
            continue
        nodes, centre = value
        out.append(tuple(_outward(nodes, centre, coords)))
    return out


def _outward(nodes, element_centre, coords):
    """The face node order that points away from the element behind it."""
    x = np.array([coords[int(t)] for t in nodes], dtype=float)
    normal = np.cross(x[1] - x[0], x[2] - x[0])
    if float(np.dot(normal, x.mean(axis=0) - element_centre)) < 0.0:
        return tuple(reversed(nodes))
    return tuple(nodes)


def _face_info(nodes, index, coords):
    """An :class:`EntityInfo` for one face, so the tag rules can read it."""
    x = np.array([coords[int(t)] for t in nodes], dtype=float)
    area = 0.0
    for i in range(1, len(nodes) - 1):
        area += 0.5 * float(np.linalg.norm(
            np.cross(x[i] - x[0], x[i + 1] - x[0])
        ))
    return EntityInfo(dim=2, tag=int(index), com=x.mean(axis=0),
                      bbox_min=x.min(axis=0), bbox_max=x.max(axis=0),
                      size=area)


def tag_faces(model, faces, rules, *, walls="rigid_walls", coords=None,
              strict=True, verbose=False):
    """
    Turn a list of faces into named surface groups, first rule wins.

    The bridge between a mesh that has no surface and a boundary
    condition that needs one: every face is classified by the same
    :class:`~pycafe.create_geom.external.TagRule` objects used on CAD
    entities, and each group becomes a discrete surface entity holding
    its elements.

    ``select="largest"`` and ``select="smallest"`` are not honoured
    here — the rules see one face at a time, not an entity — so
    :func:`~pycafe.create_geom.external.largest` has no meaning on a
    face list; ``on_plane`` and ``in_box`` are the ones that do.

    Parameters
    ----------
    model : GmshModel
        Open model holding the mesh.
    faces : sequence of tuple
        Node tags per face, as returned by :func:`free_faces`.
    rules : sequence of TagRule
        Evaluated in order; a face claimed by an earlier rule is not
        offered to a later one.
    walls : str or None, optional
        Name for the faces no rule claimed. ``None`` leaves them out of
        the mesh entirely.
    coords : dict, optional
    strict : bool, optional
        Passed to :meth:`GmshModel.physical`.
    verbose : bool, optional

    Returns
    -------
    dict
        ``{name: number of faces}``.
    """
    if coords is None:
        coords = _node_coordinates(model.gmsh)

    infos = [_face_info(f, i, coords) for i, f in enumerate(faces)]
    remaining = set(range(len(faces)))
    written = {}

    for rule in list(rules) + ([_WallRule(walls)] if walls else []):
        name = rule.name
        picked = sorted(i for i in remaining if rule.matches(infos[i]))
        if not picked:
            # Nothing left over is a normal outcome for the wall rule;
            # a named rule that matches nothing is a plane off by a
            # tolerance, and always worth saying before the solve.
            if not isinstance(rule, _WallRule):
                warnings.warn(
                    f"rule '{name}' matched none of the {len(faces)} boundary "
                    "faces; the group will not exist in the mesh.",
                    RuntimeWarning,
                )
            continue
        _write_faces(model, [faces[i] for i in picked], name, strict=strict)
        remaining.difference_update(picked)
        written[name] = len(picked)
        if verbose:
            print(f"  {name:<16} {len(picked)} face(s)")

    if remaining and verbose:
        print(f"  {len(remaining)} face(s) left unnamed")
    return written


class _WallRule:
    """Everything still unclaimed, under one name. The last rule."""

    def __init__(self, name):
        self.name = name
        self.dim = 2

    @staticmethod
    def matches(_info):
        return True


def _write_faces(model, faces, name, *, strict=True):
    """One discrete surface entity holding these faces, tagged ``name``."""
    gmsh = model.gmsh
    by_size = {}
    for face in faces:
        by_size.setdefault(len(face), []).append(face)

    entity = gmsh.model.addDiscreteEntity(2)
    for n_nodes, group in by_size.items():
        shape = {3: "triangle", 4: "quadrangle"}.get(n_nodes)
        if shape is None:
            raise ValueError(
                f"a boundary face with {n_nodes} nodes has no Gmsh surface "
                "element; only first order faces (3 or 4 nodes) are written."
            )
        etype = gmsh.model.mesh.getElementType(shape, 1)
        flat = [int(t) for face in group for t in face]
        gmsh.model.mesh.addElementsByType(entity, etype, [], flat)
    model.physical(2, [entity], name, strict=strict)
    return entity


def _clamp_nodes(deck, clamp, present):
    """
    Node tags of the support, from the deck or from the caller.

    ``"spc"`` takes the single constraint set of the deck; a file with
    several of them is ambiguous, since a Nastran subcase picks one by
    id, and there is no subcase here to read.
    """
    if clamp is None:
        return []
    if isinstance(clamp, str):
        if clamp != "spc":
            raise ValueError(
                f"clamp must be 'spc', a constraint set id, a list of grid "
                f"ids or None, not {clamp!r}."
            )
        if not deck.constraints:
            return []
        if len(deck.constraints) > 1:
            raise ValueError(
                f"{deck.path.name} holds {len(deck.constraints)} constraint "
                f"sets ({sorted(deck.constraints)}); a subcase would pick "
                "one. Pass clamp=<sid> to say which."
            )
        nodes = next(iter(deck.constraints.values())).nodes
    elif isinstance(clamp, int):
        if clamp not in deck.constraints:
            raise KeyError(
                f"no SPC set {clamp} in {deck.path.name}; it has "
                f"{sorted(deck.constraints)}."
            )
        nodes = deck.constraints[clamp].nodes
    else:
        nodes = [int(n) for n in clamp]

    kept = [int(n) for n in nodes if int(n) in present]
    dropped = len(nodes) - len(kept)
    if dropped:
        warnings.warn(
            f"{dropped} constrained grid(s) are referenced by no element and "
            "are not in the mesh; they cannot be supported.", RuntimeWarning,
        )
    return kept


def bdf_to_mesh(
    bdf_path,
    output_path,
    *,
    fluid=(),
    structure=(),
    boundaries=None,
    surface_rules=(),
    walls="rigid_walls",
    clamp="spc",
    units="m",
    deck=None,
    validate=True,
    verbose=False,
):
    """
    Read a bulk data file and write the pyCAFE mesh it describes.

    Parameters
    ----------
    bdf_path : str or Path
        The ``.bdf`` or ``.nas`` file.
    output_path : str or Path
        Destination ``.msh``.
    fluid : sequence of int
        Property ids of the acoustic domain. These are Gmsh entity tags
        once the file is read; :func:`inspect_bdf` lists them.
    structure : sequence of int, optional
        Property ids of the shell elements that are the structure.
    boundaries : dict, optional
        ``{name: [pids]}`` for surface property ids the deck already
        carries — an analyst who meshed the walls as ``CQUAD4`` has
        named them by property, and this keeps that naming.
    surface_rules : sequence of TagRule, optional
        Applied to the **free faces of the fluid**, the ones no surface
        element covers. This is how an impedance or a prescribed
        velocity gets a face to act on in a deck that holds volume
        elements only.
    walls : str or None, optional
        Name for the free faces no rule claimed. Default
        ``"rigid_walls"``, which is the natural boundary condition;
        ``None`` writes no group for them, leaving the faces out of the
        file.
    clamp : {"spc"}, int, sequence of int or None, optional
        The support of the structure. ``"spc"`` (default) reads the
        single ``SPC``/``SPC1`` set of the deck; an integer picks a set
        by id; a sequence gives the grid ids directly. Written as the
        ``plate_clamp`` group. Which degrees of freedom are blocked is
        **not** read from the card: that is
        ``Structure.support``.
    units : {"m", "auto", "mm", "cm", "in"}, optional
        Unit of the deck. A bulk data file records none, so the default
        is to take the coordinates as written.
    deck : BdfDeck, optional
        An already parsed deck, to avoid reading the file twice.
    validate : bool, optional
        Run :func:`~pycafe.create_geom.validation.validate_mesh` on the
        result. Default True.
    verbose : bool, optional

    Returns
    -------
    pathlib.Path or (pathlib.Path, MeshReport)
        The report comes along when ``validate`` is True.

    Raises
    ------
    ValueError
        If no fluid and no structure property id is given, or if a
        quoted id is not in the file.

    Examples
    --------
    >>> from pycafe.create_geom import on_plane
    >>> bdf_to_mesh("duct.bdf", "duct.msh", fluid=[101],   # doctest: +SKIP
    ...             surface_rules=[on_plane("x", 0.0, name="inlet"),
    ...                            on_plane("x", 3.0, name="outlet")])
    """
    bdf_path = pathlib.Path(bdf_path)
    boundaries = dict(boundaries or {})
    fluid = [int(t) for t in fluid]
    structure = [int(t) for t in structure]
    if not fluid and not structure:
        raise ValueError(
            f"bdf_to_mesh({bdf_path.name}) needs the property ids of the "
            "fluid, of the structure, or of both: a bulk data file carries "
            "no roles. Run inspect_bdf(path) to see what is in it."
        )

    if deck is None:
        deck = read_bdf(bdf_path)

    with GmshModel(bdf_path.stem, verbose=verbose) as m:
        _open_deck(m, bdf_path, units, verbose=verbose)
        coords = _node_coordinates(m.gmsh)
        present = {dim: {tag for _d, tag in m.model.getEntities(dim)}
                   for dim in (0, 1, 2, 3)}

        named = {"fluid": (3, fluid)} if fluid else {}
        if structure:
            named["plate"] = (2, structure)
        for name, tags in boundaries.items():
            named[name] = (2, [int(t) for t in tags])

        for name, (dim, tags) in named.items():
            missing = [t for t in tags if t not in present[dim]]
            if missing:
                raise ValueError(
                    f"property id(s) {missing} carry no dim {dim} element in "
                    f"{bdf_path.name}. Entities read: "
                    f"{ {d: sorted(v) for d, v in present.items() if v} }."
                )
            m.physical(dim, tags, name)

        untagged = sorted(present[2] - {t for _d, ts in named.values()
                                        for t in ts})
        if untagged:
            warnings.warn(
                f"surface property id(s) {untagged} were read but given no "
                "name: their elements stay in the mesh without a group and "
                "no boundary condition can select them. Name them through "
                "'structure' or 'boundaries'.", RuntimeWarning,
            )

        if fluid and (surface_rules or walls):
            faces = free_faces(m.gmsh, fluid, coords=coords,
                               skip=_existing_face_keys(m.gmsh))
            if verbose:
                print(f"{len(faces)} free face(s) on the fluid boundary:")
            tag_faces(m, faces, surface_rules, walls=walls, coords=coords,
                      verbose=verbose)

        # A deck of a fluid-only model may still carry the constraints
        # of the run it came from; without a structure there is nothing
        # for them to hold, so the automatic reading stays quiet. An
        # explicit clamp is always honoured.
        auto = isinstance(clamp, str)
        held = [] if (auto and not structure) else _clamp_nodes(deck, clamp,
                                                                coords)
        if held:
            point = m.model.addDiscreteEntity(0)
            etype = m.mesh.getElementType("point", 1)
            m.mesh.addElementsByType(point, etype, [], held)
            m.physical(0, [point], "plate_clamp")
            if verbose:
                print(f"  plate_clamp      {len(held)} node(s) from the deck")

        out = m.write(output_path)
        if verbose:
            print(m.summary())

    if not validate:
        return out

    from .validation import validate_mesh

    report = validate_mesh(str(out))
    if verbose:
        print(report)
    return out, report
