r"""
Bringing in geometry that pyCAFE did not make: CAD files and foreign meshes.

A STEP file has no idea what a "fluid" is, and a mesh exported from
somewhere else usually carries either no physical group at all or names
that mean something to another code. Both are the same problem: the
entities are there, the *roles* are missing. This module attaches them,
by geometry rather than by hand:

.. code-block:: python

    from pycafe.create_geom import cad_to_mesh, on_plane, largest, rest

    cad_to_mesh(
        "cabin.step", "cabin.msh",
        rules=[
            largest(3, "fluid"),                  # the biggest volume
            on_plane("z", 0.5, dim=2, name="plate"),
            rest(2, "rigid_walls"),               # whatever is left
        ],
        f_max=500.0,                              # sizes the mesh
    )

and for a mesh whose groups exist but are named for another solver:

.. code-block:: python

    from pycafe.create_geom import list_groups, retag_mesh

    list_groups("from_ansys.msh")
    retag_mesh("from_ansys.msh", rename={"AIR": "fluid", "SKIN": "plate"},
               output_path="fixed.msh")

Rules are evaluated in order and the first match wins, so they read as a
classification: the specific ones first, ``rest(...)`` last. Nothing is
tagged twice, and :func:`~pycafe.create_geom.validation.validate_mesh`
runs at the end to say whether the result is actually usable.

Caveats worth knowing before trusting a converted model: a CAD import
gives a *tetrahedral* mesh unless the geometry is simple enough to be
recombined, and pyCAFE's acoustic kernels are CHEXA8 and CQUAD4/8 — a
tet mesh will load and fail the validator on element types. And a plate
imported as its own surface is not conforming with the fluid unless the
two were meshed together: check ``interface conforming`` in the report,
not just the group names.
"""

import pathlib
import warnings
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .gmsh_workflow import GmshModel


@dataclass
class EntityInfo:
    """Geometric summary of one CAD or mesh entity, passed to the rules."""

    dim: int
    tag: int
    com: np.ndarray          # centre of mass (or centroid of its nodes)
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    size: Optional[float]    # length, area or volume when available

    @property
    def extent(self):
        """Bounding-box extents."""
        return self.bbox_max - self.bbox_min


@dataclass
class TagRule:
    """
    One classification rule: *what to call* the entities that match.

    Parameters
    ----------
    name : str
        Physical group name to give them.
    dim : int
        Entity dimension the rule applies to.
    where : callable, optional
        ``where(info: EntityInfo) -> bool``. ``None`` matches every
        entity of that dimension.
    select : {"all", "largest", "smallest"}, optional
        Applied after ``where``: keep all matches, or only the biggest
        or smallest one by ``size``.
    """

    name: str
    dim: int
    where: Optional[Callable[[EntityInfo], bool]] = None
    select: str = "all"

    def matches(self, info):
        return self.where is None or bool(self.where(info))


def on_plane(axis, value, *, dim=2, name, tol=1e-6):
    """Entities whose centre lies on ``axis = value`` (``axis`` in x, y, z)."""
    index = {"x": 0, "y": 1, "z": 2}[str(axis).lower()]
    return TagRule(
        name=name, dim=dim,
        where=lambda i: abs(i.com[index] - float(value)) <= tol,
    )


def in_box(lo, hi, *, dim, name):
    """Entities whose centre falls inside the box ``lo..hi``."""
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    return TagRule(
        name=name, dim=dim,
        where=lambda i: bool(np.all(i.com >= lo) and np.all(i.com <= hi)),
    )


def largest(dim, name):
    """The single biggest entity of that dimension — usually the fluid volume."""
    return TagRule(name=name, dim=dim, select="largest")


def everything(dim, name):
    """Every entity of that dimension."""
    return TagRule(name=name, dim=dim)


def rest(dim, name):
    """Whatever is still untagged at this dimension. Put it last."""
    return TagRule(name=name, dim=dim)


def _entity_info(gmsh, dim, tag):
    """Geometry of an entity, working for CAD and for discrete entities."""
    try:
        bbox = gmsh.model.getBoundingBox(dim, tag)
        bbox_min = np.array(bbox[:3], dtype=float)
        bbox_max = np.array(bbox[3:], dtype=float)
    except Exception:                       # pragma: no cover - gmsh internals
        bbox_min = bbox_max = np.zeros(3)

    com = None
    try:
        com = np.array(gmsh.model.occ.getCenterOfMass(dim, tag), dtype=float)
    except Exception:
        pass
    if com is None or not np.all(np.isfinite(com)):
        # Discrete entities (a loaded .msh) have no CAD mass properties:
        # fall back to the centroid of their nodes.
        try:
            _, coords, _ = gmsh.model.mesh.getNodes(dim, tag, includeBoundary=True)
            com = coords.reshape(-1, 3).mean(axis=0)
        except Exception:
            com = 0.5 * (bbox_min + bbox_max)

    size = None
    try:
        size = float(gmsh.model.occ.getMass(dim, tag))
    except Exception:
        extent = bbox_max - bbox_min
        positive = extent[extent > 0]
        size = float(np.prod(positive)) if positive.size else 0.0

    return EntityInfo(dim=dim, tag=int(tag), com=com,
                      bbox_min=bbox_min, bbox_max=bbox_max, size=size)


def assign_groups(model, rules, *, strict=True):
    """
    Tag entities according to geometric rules, first match wins.

    Parameters
    ----------
    model : GmshModel
        An open model (inside its ``with`` block), already synchronized.
    rules : sequence of TagRule
        Evaluated in order, per dimension. An entity already claimed by
        an earlier rule is skipped, so ``rest(...)`` collects the
        remainder.
    strict : bool, optional
        Passed to :meth:`GmshModel.physical` for the name check.

    Returns
    -------
    dict
        ``{name: [entity tags]}`` actually tagged.

    Warns
    -----
    RuntimeWarning
        For a rule that matched nothing — usually a plane that is off by
        a tolerance, and always worth knowing before the solve.
    """
    gmsh = model.gmsh
    claimed = set()
    assigned = {}

    for rule in rules:
        entities = [t for (d, t) in gmsh.model.getEntities(rule.dim)
                    if (rule.dim, t) not in claimed]
        infos = [_entity_info(gmsh, rule.dim, t) for t in entities]
        matched = [i for i in infos if rule.matches(i)]

        if matched and rule.select in ("largest", "smallest"):
            key = max if rule.select == "largest" else min
            matched = [key(matched, key=lambda i: (i.size or 0.0))]

        if not matched:
            warnings.warn(
                f"rule '{rule.name}' (dim {rule.dim}) matched no entity; "
                "the group will not exist in the mesh.", RuntimeWarning,
            )
            continue

        tags = [i.tag for i in matched]
        model.physical(rule.dim, tags, rule.name, strict=strict)
        claimed.update((rule.dim, t) for t in tags)
        assigned[rule.name] = tags

    return assigned


def cad_to_mesh(
    cad_path,
    output_path,
    *,
    rules=(),
    mesh_size=None,
    f_max=None,
    c0=343.0,
    elements_per_wavelength=13,
    order=1,
    recombine=True,
    dim=3,
    validate=True,
    verbose=False,
):
    """
    Import a CAD file, tag it by geometry, mesh it, write a pyCAFE ``.msh``.

    Parameters
    ----------
    cad_path : str or Path
        Anything OpenCASCADE reads: ``.step``, ``.stp``, ``.iges``,
        ``.brep``.
    output_path : str or Path
        Destination mesh.
    rules : sequence of TagRule
        How to name the entities; see the module header.
    mesh_size : float, optional
        Target element size [m]. Mutually exclusive with ``f_max``.
    f_max : float, optional
        Highest frequency of interest; the size is then chosen as
        ``c0 / (f_max * elements_per_wavelength)``.
    c0, elements_per_wavelength : float, optional
        Used with ``f_max``.
    order : {1, 2}, optional
    recombine : bool, optional
        Ask Gmsh to recombine into quads/hexes. It succeeds only on
        geometry that allows it; otherwise the mesh stays tetrahedral,
        which pyCAFE has no kernel for — the validator will say so.
    dim : int, optional
        Dimension to mesh.
    validate : bool, optional
        Run :func:`~pycafe.create_geom.validation.validate_mesh` on the
        result. Default True.
    verbose : bool, optional

    Returns
    -------
    pathlib.Path or (pathlib.Path, MeshReport)
        The report comes along when ``validate`` is True.
    """
    cad_path = pathlib.Path(cad_path)
    if not cad_path.exists():
        raise FileNotFoundError(f"No CAD file at {cad_path}.")
    if mesh_size is not None and f_max is not None:
        raise ValueError("Give either 'mesh_size' or 'f_max', not both.")

    with GmshModel(cad_path.stem, verbose=verbose, order=order,
                   recombine=recombine) as m:
        m.occ.importShapes(str(cad_path))
        m.occ.synchronize()

        if mesh_size is not None:
            m.size(mesh_size)
        elif f_max is not None:
            h = m.size_for_frequency(f_max, c0, elements_per_wavelength)
            if verbose:
                print(f"element size {h * 1e3:.1f} mm for {f_max} Hz "
                      f"({elements_per_wavelength} per wavelength)")

        assign_groups(m, rules)
        m.generate(dim, kernel="occ")
        out = m.write(output_path)
        if verbose:
            print(m.summary())

    if not validate:
        return out

    from .validation import validate_mesh

    report = validate_mesh(str(out), c0=c0, f_max=f_max)
    if verbose:
        print(report)
    return out, report


def list_groups(msh_path, verbose=True):
    """
    Physical groups of an existing mesh — what is in there, and under what name.

    Returns
    -------
    dict
        ``{name: {"dim": d, "tag": t, "entities": [...], "nodes": n}}``.
    """
    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(msh_path))
        out = {}
        for dim, tag in gmsh.model.getPhysicalGroups():
            name = gmsh.model.getPhysicalName(dim, tag) or f"<unnamed {dim}/{tag}>"
            entities = list(gmsh.model.getEntitiesForPhysicalGroup(dim, tag))
            n_nodes = len(gmsh.model.mesh.getNodesForPhysicalGroup(dim, tag)[0])
            out[name] = {"dim": dim, "tag": tag,
                         "entities": entities, "nodes": n_nodes}
    finally:
        gmsh.finalize()

    if verbose:
        from .conventions import role_of

        print(f"{msh_path}:")
        for name, info in out.items():
            role = role_of(name)
            label = f"role {role}" if role else "boundary name"
            print(f"  dim {info['dim']}  {name:<16} {info['nodes']:6d} nodes "
                  f"({label})")
    return out


def retag_mesh(
    msh_path,
    *,
    rename=None,
    drop=(),
    rules=(),
    output_path=None,
    validate=True,
    verbose=False,
):
    """
    Rewrite the physical groups of an existing mesh, without remeshing.

    The repair path for a foreign mesh: rename the groups to pyCAFE's
    role names, drop the ones that mean nothing here, and add missing
    ones by geometric rules on the entities the file already contains.

    Parameters
    ----------
    msh_path : str or Path
    rename : dict, optional
        ``{old_name: new_name}``.
    drop : sequence of str, optional
        Names to remove.
    rules : sequence of TagRule, optional
        Extra groups to create by geometry, applied after the renaming.
        Entities already carrying a kept group are still candidates —
        an entity may belong to several groups.
    output_path : str or Path, optional
        Where to write. Defaults to ``<name>_retagged.msh`` next to the
        input; pass the input path itself to overwrite it.
    validate : bool, optional
    verbose : bool, optional

    Returns
    -------
    pathlib.Path or (pathlib.Path, MeshReport)

    Raises
    ------
    KeyError
        If a name in ``rename`` or ``drop`` is not in the mesh — a typo
        there would silently do nothing.
    """
    import gmsh

    msh_path = pathlib.Path(msh_path)
    rename = dict(rename or {})
    drop = set(drop)
    if output_path is None:
        output_path = msh_path.with_name(f"{msh_path.stem}_retagged.msh")
    output_path = pathlib.Path(output_path)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        gmsh.open(str(msh_path))

        existing = {}
        for dim, tag in gmsh.model.getPhysicalGroups():
            name = gmsh.model.getPhysicalName(dim, tag)
            existing[name] = (dim, tag,
                              list(gmsh.model.getEntitiesForPhysicalGroup(dim, tag)))

        unknown = (set(rename) | drop) - set(existing)
        if unknown:
            raise KeyError(
                f"no group named {sorted(unknown)} in {msh_path.name}; "
                f"it has {sorted(existing)}."
            )

        gmsh.model.removePhysicalGroups()
        # removePhysicalGroups leaves the old names registered against
        # their tags: re-adding a group that lands on one of those tags
        # would inherit the name it is supposed to lose. Fresh tags plus
        # an explicit setPhysicalName avoid that entirely.
        next_tag = max((tag for _dim, tag, _e in existing.values()),
                       default=0) + 1
        kept = {}
        for name, (dim, _tag, entities) in existing.items():
            if name in drop:
                continue
            new_name = rename.get(name, name)
            gmsh.model.addPhysicalGroup(dim, entities, tag=next_tag)
            gmsh.model.setPhysicalName(dim, next_tag, new_name)
            kept[new_name] = (dim, entities)
            next_tag += 1

        if rules:
            # assign_groups needs a GmshModel; wrap the live session.
            model = GmshModel.__new__(GmshModel)
            model.name = msh_path.stem
            model.groups = {n: (d, 0) for n, (d, _e) in kept.items()}
            model._gmsh = gmsh
            assign_groups(model, rules)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        gmsh.write(str(output_path))
    finally:
        gmsh.finalize()

    if verbose:
        list_groups(output_path)
    if not validate:
        return output_path

    from .validation import validate_mesh

    report = validate_mesh(str(output_path))
    if verbose:
        print(report)
    return output_path, report
