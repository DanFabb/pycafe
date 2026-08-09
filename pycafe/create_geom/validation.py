r"""
Is this mesh usable by pyCAFE? — one function that answers before the solve.

Most of what goes wrong with an external or hand-made mesh goes wrong
*silently*: a physical group called ``"fluids"`` is not a fluid domain, a
plate whose nodes do not coincide with the fluid ones is not an
interface, an element three times too big is not an error anywhere — it
is a 15% frequency shift. :func:`validate_mesh` runs the checks that
would otherwise be discovered halfway through an analysis:

* the roles the analysis needs are present, and are spelled the way the
  solvers look them up (with a suggestion when a name is a near miss);
* every domain group holds elements of a type pyCAFE has a kernel for;
* connectivity stays inside the node array, and no node is orphaned;
* for a coupled model, whether the interface is **conforming**: every
  structural element a face of exactly one fluid element. Zero (the
  meshes do not share nodes) is reported as a warning, since the
  coupling can still be built by interpolation
  (:mod:`pycafe.build_matrices.coupling_nonconforming`) at a cost in
  accuracy; two owners (the surface is inside the fluid, where no
  outward normal exists) stays an error, as the orientation has to be
  imposed by hand;
* the element size resolves the frequency band asked for.

.. code-block:: python

    from pycafe.create_geom import validate_mesh

    report = validate_mesh("box_plate.msh", f_max=500.0)
    print(report)
    report.raise_if_invalid()      # in a script, stop here on errors

The rule behind the resolution check is the dispersion of the trilinear
acoustic element: the frequency error goes roughly as
``164 / n_lambda^2``, so 13 elements per wavelength buy 1% and 6 are the
usual bare minimum.
"""

import difflib
import pathlib
from dataclasses import dataclass, field
from typing import List

import numpy as np

from ..build_matrices.element_registry import ELEMENT_TYPES
from .conventions import REQUIRED_ROLES, ROLE_BY_NAME, find_group, role_of


@dataclass
class MeshReport:
    """
    What the checks found. ``ok`` is True when there is no error.

    Attributes
    ----------
    source : str
        Where the mesh came from.
    analysis : str
        The kind of analysis the mesh was checked against.
    n_nodes : int
    element_counts : dict
        ``{gmsh element name: count}`` over the whole mesh.
    roles : dict
        ``{role: group name}`` for the roles that were found.
    errors : list of str
        Things that will make an analysis fail or, worse, quietly
        produce a wrong answer.
    warnings : list of str
        Things worth knowing that are not fatal.
    notes : list of str
        Measurements: sizes, counts, resolution.
    """

    source: str = ""
    analysis: str = ""
    n_nodes: int = 0
    element_counts: dict = field(default_factory=dict)
    roles: dict = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def ok(self):
        """True when nothing blocking was found."""
        return not self.errors

    def raise_if_invalid(self):
        """Raise ``ValueError`` listing the errors, if any."""
        if self.errors:
            raise ValueError(
                f"{self.source} cannot be used for a {self.analysis} "
                "analysis:\n  - " + "\n  - ".join(self.errors)
            )

    def __str__(self):
        head = f"{self.source}  ({self.analysis} analysis)"
        lines = [head, "-" * len(head),
                 f"  {self.n_nodes} nodes, " + ", ".join(
                     f"{n} x {name}" for name, n in self.element_counts.items()
                 )]
        if self.roles:
            lines.append("  roles: " + ", ".join(
                f"{r} = '{g}'" for r, g in self.roles.items()
            ))
        for tag, items in (("ERROR", self.errors),
                           ("warning", self.warnings),
                           ("", self.notes)):
            for item in items:
                lines.append(f"  {tag + ':' if tag else '·'} {item}")
        lines.append(f"  => {'usable' if self.ok else 'NOT usable'}")
        return "\n".join(lines)


def _load(mesh):
    """Accept a path or an already loaded ``(nodes, elements, boundaries, groups)``."""
    if isinstance(mesh, (str, pathlib.Path)):
        from .visualize_mesh import load_mesh_with_groups

        nodes, elements, boundaries, groups = load_mesh_with_groups(
            str(mesh), verbose=False
        )
        return str(mesh), nodes, elements, boundaries, groups
    nodes, elements, boundaries, groups = mesh
    return "<mesh in memory>", nodes, elements, boundaries, groups


def _registered_fields(group):
    """Element fields ("acoustic"/"structural") a group can supply."""
    fields = set()
    for gmsh_name, conn in group["elements"].items():
        for spec in ELEMENT_TYPES.values():
            if (conn.shape[1] == spec.n_nodes
                    and any(n in gmsh_name for n in spec.gmsh_names)):
                fields.add((spec.field, spec.dim))
    return fields


def _element_sizes(nodes, conn0):
    """Largest bounding-box side of every element: min, mean, max."""
    if conn0.size == 0:
        return None
    x = nodes[conn0]
    extents = x.max(axis=1) - x.min(axis=1)
    h = extents.max(axis=1)
    return float(h.min()), float(h.mean()), float(h.max())


def _face_key_set(conn0):
    """Sorted-node keys of a set of elements, to compare surfaces."""
    return {tuple(sorted(row.tolist())) for row in np.asarray(conn0, dtype=int)}


def validate_mesh(mesh, *, analysis="auto", c0=343.0, f_max=None):
    """
    Check a mesh against the pyCAFE contract.

    Parameters
    ----------
    mesh : str, Path or tuple
        A ``.msh`` path, or the tuple returned by
        ``load_mesh_with_groups``.
    analysis : {"auto", "acoustic", "structural", "vibroacoustic"}, optional
        What the mesh has to support. ``"auto"`` infers it from the
        groups present.
    c0 : float, optional
        Speed of sound, for the resolution check.
    f_max : float, optional
        Highest frequency of interest [Hz]. Without it the resolution
        check is reported as the frequency the mesh *can* reach.

    Returns
    -------
    MeshReport
    """
    source, nodes, elements, boundaries, groups = _load(mesh)
    nodes = np.asarray(nodes, dtype=float)
    report = MeshReport(source=source, n_nodes=nodes.shape[0])
    report.element_counts = {k: int(v.shape[0]) for k, v in elements.items()}

    # ---------------------------------------------------------------- roles
    for role in ("fluid", "structure", "clamp"):
        name = find_group(groups, role)
        if name is not None:
            report.roles[role] = name

    if analysis == "auto":
        if "fluid" in report.roles and "structure" in report.roles:
            analysis = "vibroacoustic"
        elif "fluid" in report.roles:
            analysis = "acoustic"
        elif "structure" in report.roles:
            analysis = "structural"
        else:
            analysis = "acoustic"       # so the missing fluid is reported
    report.analysis = analysis

    for role in REQUIRED_ROLES.get(analysis, ()):
        if role not in report.roles:
            accepted = ROLE_BY_NAME
            expected = [n for n, s in accepted.items() if s.role == role]
            close = difflib.get_close_matches(
                expected[0], [str(g).lower() for g in groups], n=1, cutoff=0.6
            )
            hint = (f" Did you mean the group '{close[0]}'?" if close else
                    f" Groups in this mesh: {sorted(groups)}.")
            report.errors.append(
                f"no {role} domain: expected a physical group named one of "
                f"{expected}.{hint}"
            )

    if analysis == "vibroacoustic" and "clamp" not in report.roles:
        report.warnings.append(
            "no support group (plate_clamp / clamp / fixed): the structure "
            "will be left free, which leaves it with rigid-body modes."
        )

    # ------------------------------------------------------- element types
    for role, group_name in report.roles.items():
        if role == "clamp":
            continue
        want = "acoustic" if role == "fluid" else "structural"
        fields = _registered_fields(groups[group_name])
        if not any(f == want for f, _ in fields):
            report.errors.append(
                f"group '{group_name}' ({role}) holds no element type "
                f"pyCAFE can assemble as {want}; it contains "
                f"{sorted(groups[group_name]['elements'])}. Registered "
                f"types: {sorted(ELEMENT_TYPES)}."
            )

    unknown = [name for name in elements
               if not any(any(g in name for g in spec.gmsh_names)
                          for spec in ELEMENT_TYPES.values())
               and "Line" not in name and "Point" not in name]
    if unknown:
        report.warnings.append(
            f"element types with no kernel, ignored by the assembly: "
            f"{unknown}"
        )

    # -------------------------------------------------------- connectivity
    for name, conn in elements.items():
        conn = np.asarray(conn, dtype=int)
        if conn.size and (conn.min() < 1 or conn.max() > nodes.shape[0]):
            report.errors.append(
                f"'{name}' connectivity points outside the node array "
                f"(range {conn.min()}..{conn.max()} for {nodes.shape[0]} "
                "nodes). Load the mesh with load_mesh_with_groups, which "
                "remaps Gmsh tags to rows."
            )

    used = np.zeros(nodes.shape[0], dtype=bool)
    for conn in elements.values():
        conn = np.asarray(conn, dtype=int)
        if conn.size:
            used[np.clip(conn, 1, nodes.shape[0]) - 1] = True
    orphans = int((~used).sum())
    if orphans:
        report.warnings.append(
            f"{orphans} node(s) belong to no element; harmless for the "
            "assembly, but usually the sign of a merge that went wrong."
        )

    for name, group in groups.items():
        if not group["elements"]:
            report.warnings.append(
                f"group '{name}' has no element: it cannot be selected as "
                "a boundary."
            )

    # ---------------------------------------------------------- interface
    if analysis == "vibroacoustic" and {"fluid", "structure"} <= set(report.roles):
        from ..build_matrices.coupling import _fluid_face_owners
        from ..build_matrices.domains import identify_domains

        try:
            domains = identify_domains(groups)
        except RuntimeError as exc:
            report.errors.append(str(exc))
            domains = None

        if domains is not None and "structure" in domains:
            fluid_conn0 = domains["fluid"]["conn0"]
            iface_conn0 = domains["structure"]["conn0"]

            fluid_nodes = set(np.asarray(
                domains["fluid"]["nodes"], dtype=int).tolist())
            plate_nodes = set(np.asarray(
                domains["structure"]["nodes"], dtype=int).tolist())
            missing = plate_nodes - fluid_nodes
            if missing:
                report.warnings.append(
                    f"{len(missing)} structural node(s) are not fluid nodes: "
                    "the two meshes do not share their interface nodes, so "
                    "the coupling cannot be assembled by shared nodes. It "
                    "will be built by interpolation instead (Mi & Zheng "
                    "2018); mesh the plate as the faces of the fluid "
                    "elements to keep the exact, conforming one."
                )

            try:
                owners = _fluid_face_owners(fluid_conn0)
            except ValueError as exc:
                owners = None
                report.warnings.append(f"interface not checked: {exc}")

            if owners is not None:
                keys = _face_key_set(iface_conn0)
                not_a_face = [k for k in keys if k not in owners]
                interior = [k for k in keys
                            if k in owners and len(owners[k]) > 1]
                if not_a_face:
                    report.warnings.append(
                        f"{len(not_a_face)} of {len(keys)} structural "
                        "element(s) are not faces of any fluid element: the "
                        "interface is not conforming, so the coupling is "
                        "built by the interpolation method of "
                        "pycafe.build_matrices.coupling_nonconforming "
                        "(Mi & Zheng 2018) rather than by shared nodes."
                    )
                if interior:
                    report.errors.append(
                        f"{len(interior)} structural element(s) sit *inside* "
                        "the fluid (shared by two fluid elements), where "
                        "there is no outward normal to deduce. Pass an "
                        "explicit interface_sign, or fix the group."
                    )
                if not not_a_face and not interior:
                    report.notes.append(
                        f"interface conforming: {len(keys)} faces, one fluid "
                        "element behind each"
                    )

    # --------------------------------------------------------- resolution
    if "fluid" in report.roles:
        from ..build_matrices.domains import identify_domains

        try:
            conn0 = identify_domains(groups)["fluid"]["conn0"]
        except RuntimeError:
            conn0 = None
        sizes = None if conn0 is None else _element_sizes(nodes, conn0)
        if sizes is not None:
            h_min, h_mean, h_max = sizes
            report.notes.append(
                f"fluid element size: {h_min * 1e3:.1f} / "
                f"{h_mean * 1e3:.1f} / {h_max * 1e3:.1f} mm "
                "(min / mean / max)"
            )
            f_1pct = c0 / (13.0 * h_max)
            report.notes.append(
                f"good to ~1% up to {f_1pct:.0f} Hz "
                f"(13 elements per wavelength at c0 = {c0:.0f} m/s)"
            )
            if f_max is not None:
                n_lambda = c0 / (float(f_max) * h_max)
                report.notes.append(
                    f"at {f_max:.0f} Hz: {n_lambda:.1f} elements per "
                    f"wavelength, dispersion error ~"
                    f"{164.0 / n_lambda ** 2:.2f}%"
                )
                if n_lambda < 6.0:
                    report.errors.append(
                        f"only {n_lambda:.1f} elements per wavelength at "
                        f"{f_max:.0f} Hz: below the practical minimum of 6, "
                        "the result is dispersion, not physics."
                    )
                elif n_lambda < 13.0:
                    report.warnings.append(
                        f"{n_lambda:.1f} elements per wavelength at "
                        f"{f_max:.0f} Hz: expect around "
                        f"{164.0 / n_lambda ** 2:.1f}% frequency error; "
                        "13 buys 1%."
                    )

    # ------------------------------------------------------- free groups
    free = [g for g in groups if role_of(g) is None]
    if free:
        report.notes.append(
            "selectable boundaries: " + ", ".join(sorted(free))
        )

    return report


def describe_mesh(mesh, **kwargs):
    """
    Validate and print — the one-liner for a notebook.

    Returns the :class:`MeshReport` as well, so it can still be
    inspected.
    """
    report = validate_mesh(mesh, **kwargs)
    print(report)
    return report
