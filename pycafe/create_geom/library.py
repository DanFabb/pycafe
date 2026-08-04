r"""
Parametric geometries, all tagged the same way.

Four building blocks that cover most of what pyCAFE is used for, each a
plain function returning the path of a written ``.msh``:

======================  =========================================
:func:`box_cavity`      rigid box of fluid, optionally with open faces
:func:`box_with_plate`  the coupled case: cavity closed by a plate
:func:`duct_2d`         2D duct with an inlet and an outlet
:func:`plate`           a shell on its own, clamped or free
======================  =========================================

They all follow :mod:`pycafe.create_geom.conventions`, so a model can be
swapped for another without touching the analysis script:

.. code-block:: python

    from pycafe.create_geom import box_with_plate, load_mesh_with_groups

    msh = box_with_plate(Lx=0.4, Ly=0.3, Lz=0.5, nx=12, ny=9, nz=12)
    nodes, elements, boundaries, groups = load_mesh_with_groups(str(msh))

All of them mesh structured (transfinite) grids, which is what the
fluid-structure coupling needs: the plate quadrilaterals come out as
exactly the top faces of the fluid hexahedra, sharing their nodes, so
the interface is conforming by construction.

Writing your own is the same exercise with :class:`GmshModel` — see the
module header of :mod:`pycafe.create_geom.gmsh_workflow` — and
:func:`pycafe.create_geom.validation.validate_mesh` will tell you
whether it satisfies the contract.
"""

import pathlib

from .gmsh_workflow import GmshModel

FACE_NAMES = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")


def _finish(model, path, dim, verbose, description):
    """Generate and write; shared tail of every builder."""
    model.generate(dim)
    out = model.write(path)
    if verbose:
        print(f"{description} -> {out}")
        print(model.summary())
    return out


def _maybe_load(path, load):
    """Load the mesh once the Gmsh session is closed, if asked."""
    if not load:
        return path

    from .visualize_mesh import load_mesh_with_groups

    return load_mesh_with_groups(str(path), verbose=False)


def _box_entities(model, Lx, Ly, Lz, nx, ny, nz):
    """Transfinite box: returns (volume, faces dict, top edge curves)."""
    geo = model.geo
    p = [
        geo.addPoint(0, 0, 0), geo.addPoint(Lx, 0, 0),
        geo.addPoint(Lx, Ly, 0), geo.addPoint(0, Ly, 0),
        geo.addPoint(0, 0, Lz), geo.addPoint(Lx, 0, Lz),
        geo.addPoint(Lx, Ly, Lz), geo.addPoint(0, Ly, Lz),
    ]
    l1, l2 = geo.addLine(p[0], p[1]), geo.addLine(p[1], p[2])
    l3, l4 = geo.addLine(p[2], p[3]), geo.addLine(p[3], p[0])
    l5, l6 = geo.addLine(p[4], p[5]), geo.addLine(p[5], p[6])
    l7, l8 = geo.addLine(p[6], p[7]), geo.addLine(p[7], p[4])
    l9, l10 = geo.addLine(p[0], p[4]), geo.addLine(p[1], p[5])
    l11, l12 = geo.addLine(p[2], p[6]), geo.addLine(p[3], p[7])

    def surf(loop):
        return geo.addPlaneSurface([geo.addCurveLoop(loop)])

    faces = {
        "z_min": surf([l1, l2, l3, l4]),
        "z_max": surf([l5, l6, l7, l8]),
        "y_min": surf([l1, l10, -l5, -l9]),
        "x_max": surf([l2, l11, -l6, -l10]),
        "y_max": surf([l3, l12, -l7, -l11]),
        "x_min": surf([l4, l9, -l8, -l12]),
    }
    volume = geo.addVolume([geo.addSurfaceLoop(list(faces.values()))])

    model.structured(
        curves={l1: nx + 1, l3: nx + 1, l5: nx + 1, l7: nx + 1,
                l2: ny + 1, l4: ny + 1, l6: ny + 1, l8: ny + 1,
                l9: nz + 1, l10: nz + 1, l11: nz + 1, l12: nz + 1},
        surfaces=list(faces.values()),
        volumes=[volume],
    )
    return volume, faces, (l5, l6, l7, l8)


def box_cavity(
    Lx=0.4,
    Ly=0.3,
    Lz=0.5,
    nx=8,
    ny=6,
    nz=10,
    *,
    output_path="box_cavity.msh",
    open_faces=(),
    face_groups=True,
    load=False,
    verbose=False,
):
    """
    Rigid rectangular cavity, structured CHEXA8.

    The reference acoustic case: analytical modes
    ``f_lmn = c0/2 * sqrt((l/Lx)^2 + (m/Ly)^2 + (n/Lz)^2)`` to check any
    solver against.

    Parameters
    ----------
    Lx, Ly, Lz : float
        Cavity dimensions [m].
    nx, ny, nz : int
        Elements per direction.
    output_path : str or Path, optional
        Destination ``.msh``.
    open_faces : sequence of str, optional
        Faces to group as ``"opening"`` instead of ``"rigid_walls"``,
        named among ``x_min, x_max, y_min, y_max, z_min, z_max``. They
        are still rigid until a boundary condition is put on them —
        naming them is what makes that possible.
    face_groups : bool, optional
        Also tag each face individually under its own name, so a single
        wall can be lined. Default True.
    load : bool, optional
        Return ``(nodes, elements, boundaries, groups)`` instead of the
        path.
    verbose : bool, optional
        Print the summary and let Gmsh log.

    Returns
    -------
    pathlib.Path or tuple
    """
    open_faces = tuple(open_faces)
    unknown = set(open_faces) - set(FACE_NAMES)
    if unknown:
        raise ValueError(
            f"Unknown face name(s) {sorted(unknown)}; expected among "
            f"{list(FACE_NAMES)}."
        )

    with GmshModel("box_cavity", verbose=verbose) as m:
        volume, faces, _ = _box_entities(m, Lx, Ly, Lz, nx, ny, nz)
        m.geo.synchronize()

        m.physical(3, [volume], "fluid")
        rigid = [faces[f] for f in FACE_NAMES if f not in open_faces]
        if rigid:
            m.physical(2, rigid, "rigid_walls")
        if open_faces:
            m.physical(2, [faces[f] for f in open_faces], "opening")
        if face_groups:
            for name in FACE_NAMES:
                m.physical(2, [faces[name]], name)

        out = _finish(m, output_path, 3, verbose,
                      f"cavity {Lx} x {Ly} x {Lz} m, {nx * ny * nz} CHEXA8")

    return _maybe_load(out, load)


def box_with_plate(
    Lx=0.8,
    Ly=0.6,
    Lz=0.5,
    nx=8,
    ny=6,
    nz=5,
    *,
    output_path="box_plate.msh",
    face_groups=False,
    load=False,
    verbose=False,
):
    """
    Cavity closed by a flexible plate on ``z = Lz`` — the coupled case.

    The plate surface is meshed with QUAD4 that are exactly the top
    faces of the hexahedra, which is what makes the mesh conforming: the
    plate nodes *are* the fluid interface nodes, and the coupling
    integral can be assembled face by face without interpolation.

    Groups: ``fluid`` (volume), ``plate`` (top surface — structural
    domain *and* interface), ``rigid_walls`` (the other five faces),
    ``plate_clamp`` (the four edges of the plate).

    Parameters
    ----------
    Lx, Ly, Lz : float
        Cavity dimensions [m]; the plate lies on ``z = Lz``.
    nx, ny, nz : int
        Elements per direction.
    output_path : str or Path, optional
    face_groups : bool, optional
        Also tag the five rigid faces individually. Default False, which
        keeps the group list to the four canonical ones.
    load : bool, optional
        Return the loaded mesh instead of the path.
    verbose : bool, optional

    Returns
    -------
    pathlib.Path or tuple
    """
    with GmshModel("box_plate", verbose=verbose) as m:
        volume, faces, top_edges = _box_entities(m, Lx, Ly, Lz, nx, ny, nz)
        m.geo.synchronize()

        m.physical(3, [volume], "fluid")
        m.physical(2, [faces["z_max"]], "plate")
        m.physical(2, [faces[f] for f in FACE_NAMES if f != "z_max"],
                   "rigid_walls")
        m.physical(1, list(top_edges), "plate_clamp")
        if face_groups:
            for name in FACE_NAMES:
                if name != "z_max":
                    m.physical(2, [faces[name]], name)

        out = _finish(m, output_path, 3, verbose,
                      f"cavity {Lx} x {Ly} x {Lz} m, {nx * ny * nz} CHEXA8 "
                      f"+ plate {nx * ny} CQUAD4")

    return _maybe_load(out, load)


def duct_2d(
    L=1.0,
    H=0.1,
    nx=40,
    ny=4,
    *,
    output_path="duct_2d.msh",
    order=1,
    load=False,
    verbose=False,
):
    """
    2D rectangular duct: fluid surface, inlet, outlet, walls.

    Groups: ``fluid`` (the surface), ``inlet`` (``x = 0``), ``outlet``
    (``x = L``), ``top``/``bottom`` (the long walls). The classical
    check on it is the plane-wave resonance ``c0 / (2 L)``.

    Parameters
    ----------
    L, H : float
        Length and height [m].
    nx, ny : int
        Elements along each direction.
    order : {1, 2}, optional
        1 gives CQUAD4, 2 gives CQUAD8 (incomplete second order, as
        pyCAFE requires).
    output_path : str or Path, optional
    load, verbose : bool, optional

    Returns
    -------
    pathlib.Path or tuple
    """
    with GmshModel("duct_2d", verbose=verbose, order=order) as m:
        geo = m.geo
        p1, p2 = geo.addPoint(0, 0, 0), geo.addPoint(L, 0, 0)
        p3, p4 = geo.addPoint(L, H, 0), geo.addPoint(0, H, 0)
        bottom, right = geo.addLine(p1, p2), geo.addLine(p2, p3)
        top, left = geo.addLine(p3, p4), geo.addLine(p4, p1)
        surface = geo.addPlaneSurface([geo.addCurveLoop(
            [bottom, right, top, left]
        )])

        m.structured(
            curves={bottom: nx + 1, top: nx + 1,
                    right: ny + 1, left: ny + 1},
            surfaces=[surface],
        )
        geo.synchronize()

        m.physical(2, [surface], "fluid")
        m.physical(1, [left], "inlet")
        m.physical(1, [right], "outlet")
        m.physical(1, [top], "top")
        m.physical(1, [bottom], "bottom")

        out = _finish(m, output_path, 2, verbose,
                      f"duct {L} x {H} m, {nx * ny} elements of order {order}")

    return _maybe_load(out, load)


def plate(
    Lx=0.4,
    Ly=0.3,
    nx=12,
    ny=9,
    *,
    output_path="plate.msh",
    clamped=True,
    load=False,
    verbose=False,
):
    """
    A flat shell on its own, for structural checks.

    Groups: ``plate`` (the surface) and, when ``clamped``,
    ``plate_clamp`` (its four edges). Whether that support is a clamp or
    a simple support is decided later, by the ``support`` argument of
    the system preparation — the group only says *which* nodes are held.

    Parameters
    ----------
    Lx, Ly : float
        Plate dimensions [m].
    nx, ny : int
        Elements per direction.
    clamped : bool, optional
        Tag the edges as a support group. Default True.
    output_path : str or Path, optional
    load, verbose : bool, optional

    Returns
    -------
    pathlib.Path or tuple
    """
    with GmshModel("plate", verbose=verbose) as m:
        geo = m.geo
        p1, p2 = geo.addPoint(0, 0, 0), geo.addPoint(Lx, 0, 0)
        p3, p4 = geo.addPoint(Lx, Ly, 0), geo.addPoint(0, Ly, 0)
        bottom, right = geo.addLine(p1, p2), geo.addLine(p2, p3)
        top, left = geo.addLine(p3, p4), geo.addLine(p4, p1)
        surface = geo.addPlaneSurface([geo.addCurveLoop(
            [bottom, right, top, left]
        )])

        m.structured(
            curves={bottom: nx + 1, top: nx + 1,
                    right: ny + 1, left: ny + 1},
            surfaces=[surface],
        )
        geo.synchronize()

        m.physical(2, [surface], "plate")
        if clamped:
            m.physical(1, [bottom, right, top, left], "plate_clamp")

        out = _finish(m, output_path, 2, verbose,
                      f"plate {Lx} x {Ly} m, {nx * ny} CQUAD4")

    return _maybe_load(out, load)


FACE_AXIS = {"x_min": (0, 0), "x_max": (0, 1), "y_min": (1, 0),
             "y_max": (1, 1), "z_min": (2, 0), "z_max": (2, 1)}


def _cad_box(model, cad_path, units="auto", tol=1e-2):
    """
    Import a CAD solid and return its box in metres.

    Rejects anything that is not box-shaped, since the structured
    (hexahedral) meshing below would silently fall back to tetrahedra,
    which pyCAFE has no acoustic kernel for.
    """
    import numpy as np

    model.occ.importShapes(str(cad_path))
    model.occ.synchronize()

    volumes = model.model.getEntities(3)
    if len(volumes) != 1:
        raise ValueError(
            f"{cad_path} holds {len(volumes)} solids; this builder expects "
            "exactly one."
        )

    bb = np.array(model.model.getBoundingBox(-1, -1))
    extent = bb[3:] - bb[:3]
    if units == "auto":
        # A duct is metres-sized: an extent in the hundreds means the file
        # is in millimetres, which is what most CAD exports use.
        scale = 1e-3 if extent.max() > 100.0 else 1.0
    else:
        scale = {"m": 1.0, "mm": 1e-3, "cm": 1e-2}[units]
    if scale != 1.0:
        model.occ.dilate(volumes, 0, 0, 0, scale, scale, scale)
        model.occ.synchronize()
        bb = np.array(model.model.getBoundingBox(-1, -1))

    lo, hi = bb[:3], bb[3:]
    lengths = hi - lo
    volume = model.occ.getMass(3, volumes[0][1])
    if abs(volume - float(np.prod(lengths))) > tol * float(np.prod(lengths)):
        raise ValueError(
            f"{cad_path} is not box-shaped (volume {volume:.4g} m3 against "
            f"{np.prod(lengths):.4g} m3 for its bounding box). Structured "
            "hexahedral meshing needs a box; a general shape would come out "
            "tetrahedral, and pyCAFE has no tetrahedral acoustic element."
        )
    return lo, hi, lengths, scale


def duct_with_flush_plate(
    cad_path=None,
    *,
    lengths=None,
    origin=(0.0, 0.0, 0.0),
    plate_length=1.0,
    plate_width=None,
    plate_face="z_min",
    plate_centre=None,
    end_groups=True,
    element_size=None,
    f_max=None,
    c0=343.0,
    elements_per_wavelength=13,
    units="auto",
    output_path="duct_flush_plate.msh",
    load=False,
    validate=True,
    verbose=False,
):
    r"""
    Rectangular duct with a **flush-mounted** flexible panel on one wall.

    The panel is not an added part: it is a patch of the wall itself, so
    it lies flush with the rigid surface around it and shares its nodes
    with the fluid. The duct is cut into boxes at the edges of the patch
    and every piece is meshed transfinitely, which keeps the whole model
    hexahedral and the interface conforming — the panel quadrilaterals
    are exactly the faces of the fluid hexahedra behind them.

    The geometry can come from a **CAD file** (STEP/IGES/BREP), in which
    case its bounding box is read, the units are detected, and the
    solid is checked for being a box. Or it can be given directly with
    ``lengths``.

    Parameters
    ----------
    cad_path : str or Path, optional
        CAD file to import. Mutually exclusive with ``lengths``.
    lengths : (3,) sequence of float, optional
        Duct dimensions [m] when no CAD file is given.
    origin : (3,) sequence of float, optional
        Corner of the duct when built from ``lengths``.
    plate_length : float, optional
        Extent of the panel **along the duct axis** (the longest
        dimension) [m].
    plate_width : float, optional
        Extent across the axis [m]. Default: the full width, i.e. a
        panel spanning the wall from side to side.
    plate_face : str, optional
        Which wall carries it: ``z_min`` (floor, the default), ``z_max``
        (ceiling), ``y_min``, ``y_max``, ``x_min``, ``x_max``.
    plate_centre : float, optional
        Position of the panel centre along the axis [m]. Default: the
        middle of the duct.
    end_groups : bool, optional
        Tag the two end faces separately as ``inlet`` (low end of the
        axis) and ``outlet`` (high end), so they can carry their own
        boundary condition — an incident wave on one side, an anechoic
        termination on the other. With ``False`` they join
        ``rigid_walls`` and the duct is closed. Default True.
    element_size : float, optional
        Target element size [m]. Mutually exclusive with ``f_max``.
    f_max : float, optional
        Highest frequency of interest [Hz]; the size then follows from
        ``c0 / (f_max * elements_per_wavelength)``.
    c0, elements_per_wavelength : float, optional
    units : {"auto", "m", "mm", "cm"}, optional
        Unit of the CAD file. ``"auto"`` calls anything bigger than 100
        units millimetres.
    output_path : str or Path, optional
    load : bool, optional
        Return the loaded mesh instead of the path.
    validate : bool, optional
        Also return a :class:`~pycafe.create_geom.validation.MeshReport`.
    verbose : bool, optional

    Returns
    -------
    path (or loaded mesh), and the report when ``validate``

    Raises
    ------
    ValueError
        If the CAD solid is not a box, if neither or both of
        ``cad_path``/``lengths`` are given, or if the panel does not fit
        on the chosen wall.

    Notes
    -----
    Groups written: ``fluid`` (every sub-volume), ``plate`` (the patch),
    ``rigid_walls`` (the side walls), ``inlet``/``outlet`` (the two ends,
    unless ``end_groups=False``) and ``plate_clamp`` (the four edges of
    the patch). Whether that support behaves as clamped
    (C-C-C-C) or simply supported is chosen later, by the ``support``
    argument of ``prepare_vibroacoustic_system`` — the mesh only says
    which nodes are held.
    """
    import numpy as np

    if (cad_path is None) == (lengths is None):
        raise ValueError("Give either 'cad_path' or 'lengths'.")
    if element_size is not None and f_max is not None:
        raise ValueError("Give either 'element_size' or 'f_max', not both.")
    if plate_face not in FACE_AXIS:
        raise ValueError(f"plate_face must be one of {sorted(FACE_AXIS)}.")

    with GmshModel("duct_flush_plate", verbose=verbose) as m:
        if cad_path is not None:
            lo, hi, size, scale = _cad_box(m, cad_path, units=units)
        else:
            lo = np.asarray(origin, dtype=float)
            size = np.asarray(lengths, dtype=float)
            hi = lo + size
            m.occ.addBox(*lo, *size)
            m.occ.synchronize()
            scale = 1.0

        tol = 1e-5 * float(size.max())
        axis = int(np.argmax(size))                 # the duct runs this way
        normal_axis, side = FACE_AXIS[plate_face]
        if normal_axis == axis:
            raise ValueError(
                f"plate_face '{plate_face}' is an end of the duct (its axis "
                f"is {'xyz'[axis]}); mount the panel on a side wall."
            )
        cross = [i for i in range(3) if i not in (axis, normal_axis)][0]

        centre = (0.5 * (lo[axis] + hi[axis]) if plate_centre is None
                  else float(plate_centre))
        a0, a1 = centre - plate_length / 2.0, centre + plate_length / 2.0
        if a0 < lo[axis] - tol or a1 > hi[axis] + tol:
            raise ValueError(
                f"a {plate_length} m panel centred at {centre} m does not fit "
                f"in a duct spanning {lo[axis]:.3f}..{hi[axis]:.3f} m."
            )

        width = size[cross] if plate_width is None else float(plate_width)
        c_mid = 0.5 * (lo[cross] + hi[cross])
        c0_, c1_ = c_mid - width / 2.0, c_mid + width / 2.0
        if c0_ < lo[cross] - tol or c1_ > hi[cross] + tol:
            raise ValueError(
                f"a {width} m wide panel does not fit across "
                f"{size[cross]:.3f} m."
            )

        # Cut the duct at the edges of the patch, so the patch is a whole
        # face of one sub-volume and the mesh stays structured.
        cuts = [(axis, a0), (axis, a1)]
        if plate_width is not None and width < size[cross] - tol:
            cuts += [(cross, c0_), (cross, c1_)]

        cutters = []
        for direction, value in cuts:
            in_plane = [i for i in range(3) if i != direction]
            corners = []
            for du, dv in ((0, 0), (1, 0), (1, 1), (0, 1)):
                point = np.empty(3)
                point[direction] = value
                point[in_plane[0]] = hi[in_plane[0]] if du else lo[in_plane[0]]
                point[in_plane[1]] = hi[in_plane[1]] if dv else lo[in_plane[1]]
                corners.append(m.occ.addPoint(*point))
            curves = [m.occ.addLine(corners[i], corners[(i + 1) % 4])
                      for i in range(4)]
            cutters.append(
                (2, m.occ.addPlaneSurface([m.occ.addCurveLoop(curves)]))
            )

        m.occ.fragment(m.model.getEntities(3), cutters)
        m.occ.synchronize()

        # Element size, then a structured mesh everywhere.
        if element_size is None and f_max is None:
            element_size = float(size[[axis, cross, normal_axis]].min()) / 8.0
        h = (float(element_size) if element_size is not None
             else c0 / (float(f_max) * elements_per_wavelength))
        for _, curve in m.model.getEntities(1):
            bb = np.array(m.model.getBoundingBox(1, curve))
            n = max(2, int(round(float(np.linalg.norm(bb[3:] - bb[:3])) / h)) + 1)
            m.mesh.setTransfiniteCurve(curve, n)
        for _, surface in m.model.getEntities(2):
            m.mesh.setTransfiniteSurface(surface)
            m.mesh.setRecombine(2, surface)
        for _, volume in m.model.getEntities(3):
            m.mesh.setTransfiniteVolume(volume)

        # Classify the faces: outer ones carry one volume, the cuts two.
        plane = hi[normal_axis] if side else lo[normal_axis]
        plate_faces, wall_faces, inlet_faces, outlet_faces = [], [], [], []
        for _, surface in m.model.getEntities(2):
            com = np.array(m.occ.getCenterOfMass(2, surface))
            attached, _ = m.model.getAdjacencies(2, surface)
            if len(attached) != 1:
                continue                    # internal cut, not a boundary
            on_wall = abs(com[normal_axis] - plane) < tol
            in_patch = (a0 - tol < com[axis] < a1 + tol
                        and c0_ - tol < com[cross] < c1_ + tol)
            if on_wall and in_patch:
                plate_faces.append(surface)
            elif end_groups and abs(com[axis] - lo[axis]) < tol:
                inlet_faces.append(surface)
            elif end_groups and abs(com[axis] - hi[axis]) < tol:
                outlet_faces.append(surface)
            else:
                wall_faces.append(surface)

        if not plate_faces:
            raise ValueError(
                "no face of the duct matched the panel; check plate_face, "
                "plate_centre and the CAD units."
            )

        m.physical(3, [v for _, v in m.model.getEntities(3)], "fluid")
        m.physical(2, plate_faces, "plate")
        m.physical(2, wall_faces, "rigid_walls")
        if inlet_faces:
            m.physical(2, inlet_faces, "inlet")
        if outlet_faces:
            m.physical(2, outlet_faces, "outlet")
        clamp = sorted({abs(t) for _, t in m.model.getBoundary(
            [(2, f) for f in plate_faces], oriented=False, combined=True
        )})
        m.physical(1, clamp, "plate_clamp")

        if verbose:
            print(f"duct {np.round(size, 3)} m (axis {'xyz'[axis]}, "
                  f"units scaled by {scale}), panel {plate_length} x {width} m "
                  f"on {plate_face}, element size {h * 1e3:.0f} mm")

        out = _finish(m, output_path, 3, verbose, "duct with flush plate")

    result = _maybe_load(out, load)
    if not validate:
        return result

    from .validation import validate_mesh

    report = validate_mesh(str(out), c0=c0, f_max=f_max)
    if verbose:
        print(report)
    return result, report


GEOMETRIES = {
    "box_cavity": box_cavity,
    "box_with_plate": box_with_plate,
    "duct_2d": duct_2d,
    "duct_with_flush_plate": duct_with_flush_plate,
    "plate": plate,
}


def build(name, **kwargs):
    """
    Build a geometry by name — handy for parameter studies and scripts.

    Parameters
    ----------
    name : str
        One of ``box_cavity``, ``box_with_plate``, ``duct_2d``, ``plate``.
    **kwargs
        Passed to the builder.

    Returns
    -------
    pathlib.Path or tuple
    """
    try:
        builder = GEOMETRIES[name]
    except KeyError:
        raise ValueError(
            f"Unknown geometry '{name}'. Available: {sorted(GEOMETRIES)}."
        ) from None
    return builder(**kwargs)


def default_path(name, directory="."):
    """Conventional file name for a geometry, inside ``directory``."""
    return pathlib.Path(directory) / f"{name}.msh"
