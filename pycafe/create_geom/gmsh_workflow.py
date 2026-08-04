r"""
A standard way to write a Gmsh script for pyCAFE.

Every mesh generator ends up repeating the same boilerplate:
``initialize`` / ``finalize`` in a ``try/finally``, silence the terminal,
set the element order, remember ``SecondOrderIncomplete`` for CQUAD8,
recombine into quads, tag the physical groups with the right names,
generate, write. :class:`GmshModel` does that once so a geometry script
is only geometry:

.. code-block:: python

    from pycafe.create_geom import GmshModel

    with GmshModel("duct", order=1, recombine=True) as m:
        rect = m.geo.addPlaneSurface([m.geo.addCurveLoop([...])])
        m.geo.synchronize()
        m.physical(2, rect, "fluid")            # role name, checked
        m.physical(1, [left], "inlet")          # free name, kept as is
        m.generate(2)
        path = m.write("duct.msh")

What the class adds beyond convenience:

* the group names are checked against
  :mod:`pycafe.create_geom.conventions` — a typo like ``"fluids"`` is
  caught when the mesh is written, not three functions later when
  ``identify_domains`` fails to find a domain;
* ``order=2`` automatically sets ``Mesh.SecondOrderIncomplete``, without
  which Gmsh writes 9-node quadrangles that pyCAFE has no element for;
* ``structured()`` wraps the transfinite/recombine dance that produces
  the conforming hexa+quad meshes the coupling needs;
* the model remembers what it tagged, so ``summary()`` prints the
  contract the mesh actually fulfils.

See Also
--------
pycafe.create_geom.library : Ready-made parametric geometries.
pycafe.create_geom.validation : Checking a mesh before running on it.
pycafe.create_geom.external : Bringing in CAD or third-party meshes.
"""

import pathlib

from .conventions import ROLE_BY_NAME, role_of


class GmshModel:
    """
    Context manager around a Gmsh model, with pyCAFE's conventions built in.

    Parameters
    ----------
    name : str, optional
        Model name.
    verbose : bool, optional
        Let Gmsh print its meshing log. Default False.
    order : {1, 2}, optional
        Element order. Order 2 also sets ``Mesh.SecondOrderIncomplete``
        (8-node quads, not 9) unless ``complete_second_order=True``.
    recombine : bool, optional
        Recombine triangles into quadrilaterals everywhere
        (``Mesh.RecombineAll``). Per-surface control is available
        through :meth:`structured`.
    complete_second_order : bool, optional
        Keep the centre node of second-order quads. pyCAFE has no
        9-node element, so this is only useful for export.
    options : dict, optional
        Extra Gmsh options, e.g. ``{"Mesh.Algorithm": 8}``, applied
        after the ones above.

    Attributes
    ----------
    geo, occ, model, mesh
        The Gmsh sub-APIs, so a script never imports gmsh itself.
    groups : dict
        ``{name: (dim, tag)}`` of everything tagged so far.

    Raises
    ------
    RuntimeError
        If used outside its ``with`` block.
    """

    def __init__(
        self,
        name="model",
        *,
        verbose=False,
        order=1,
        recombine=False,
        complete_second_order=False,
        options=None,
    ):
        self.name = str(name)
        self.verbose = bool(verbose)
        self.order = int(order)
        self.recombine = bool(recombine)
        self.complete_second_order = bool(complete_second_order)
        self.options = dict(options or {})
        self.groups = {}
        self._gmsh = None

    # ------------------------------------------------------------------
    # context manager
    # ------------------------------------------------------------------
    def __enter__(self):
        import gmsh

        self._gmsh = gmsh
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1 if self.verbose else 0)
        gmsh.model.add(self.name)

        gmsh.option.setNumber("Mesh.ElementOrder", self.order)
        if self.order >= 2 and not self.complete_second_order:
            # Without this Gmsh writes "Quadrangle 9"; pyCAFE's CQUAD8
            # expects the 8-node serendipity element.
            gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 1)
        if self.recombine:
            gmsh.option.setNumber("Mesh.RecombineAll", 1)
        for key, value in self.options.items():
            gmsh.option.setNumber(key, value)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._gmsh is not None:
            self._gmsh.finalize()
            self._gmsh = None
        return False

    @property
    def gmsh(self):
        """The gmsh module, only inside the ``with`` block."""
        if self._gmsh is None:
            raise RuntimeError(
                "GmshModel is only usable inside its 'with' block."
            )
        return self._gmsh

    @property
    def geo(self):
        """``gmsh.model.geo`` — the built-in CAD kernel."""
        return self.gmsh.model.geo

    @property
    def occ(self):
        """``gmsh.model.occ`` — the OpenCASCADE kernel (STEP, booleans)."""
        return self.gmsh.model.occ

    @property
    def model(self):
        """``gmsh.model``."""
        return self.gmsh.model

    @property
    def mesh(self):
        """``gmsh.model.mesh``."""
        return self.gmsh.model.mesh

    # ------------------------------------------------------------------
    # tagging
    # ------------------------------------------------------------------
    def physical(self, dim, tags, name, *, strict=True):
        """
        Tag entities as a physical group, checking the name.

        Parameters
        ----------
        dim : int
            Dimension of the entities (3 volume, 2 surface, 1 curve).
        tags : int or sequence of int
            Entity tags.
        name : str
            Group name. A role name (``fluid``, ``plate``, ``clamp``,
            ...) is checked for consistency: the role's expected element
            field cannot be satisfied by an entity of the wrong
            dimension, and a role cannot be tagged twice.
        strict : bool, optional
            Raise on a suspicious name (default) or only warn.

        Returns
        -------
        int
            The physical group tag.

        Raises
        ------
        ValueError
            If the same name, or the same role, is tagged twice, or if a
            role name is used on an implausible dimension.
        """
        tags = [int(tags)] if isinstance(tags, (int,)) else [int(t) for t in tags]
        key = str(name)

        if key in self.groups:
            raise ValueError(
                f"Physical group '{key}' already exists at "
                f"{self.groups[key]}. Names must be unique: pyCAFE selects "
                "boundaries by name."
            )

        role = role_of(key)
        if role is not None:
            already = [n for n in self.groups if role_of(n) == role]
            if already:
                raise ValueError(
                    f"The '{role}' role is already taken by group "
                    f"'{already[0]}'. Only one group per role: the solvers "
                    "look it up by name."
                )
            spec = ROLE_BY_NAME[key.strip().lower()]
            if spec.role == "clamp" and dim > 2:
                message = (
                    f"'{key}' marks the support of the structure, expected "
                    f"on a curve (3D) or a point (2D), not on dim {dim}."
                )
                self._complain(message, strict)
            if spec.role in ("fluid", "structure") and dim < 2:
                message = (
                    f"'{key}' is a domain group and needs elements of "
                    f"dimension 2 or 3, not {dim}."
                )
                self._complain(message, strict)

        tag = self.model.addPhysicalGroup(dim, tags, name=key)
        self.groups[key] = (dim, tag)
        return tag

    @staticmethod
    def _complain(message, strict):
        if strict:
            raise ValueError(message + " Pass strict=False to override.")
        import warnings

        warnings.warn(message, RuntimeWarning)

    # ------------------------------------------------------------------
    # meshing helpers
    # ------------------------------------------------------------------
    def structured(self, curves=None, surfaces=None, volumes=None):
        """
        Set up a transfinite (structured) mesh.

        This is what produces the conforming hexa + quad meshes the
        fluid-structure coupling needs: the plate quads come out as
        exactly the faces of the fluid hexahedra, sharing their nodes.

        Parameters
        ----------
        curves : dict, optional
            ``{curve_tag: number_of_nodes}``. Gmsh counts *nodes*, so a
            side with ``n`` elements takes ``n + 1``.
        surfaces : sequence of int, optional
            Surfaces made transfinite and recombined into quads.
        volumes : sequence of int, optional
            Volumes made transfinite (hexahedra).
        """
        for tag, n_nodes in (curves or {}).items():
            self.geo.mesh.setTransfiniteCurve(int(tag), int(n_nodes))
        for tag in surfaces or ():
            self.geo.mesh.setTransfiniteSurface(int(tag))
            self.geo.mesh.setRecombine(2, int(tag))
        for tag in volumes or ():
            self.geo.mesh.setTransfiniteVolume(int(tag))

    def size(self, target, *, min_size=None, max_size=None):
        """
        Set the characteristic element size.

        Parameters
        ----------
        target : float
            Target element size [m]. Also used as the max when
            ``max_size`` is not given.
        min_size, max_size : float, optional
            Bounds passed to ``Mesh.MeshSizeMin`` / ``Mesh.MeshSizeMax``.
        """
        self.gmsh.option.setNumber("Mesh.MeshSizeMin",
                                   float(min_size if min_size else target))
        self.gmsh.option.setNumber("Mesh.MeshSizeMax",
                                   float(max_size if max_size else target))

    def size_for_frequency(self, f_max, c0=343.0, elements_per_wavelength=13):
        """
        Element size that resolves ``f_max`` with the chosen density.

        The default of 13 elements per wavelength is the 1% dispersion
        error of the trilinear acoustic element (the error goes roughly
        as ``164 / n_lambda^2``); 6 is the usual bare minimum, 41 buys
        0.1%.

        Returns
        -------
        float
            The size, in metres, after applying it to the model.
        """
        h = float(c0) / (float(f_max) * float(elements_per_wavelength))
        self.size(h)
        return h

    def synchronize(self, kernel="geo"):
        """Synchronize the CAD kernel that built the geometry."""
        if kernel == "occ":
            self.occ.synchronize()
        elif kernel == "geo":
            self.geo.synchronize()
        elif kernel is not None:
            raise ValueError("kernel must be 'geo', 'occ' or None.")

    def generate(self, dim=3, *, kernel="geo"):
        """
        Mesh the model.

        Parameters
        ----------
        dim : int
            Topological dimension to mesh.
        kernel : {"geo", "occ", None}, optional
            Which CAD kernel to synchronize first. Use ``"occ"`` after
            importing a STEP file, ``None`` if already synchronized.
        """
        self.synchronize(kernel)
        self.mesh.generate(int(dim))

    def write(self, path):
        """
        Write the ``.msh`` file.

        Returns
        -------
        pathlib.Path
        """
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.gmsh.write(str(path))
        return path

    def show(self):
        """Open the Gmsh GUI on the current model (blocking)."""
        self.gmsh.fltk.run()

    # ------------------------------------------------------------------
    # reporting
    # ------------------------------------------------------------------
    def summary(self):
        """Printable list of what has been tagged, with the roles found."""
        if not self.groups:
            return "No physical group tagged yet."
        lines = ["Physical groups:"]
        for name, (dim, tag) in self.groups.items():
            role = role_of(name)
            label = f"role: {role}" if role else "boundary name"
            lines.append(f"  dim {dim}  tag {tag:3d}  {name:<14} ({label})")
        missing = [r for r in ("fluid", "structure")
                   if not any(role_of(n) == r for n in self.groups)]
        if missing:
            lines.append(f"  no group for: {', '.join(missing)}")
        return "\n".join(lines)
