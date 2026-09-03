pycafe-vibro — structural and vibroacoustic pyCAFE
==================================================

The structural half of pyCAFE: the CQUAD4F shell element, the plate
material, the fluid-structure coupling matrix on a conforming mesh, and
the coupled solvers, direct and modal.

Coupling two meshes that do not share their interface nodes
(``pycafe_vibro.coupling_nonconforming``) ships with it but is **work
in progress**: see ``examples/WIP/`` in the repository.

::

    pip install pycafe-vibro

which installs ``pycafe-acoustics``, the acoustic half, with it — an
acoustic model needs nothing from here, a coupled one needs both. Both
are imported as ``pycafe`` and ``pycafe_vibro``.

Importing the package registers CQUAD4F in pyCAFE's element registry
and re-exports the whole API, acoustic names included::

    import pycafe_vibro as pcv

    spec = pcv.ModelSpec(
        geometry=pcv.Library("box_with_plate", Lx=0.4, Ly=0.3, Lz=0.5),
        f_max=500.0,
        structure=pcv.aluminium(t=2e-3),
    )
    model = pcv.build_model(spec, show=True)

Scripts written against pyCAFE alone keep working: ``pycafe.aluminium``
and the other coupled names resolve here as soon as this package is
installed, and say so plainly when it is not.

The full documentation, the examples and the mesh library live in the
repository: https://github.com/DanFabb/pycafe
