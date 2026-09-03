pyCAFE — acoustic finite elements
=================================

The acoustic half of pyCAFE: fluid elements (CQUAD4, CQUAD8, CHEXA8,
CTETRA4), acoustic boundary conditions (impedance, velocity,
radiation), the Helmholtz and modal solvers, and the post-processing of
the fields they produce.

The perfectly matched layer (``pycafe.build_matrices.pml``) ships with
it but is **work in progress**: it is not part of the release, and the
examples that exercised it are not either.

::

    pip install pycafe-acoustics

(The distribution is ``pycafe-acoustics``; the bare ``pycafe`` on PyPI
belongs to an unrelated project. The name to import is ``pycafe``.)

A flexible shell, the fluid-structure coupling matrix and the coupled
solvers are the other half, ``pycafe-vibro``, which depends on this one::

    pip install pycafe-vibro     # installs pycafe-acoustics with it

The full documentation, the examples and the mesh library live in the
repository: https://github.com/DanFabb/pycafe
