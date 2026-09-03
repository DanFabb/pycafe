pyCAFE
======

**pyCAFE** is a finite element framework written in Python for solving acoustic
problems in the frequency domain.

The library is designed for research and engineering applications in acoustics.
It provides a clear and modular workflow for defining the fluid medium,
geometry and boundary conditions, assembling the acoustic finite element
system, and solving the Helmholtz equation.

pyCAFE is particularly suited to numerical studies and to the development and
extension of classical acoustic finite element formulations toward more
advanced models.

Contributions from the community are welcome and can help improve and extend
the capabilities of the software. Although the development of pyCAFE is based
on extensive numerical and methodological work, several features can still be
improved or expanded. Further development is therefore ongoing with the aim of
making pyCAFE a more complete acoustic finite element package.


Two packages, one repository
============================

The repository contains two Python distributions. This allows purely acoustic
applications to install only the components required for acoustic analyses,
while vibroacoustic applications can additionally install the structural and
coupling capabilities.


pycafe acoustics
~~~~~~~~~~~~~~~~

The acoustic package is installed with:

.. code-block:: bash

    pip install pycafe-acoustics

It contains the acoustic finite element formulation, including CQUAD4, CQUAD8,
CHEXA8 and CTETRA4 fluid elements, acoustic boundary conditions such as
impedance, prescribed velocity and radiation conditions, Helmholtz and modal
solvers, and post processing tools.

The package is imported as:

.. code-block:: python

    import pycafe

Its source code is located in:

.. code-block:: text

    packages/pycafe


pycafe vibro
~~~~~~~~~~~~

The vibroacoustic package is installed with:

.. code-block:: bash

    pip install pycafe-vibro

It contains the structural and vibroacoustic capabilities, including the
CQUAD4F shell element, plate material definitions, conforming fluid structure
coupling and coupled vibroacoustic solvers.

The package depends on ``pycafe-acoustics``, so installing ``pycafe-vibro``
also installs the acoustic package.

It is imported as:

.. code-block:: python

    import pycafe_vibro

Its source code is located in:

.. code-block:: text

    packages/pycafe-vibro

Two features are currently included in the source code but should be considered
work in progress and are not part of the stable release. These are the
perfectly matched layer implementation in ``pycafe.build_matrices.pml`` and
the coupling between meshes that do not share interface nodes in
``pycafe_vibro.coupling_nonconforming``.

Examples related to features that are still under development are available in:

.. code-block:: text

    examples/WIP/

The acoustic distribution is named ``pycafe-acoustics`` because the name
``pycafe`` on PyPI is already assigned to an unrelated project. The Python
package itself is nevertheless imported as ``pycafe``.

When ``pycafe_vibro`` is imported, its structural elements are registered in
the pyCAFE element registry and the acoustic API is also made available.
Therefore,

.. code-block:: python

    import pycafe_vibro as pcv

provides access to the complete vibroacoustic API.

Scripts written using ``pycafe`` remain compatible with the acoustic package.
If a functionality belonging to the vibroacoustic package is requested from an
installation containing only ``pycafe-acoustics``, pyCAFE identifies the
package in which that functionality is defined.

When working directly from the repository, both packages can be installed in
editable mode with:

.. code-block:: bash

    pip install -e packages/pycafe -e packages/pycafe-vibro


Typical workflow
================

A typical pyCAFE acoustic analysis follows an explicit and modular sequence.

The geometry and mesh can first be created if required. The fluid properties
are then defined, followed by the computational geometry and finite element
mesh. Boundary conditions are assigned to the relevant parts of the model.
The acoustic finite element matrices are subsequently assembled and the
Helmholtz equation is solved in the frequency domain.

Each stage is handled by dedicated functions, making the numerical workflow
easy to understand, inspect, debug and extend.


Example
=======

The following example illustrates a complete acoustic simulation workflow
using pyCAFE:

.. code-block:: python

    import pycafe

    # Load fluid properties

    fluid = pycafe.load_fluid("air")

    # Create geometry

    msh_file = pycafe.create_geometry(c0=fluid["c0"])

    # Load mesh

    nodes, elements, boundaries = pycafe.load_mesh("rectangle_CQUAD8.msh")

    # Assign boundary conditions

    bc = pycafe.assign_boundary_conditions(
        boundaries=boundaries,
        nodes=nodes,
    )

    # Assemble the acoustic FEM system

    system = pycafe.prepare_acoustic_system(
        nodes=nodes,
        elements=elements,
        boundaries=boundaries,
        rho=fluid["rho"],
        c0=fluid["c0"],
        bc=bc,
        debug=True,
    )

    # Solve the acoustic problem

    results = pycafe.run_analysis(
        system=system,
        nodes=nodes,
        boundaries=boundaries,
        rho=fluid["rho"],
    )


Standardized input
==================

A pyCAFE model can also be defined through a ``ModelSpec`` object. This
provides a standardized description of the geometry source, materials,
structural properties and maximum frequency of interest.

The characteristic element size can be derived automatically from the maximum
frequency according to

.. math::

    h = \frac{c_0}{f_{\max} n_{\lambda}}

where ``n_lambda`` defines the required number of elements per wavelength.
Using at least ten elements per wavelength provides a conservative default for
the spatial discretization.

A model based on a geometry generated directly by pyCAFE can be defined as
follows:

.. code-block:: python

    from pycafe.core.model_spec import (
        AIR,
        CadFile,
        Library,
        MeshFile,
        ModelSpec,
        aluminium,
        build_model,
    )

    spec = ModelSpec(
        geometry=Library(
            "box_with_plate",
            Lx=0.4,
            Ly=0.3,
            Lz=0.5,
        ),
        fluid=AIR,
        structure=aluminium(
            t=2e-3,
            support="clamped",
        ),
        f_max=500.0,
    )

    print(spec.describe())

    model = build_model(spec)

    system = model["system"]


External CAD geometry
=====================

Geometry provided by the user can be handled through the same workflow by
selecting a different geometry source.

Because a generic CAD file does not inherently contain the physical roles
required by the finite element model, the entities contained in the file can
first be inspected:

.. code-block:: python

    from pycafe import inspect_cad

    inspect_cad("Library/Tubo_1m_1m.stp")

An example output is:

.. code-block:: text

    dim 3  tag 1  volume 3.000e+00  bbox 3.000 x 1.000 x 1.000
    dim 2  tag 5  area   1.000e+00  bbox 1.000 x 1.000 x 0.000

The corresponding model can then be defined by explicitly assigning the
physical roles of the CAD entities:

.. code-block:: python

    spec = ModelSpec(
        geometry=CadFile(
            "Library/Tubo_1m_1m.stp",
            fluid=[1],
            structure=[5],
            units="auto",
        ),
        structure=aluminium(t=2e-3),
        f_max=400.0,
    )

When ``units="auto"`` is used, CAD geometries expressed in millimetres can be
detected and converted automatically.


Existing meshes
===============

An existing finite element mesh can be introduced through ``MeshFile``.

For example:

.. code-block:: python

    MeshFile(
        "from_ansys.msh",
        rename={
            "AIR": "fluid",
            "SKIN": "plate",
        },
    )


Project structure
=================

The main acoustic package is organized as follows:

.. code-block:: text

    pycafe/
        boundary_condition/    Acoustic boundary condition definitions
        build_matrices/        Finite element matrix assembly
        core/                  Workflow and model definition utilities
        create_geom/           Geometry creation and mesh visualization
        material/              Fluid and material models
        solver/                Helmholtz and modal solvers
        post_processing/       Post processing and visualization tools

    tests/
        Unit and integration tests

    docs/
        Sphinx documentation source files


Documentation
=============

The project documentation is built using Sphinx and can be hosted through
Read the Docs.

To build the documentation locally, run the following commands from the
project root directory:

.. code-block:: console

    cd docs
    make clean
    make html

The generated HTML documentation is available in:

.. code-block:: text

    docs/build/html


Development and testing
=======================

The test suite is located in the ``tests/`` directory and can be executed with:

.. code-block:: console

    pytest

Contributions, bug reports and feature requests are welcome.


License
=======

pyCAFE is distributed under the MIT License.

See the ``LICENSE`` file for additional information.


DOI
===

.. image:: https://zenodo.org/badge/1121240895.svg
   :target: https://doi.org/10.5281/zenodo.19599893
   :alt: DOI