pyCAFE
======

**pyCAFE** is a Python based 2D finite element framework for solving acoustic problems in the frequency domain.

The library is designed for research and engineering applications in acoustics. It provides a clear and modular workflow for defining the
fluid medium, geometry, boundary conditions, assembling the acoustic finite
element system, and solving the Helmholtz equation.

pyCAFE is particularly suited for  numerical studies
and for extending classical acoustic FEM formulations toward more advanced
models.

The authors welcome contributions from the community to enhance and expand the software's capabilities. Even if the studies behind the writing of the software
were broad and deep, there is always room for improvement and new features. A lot of work is still needed to make pyCAFE a fully-featured acoustic FEM package.

---

Typical workflow
----------------

A typical pyCAFE analysis follows an explicit and modular pipeline:

0. Create geometry and mesh (optional step)
1. Load fluid properties
2. Define the computational geometry
3. Load the finite element mesh
4. Assign boundary conditions
5. Assemble the acoustic finite element system
6. Solve the Helmholtz equation in the frequency domain

Each step is handled by a dedicated function, making the workflow easy to
understand, debug, and extend.

---

Example
-------

The following example illustrates a complete acoustic simulation workflow using
pyCAFE:

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


---

Standardized input
------------------

The steps above can also be stated once, as a ``ModelSpec``: where the
geometry comes from, what the materials are, and the highest frequency of
interest. The element size is *derived* from that frequency,
``h = c0 / (f_max * n_lambda)`` with at least 10 elements per wavelength, so a
model is never quietly run on a mesh too coarse for the band it is swept over.

.. code-block:: python

    from pycafe.core.model_spec import (
        AIR, CadFile, Library, MeshFile, ModelSpec, aluminium, build_model,
    )

    # Geometry we build, from parameters
    spec = ModelSpec(
        geometry=Library("box_with_plate", Lx=0.4, Ly=0.3, Lz=0.5),
        fluid=AIR,
        structure=aluminium(t=2e-3, support="clamped"),
        f_max=500.0,
    )
    print(spec.describe())

    model = build_model(spec)      # mesh -> validate -> assemble
    system = model["system"]

Geometry the user brings in is the same call with a different source. A CAD
file carries no roles, so they are given explicitly, after asking the file what
is in it:

.. code-block:: python

    from pycafe import inspect_cad

    inspect_cad("Geom/Tubo_1m_1m.stp")     # tags, sizes, bounding boxes
    #   dim 3  tag  1   volume 3.000e+00   bbox 3.000 x 1.000 x 1.000
    #   dim 2  tag  5   area   1.000e+00   bbox 1.000 x 1.000 x 0.000

    spec = ModelSpec(
        geometry=CadFile("Geom/Tubo_1m_1m.stp", fluid=[1], structure=[5],
                         units="auto"),     # mm files are detected and scaled
        structure=aluminium(t=2e-3),
        f_max=400.0,
    )

and a mesh that already exists is ``MeshFile("from_ansys.msh",
rename={"AIR": "fluid", "SKIN": "plate"})``.

---

Project structure
-----------------

The core package is organized as follows:

::

    pycafe/
        boundary_condition/    Acoustic boundary condition definitions
        build_matrices/        Finite element matrix assembly
        core/                  Workflow utilities
        create_geom/           Geometry definition and mesh visualization
        material/              Fluid and material models
        solver/                Helmholtz and modal solvers
        post_processing/       Post processing and visualization tools

    tests/
        Basic unit tests

    docs/
        Sphinx documentation source and build files

---

Documentation
-------------

The project documentation is built using Sphinx and can be hosted via
Read the Docs.

To build the documentation locally, run the following commands from the project
root directory:

.. code-block:: console

    cd docs
    make clean
    make html

The generated HTML documentation will be available in
``docs/build/html``.

---

Development and testing
-----------------------

Unit tests are located in the ``tests/`` directory and can be run using:

.. code-block:: console

    pytest

Contributions, bug reports, and feature requests are welcome.

---

License
-------

pyCAFE is released under the MIT License.
See the ``LICENSE`` file for more information.


---

DOI :

[![DOI](https://zenodo.org/badge/1121240895.svg)](https://doi.org/10.5281/zenodo.19599893)
