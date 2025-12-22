Code documentation
==================

This section documents the public Python API of pyCAFE.
Only user-facing functions and classes are documented here.

---

The main user interface is provided directly by the ``pycafe`` package.

.. automodule:: pycafe
   :members:
   :undoc-members:

---

Core workflow
-------------

The core workflow utilities implement the main analysis pipeline.

.. automodule:: pycafe.core.load_fluid
   :members:

.. automodule:: pycafe.core.create_geom
   :members:

.. automodule:: pycafe.core.load_mesh
   :members:

.. automodule:: pycafe.core.assign_boundary_conditions
   :members:

.. automodule:: pycafe.core.prepare_acoustic_system
   :members:

.. automodule:: pycafe.core.run_analysis
   :members:

---

Solvers
-------

The solver module provides frequency-domain solution strategies
for the acoustic Helmholtz equation.

.. automodule:: pycafe.solver.solver_helmholtz_1
   :members:

.. automodule:: pycafe.solver.solver_modale
   :members:

---

Materials
---------

Material and fluid property definitions.

.. automodule:: pycafe.material.material
   :members:

---

Boundary conditions
-------------------

Boundary condition definitions and utilities.

.. automodule:: pycafe.boundary_condition.boundary_condition
   :members:
