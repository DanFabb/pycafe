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

.. automodule:: pycafe.core.prepare_vibroacoustic_system
   :members:

---

Matrix assembly
---------------

Element registry, generic sparse assembly and boundary-condition
operations.

.. automodule:: pycafe.build_matrices.element_registry
   :members:

.. automodule:: pycafe.build_matrices.assembly
   :members:

.. automodule:: pycafe.build_matrices.element_hex8
   :members:

.. automodule:: pycafe.build_matrices.element_cquad4f
   :members:

.. automodule:: pycafe.build_matrices.bc_ops
   :members:

---

Vibroacoustic coupling
----------------------

Fluid-structure coupling on a conforming mesh, and the solvers of the
fully coupled (unsymmetric) system.

.. automodule:: pycafe.build_matrices.coupling
   :members:

.. automodule:: pycafe.solver.solver_vibroacoustic
   :members:

---

Mesh generation and loading
---------------------------

Scriptable mesh generators and Gmsh mesh loading, including physical
groups with element connectivity.

.. automodule:: pycafe.create_geom.create_geom
   :members:

.. automodule:: pycafe.create_geom.visualize_mesh
   :members:

Physical group conventions
~~~~~~~~~~~~~~~~~~~~~~~~~~

External ``.msh`` files (Gmsh MSH 4.1) are supported directly. pyCAFE
identifies mesh regions by **physical group name**:

- ``fluid`` (dim 2 or 3): the acoustic domain;
- ``plate`` / ``structure`` (dim 2): the structural shell domain,
  which for conforming meshes is also the coupling interface;
- ``rigid_walls`` (dim 1 or 2): acoustically rigid boundaries
  (natural condition);
- free names (e.g. ``inlet``, ``left``, ``top``) for boundaries used
  by :func:`pycafe.assign_boundary_conditions`.

Elements are recognized by their Gmsh element name (e.g.
``"Quadrilateral 8"``, ``"Hexahedron 8"``) through the element
registry, never by node count alone. Note that Gmsh only saves
elements belonging to a physical group (``Mesh.SaveAll=0`` by
default): every entity needed by the analysis must be in a group.

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

.. automodule:: pycafe.boundary_condition.acoustic_bc
   :members:

---

Post-processing
---------------

Pressure fields, frequency responses and movies. The 2D entry point
forwards to the 3D one when the mesh is a volume.

.. automodule:: pycafe.post_processing.post_processing
   :members:

.. automodule:: pycafe.post_processing.post_processing_3d
   :members:

.. automodule:: pycafe.post_processing.post_processing_modal
   :members:
