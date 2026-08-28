# Work in progress

What is written and works, but is **not part of the first release**.

The release reads one kind of input: a mesh gmsh has already cut, with its
physical groups already named. Everything here starts one step earlier, from
a geometry that still has to be meshed or from a deck written by another
solver, and that stage is not settled yet: which entity is fluid, what unit
the file is drawn in, which property id is which role.

| notebook | what it does | what it needs |
|---|---|---|
| `duct_from_step_radiation.ipynb` | the whole way from a CAD file: `Library/WIP/Tubo_1m_1m.stp` to a duct with a flush panel, driven by a piston and truncated with the plane-wave radiation condition | `CadFile`, meshed by gmsh here |
| `shell_in_cavity_nonconforming.ipynb` | a plane wave at 60 kHz scattered by a cylinder in water, `Library/WIP/Vibro_ac1.step`; fluid and shell meshed separately and coupled by interpolation | `CadFile`, non-conforming coupling |

The inputs they read are in `Library/WIP/`, next to the Nastran deck
`demo_cavity.bdf` for the same reason.

Run them from this folder; they add `../..` to the path, so they import the
same `pycafe` as the released examples.
