# %%

import gmsh
import pathlib


#  Mesh sizing
def compute_mesh_size_from_frequency(fmax, c0=343.0, n_per_wavelength=6):
    """
    Compute mesh size using acoustic wavelength criterion.
    """
    lambda_min = c0 / fmax
    h = lambda_min / n_per_wavelength

    print(f"\n--- Mesh size estimation ---")
    print(f"f_max        = {fmax:.1f} Hz")
    print(f"λ_min        = {lambda_min:.4f} m")
    print(f"mesh_size h  = {h:.4f} m  (~ {n_per_wavelength} elems / λ)")

    return h


def _set_element_order(element_type):
    if element_type == "CQUAD4":
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
    elif element_type == "CQUAD8":
        gmsh.option.setNumber("Mesh.ElementOrder", 2)
        # Serendipity (8-node) quads, not 9-node Lagrange.
        gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 1)
    else:
        raise ValueError(f"Unsupported element type: {element_type}")


#  Scriptable mesh generators (no input(), no tkinter)
def generate_rectangle_mesh_physical(
    length,
    height,
    mesh_size,
    element_type="CQUAD4",
    output_path=None,
    show_gui=False,
):
    """
    Generate a rectangular 2D acoustic mesh and write it to ``output_path``.

    Fully scriptable: all parameters are explicit, no interactive
    prompts. Physical groups: ``bottom``, ``right``, ``top``, ``left``
    (dim 1) and ``domain`` (dim 2).

    Parameters
    ----------
    length, height : float
        Rectangle dimensions [m].
    mesh_size : float
        Target element size [m] (see
        :func:`compute_mesh_size_from_frequency`).
    element_type : str
        ``"CQUAD4"`` (linear) or ``"CQUAD8"`` (quadratic).
    output_path : str or pathlib.Path
        Destination ``.msh`` file. Required.
    show_gui : bool
        If True, open the Gmsh GUI before finalizing.

    Returns
    -------
    pathlib.Path
        Path of the written mesh file.
    """
    if output_path is None:
        raise ValueError("output_path is required (path to the .msh file).")
    output_path = pathlib.Path(output_path)

    gmsh.initialize()
    try:
        gmsh.model.add("rectangle")

        p1 = gmsh.model.geo.addPoint(0, 0, 0, mesh_size)
        p2 = gmsh.model.geo.addPoint(length, 0, 0, mesh_size)
        p3 = gmsh.model.geo.addPoint(length, height, 0, mesh_size)
        p4 = gmsh.model.geo.addPoint(0, height, 0, mesh_size)

        l1 = gmsh.model.geo.addLine(p1, p2)
        l2 = gmsh.model.geo.addLine(p2, p3)
        l3 = gmsh.model.geo.addLine(p3, p4)
        l4 = gmsh.model.geo.addLine(p4, p1)

        cl = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
        surf = gmsh.model.geo.addPlaneSurface([cl])

        gmsh.model.addPhysicalGroup(1, [l1], name="bottom")
        gmsh.model.addPhysicalGroup(1, [l2], name="right")
        gmsh.model.addPhysicalGroup(1, [l3], name="top")
        gmsh.model.addPhysicalGroup(1, [l4], name="left")
        gmsh.model.addPhysicalGroup(2, [surf], name="domain")

        gmsh.model.geo.synchronize()

        _set_element_order(element_type)
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.model.mesh.generate(2)

        gmsh.write(str(output_path))
        print(f"Mesh written to {output_path}")

        if show_gui:
            gmsh.fltk.run()
    finally:
        gmsh.finalize()

    return output_path


def generate_circle_mesh_physical(
    radius,
    mesh_size,
    element_type="CQUAD4",
    output_path=None,
    show_gui=False,
):
    """
    Generate a circular 2D acoustic mesh and write it to ``output_path``.

    Fully scriptable: all parameters are explicit, no interactive
    prompts. Physical groups: ``right``, ``top``, ``left``, ``bottom``
    (quarter arcs, dim 1) and ``domain`` (dim 2).

    Parameters
    ----------
    radius : float
        Circle radius [m].
    mesh_size : float
        Target element size [m].
    element_type : str
        ``"CQUAD4"`` (linear) or ``"CQUAD8"`` (quadratic).
    output_path : str or pathlib.Path
        Destination ``.msh`` file. Required.
    show_gui : bool
        If True, open the Gmsh GUI before finalizing.

    Returns
    -------
    pathlib.Path
        Path of the written mesh file.
    """
    if output_path is None:
        raise ValueError("output_path is required (path to the .msh file).")
    output_path = pathlib.Path(output_path)

    gmsh.initialize()
    try:
        gmsh.model.add("circle")

        center = gmsh.model.geo.addPoint(0.0, 0.0, 0.0, mesh_size)

        p1 = gmsh.model.geo.addPoint(radius, 0.0, 0.0, mesh_size)
        p2 = gmsh.model.geo.addPoint(0.0, radius, 0.0, mesh_size)
        p3 = gmsh.model.geo.addPoint(-radius, 0.0, 0.0, mesh_size)
        p4 = gmsh.model.geo.addPoint(0.0, -radius, 0.0, mesh_size)

        c1 = gmsh.model.geo.addCircleArc(p1, center, p2)
        c2 = gmsh.model.geo.addCircleArc(p2, center, p3)
        c3 = gmsh.model.geo.addCircleArc(p3, center, p4)
        c4 = gmsh.model.geo.addCircleArc(p4, center, p1)

        cl = gmsh.model.geo.addCurveLoop([c1, c2, c3, c4])
        surf = gmsh.model.geo.addPlaneSurface([cl])

        gmsh.model.addPhysicalGroup(1, [c1], name="right")   # x > 0, y > 0
        gmsh.model.addPhysicalGroup(1, [c2], name="top")     # x < 0, y > 0
        gmsh.model.addPhysicalGroup(1, [c3], name="left")    # x < 0, y < 0
        gmsh.model.addPhysicalGroup(1, [c4], name="bottom")  # x > 0, y < 0
        gmsh.model.addPhysicalGroup(2, [surf], name="domain")

        gmsh.model.geo.synchronize()

        _set_element_order(element_type)
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.model.mesh.generate(2)

        gmsh.write(str(output_path))
        print(f"Circle mesh written to {output_path}")

        if show_gui:
            gmsh.fltk.run()
    finally:
        gmsh.finalize()

    return output_path


#  Interactive wrapper (terminal prompts + save dialog)
def ask_geometry_type():
    while True:
        choice = input(
            "Geometry type [RECTANGLE / CIRCLE ]: "
        ).strip().upper()

        if choice in ("RECTANGLE", "CIRCLE"):
            print(f" Selected geometry: {choice}")
            return choice

        print("No : invalid choice.")


def ask_save_msh_file(default_name="mesh.msh"):
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()

    root.withdraw()
    root.update()

    filename = filedialog.asksaveasfilename(
        title="Save Gmsh mesh",
        defaultextension=".msh",
        initialfile=default_name,
        filetypes=[("Gmsh mesh", "*.msh")]
    )

    root.destroy()

    if not filename:
        print(" User cancelled save dialog.")
        return None

    return pathlib.Path(filename)


def run_geometry_and_mesh_generator(c0):
    """
    Interactive front-end: asks geometry, frequency and element type,
    then delegates to the scriptable generators above.
    """
    geom_type = ask_geometry_type()

    fmax = float(input("Maximum frequency of interest [Hz]: "))
    mesh_size = compute_mesh_size_from_frequency(fmax, c0=c0)

    element_type = input("Element type [CQUAD4 / CQUAD8]: ").strip().upper()

    if geom_type == "RECTANGLE":
        length = float(input("Rectangle length [m]: "))
        height = float(input("Rectangle height [m]: "))

        output_path = ask_save_msh_file(f"rectangle_{element_type}.msh")
        if output_path is None:
            return None

        return generate_rectangle_mesh_physical(
            length, height, mesh_size,
            element_type=element_type,
            output_path=output_path,
        )

    elif geom_type == "CIRCLE":
        radius = float(input("Circle radius [m]: "))

        output_path = ask_save_msh_file(f"circle_{element_type}.msh")
        if output_path is None:
            return None

        return generate_circle_mesh_physical(
            radius, mesh_size,
            element_type=element_type,
            output_path=output_path,
        )


def define_fluid_material(
    name="air",
    rho0=None,
    c0=None,
):
    """
    Define fluid material properties.

    Parameters
    ----------
    name : str
        Name of the fluid ('air', 'water', 'custom')
    rho0 : float, optional
        Fluid density [kg/m^3] (required if name='custom')
    c0 : float, optional
        Speed of sound [m/s] (required if name='custom')

    Returns
    -------
    rho0 : float
        Fluid density [kg/m^3]
    c0 : float
        Speed of sound [m/s]
    Z0 : float
        Characteristic acoustic impedance [Pa·s/m]
    """

    name = name.lower()

    if name == "air":
        rho0 = 1.204      # kg/m^3 (20°C)
        c0   = 343.0      # m/s

    elif name == "water":
        rho0 = 998.0      # kg/m^3
        c0   = 1480.0     # m/s

    elif name == "custom":
        if rho0 is None or c0 is None:
            raise ValueError(
                "For custom material you must provide rho0 and c0."
            )
        rho0 = float(rho0)
        c0   = float(c0)

    else:
        raise ValueError(
            f"Unknown fluid '{name}'. "
            "Available: 'air', 'water', 'custom'."
        )

    Z0 = rho0 * c0
    return rho0, c0, Z0
