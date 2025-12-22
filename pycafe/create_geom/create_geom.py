# %%

import gmsh
import numpy as np
import pathlib
import tkinter as tk
from tkinter import filedialog

def ask_geometry_type():
    while True:
        choice = input(
            "Geometry type [RECTANGLE / CIRCLE ]: "
        ).strip().upper()

        if choice in ("RECTANGLE", "CIRCLE"):
            print(f" Selected geometry: {choice}")
            return choice

        print("No : invalid choice.")


# %%
def ask_free_shape_points():
    """
    Ask user for a closed polygon defined by N points.
    """
    n = int(input("Number of boundary curves / points: "))
    if n < 3:
        raise ValueError("At least 3 points are required.")

    points = []
    print("Insert points in order (counter-clockwise suggested):")

    for i in range(n):
        x = float(input(f"Point {i+1} - x: "))
        y = float(input(f"Point {i+1} - y: "))
        points.append((x, y))

    return points


# %%
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







def ask_save_msh_file(default_name="mesh.msh"):
    root = tk.Tk()

    # QUESTO È FONDAMENTALE
    root.withdraw()
    root.update()   # forza l'inizializzazione GUI

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



def generate_rectangle_mesh_physical(
    length,
    height,
    mesh_size,
    element_type="CQUAD4",
    show_gui=False,
):
    gmsh.initialize()
    gmsh.model.add("rectangle")

    # -------------------------
    # Geometry
    # -------------------------
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

    # -------------------------
    # Physical groups
    # -------------------------
    gmsh.model.addPhysicalGroup(1, [l1], name="bottom")
    gmsh.model.addPhysicalGroup(1, [l2], name="right")
    gmsh.model.addPhysicalGroup(1, [l3], name="top")
    gmsh.model.addPhysicalGroup(1, [l4], name="left")
    gmsh.model.addPhysicalGroup(2, [surf], name="domain")

    gmsh.model.geo.synchronize()

    # -------------------------
    # Element type
    # -------------------------
    if element_type == "CQUAD4":
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
    elif element_type == "CQUAD8":
        gmsh.option.setNumber("Mesh.ElementOrder", 2)
    else:
        raise ValueError("Unsupported element type")

    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.model.mesh.generate(2)

    # -------------------------
    # Export
    # -------------------------
    outdir = pathlib.Path().resolve()
    msh_file = ask_save_msh_file(f"rectangle_{element_type}.msh")

    if msh_file is None:
        gmsh.finalize()
        return None

    gmsh.write(str(msh_file))
    print(f"Mesh written to {msh_file}")



def generate_circle_mesh_physical(
    radius,
    mesh_size,
    element_type="CQUAD4",
    show_gui=False,
):
    gmsh.initialize()
    gmsh.model.add("circle")

    # -------------------------
    # Geometry
    # -------------------------
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

    # -------------------------
    # Physical groups (1D)
    # -------------------------
    gmsh.model.addPhysicalGroup(1, [c1], name="right")   # x > 0, y > 0
    gmsh.model.addPhysicalGroup(1, [c2], name="top")     # x < 0, y > 0
    gmsh.model.addPhysicalGroup(1, [c3], name="left")    # x < 0, y < 0
    gmsh.model.addPhysicalGroup(1, [c4], name="bottom")  # x > 0, y < 0

    # -------------------------
    # Physical group (2D)
    # -------------------------
    gmsh.model.addPhysicalGroup(2, [surf], name="domain")

    gmsh.model.geo.synchronize()

    # -------------------------
    # Element type
    # -------------------------
    if element_type == "CQUAD4":
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
    elif element_type == "CQUAD8":
        gmsh.option.setNumber("Mesh.ElementOrder", 2)
    else:
        raise ValueError("Unsupported element type")

    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.model.mesh.generate(2)

    # -------------------------
    # Save
    # -------------------------
    msh_file = ask_save_msh_file(f"circle_{element_type}.msh")
    if msh_file is None:
        gmsh.finalize()
        return None

    gmsh.write(str(msh_file))

    if show_gui:
        gmsh.fltk.run()

    gmsh.finalize()

    print(f"Circle mesh written to {msh_file}")
    return msh_file


# %%
def ask_free_shape_points():
    """
    Ask user for a closed polygon defined by N points.
    """
    n = int(input("Number of boundary curves / points: "))
    if n < 3:
        raise ValueError("At least 3 points are required.")

    points = []
    print("Insert points in order (counter-clockwise suggested):")

    for i in range(n):
        x = float(input(f"Point {i+1} - x: "))
        y = float(input(f"Point {i+1} - y: "))
        points.append((x, y))

    return points





# %%
def run_geometry_and_mesh_generator(c0):
    geom_type = ask_geometry_type()

    fmax = float(input("Maximum frequency of interest [Hz]: "))
    mesh_size = compute_mesh_size_from_frequency(fmax, c0=c0)

    element_type = input("Element type [CQUAD4 / CQUAD8]: ").strip().upper()

    if geom_type == "RECTANGLE":
        length = float(input("Rectangle length [m]: "))
        height = float(input("Rectangle height [m]: "))

        return generate_rectangle_mesh_physical(
            length, height, mesh_size,
            element_type=element_type
        )

    elif geom_type == "CIRCLE":
        radius = float(input("Circle radius [m]: "))

        return generate_circle_mesh_physical(
            radius, mesh_size,
            element_type=element_type
        )

    elif geom_type == "FREE_SHAPE":
        points = ask_free_shape_points()

        return generate_free_shape_mesh_physical(
            points, mesh_size,
            element_type=element_type
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