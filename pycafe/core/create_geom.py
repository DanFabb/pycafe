def create_geometry(*, c0):
    """
    Generate the computational geometry and mesh using Gmsh.

    This function creates the acoustic domain geometry and
    generates the corresponding mesh. The speed of sound is
    used to scale the geometry when required.

    Parameters
    ----------
    c0 : float
        Speed of sound in the fluid.

    Returns
    -------
    msh_file : str
        Path to the generated Gmsh ``.msh`` file.

    Notes
    -----
    The actual geometry definition and mesh generation are
    handled by :func:`run_geometry_and_mesh_generator`.
    """
    from pycafe.create_geom.create_geom import run_geometry_and_mesh_generator

    msh_file = run_geometry_and_mesh_generator(c0)
    return msh_file
