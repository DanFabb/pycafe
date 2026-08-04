from pycafe.material.material import define_fluid_material


def load_fluid(name="air"):
    """
    Load fluid properties.

    Parameters
    ----------
    name : str
        Fluid name (e.g. 'air', 'water')

    Returns
    -------
    fluid : dict
        Dictionary with rho, c0, Z0
    """
    rho, c0, Z0 = define_fluid_material(name)

    return {
        "name": name,
        "rho": rho,
        "c0": c0,
        "Z0": Z0,
    }
