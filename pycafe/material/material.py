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
