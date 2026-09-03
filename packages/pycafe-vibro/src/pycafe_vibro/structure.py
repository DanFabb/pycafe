"""
Shell material of a vibroacoustic model.

The fluid of a model is acoustic and lives in
:mod:`pycafe.core.model_spec`; the shell that bounds it is structural
and lives here, so that an acoustic-only install carries neither.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Structure:
    """
    Shell material and thickness, plus how its support behaves.

    Parameters
    ----------
    t : float
        Thickness [m].
    E : float
        Young's modulus [Pa].
    nu : float
        Poisson ratio.
    rho_s : float
        Density [kg/m^3].
    nsm : float, optional
        Non-structural mass per unit area [kg/m^2].
    support : {"clamped", "simply_supported", "free"}, optional
        What is blocked on the nodes of the clamp group. The mesh only
        says *which* nodes are held; this says how.
    name : str, optional
        Label, for reporting only.
    """

    t: float
    E: float = 70e9
    nu: float = 0.33
    rho_s: float = 2700.0
    nsm: float = 0.0
    support: str = "clamped"
    name: str = "custom"

    @property
    def D(self):
        """Bending stiffness ``E t^3 / (12 (1 - nu^2))`` [N m]."""
        return self.E * self.t ** 3 / (12.0 * (1.0 - self.nu ** 2))

    def bending_wavelength(self, f):
        """
        Free bending wavelength of the plate at ``f`` [m].

        ``lambda_b = 2 pi / (rho_s t omega^2 / D)^(1/4)``. Not used to
        size the mesh — the sizing rule is acoustic — but reported, so
        that a model run above coincidence shows it.
        """
        omega = 2.0 * math.pi * float(f)
        k_b = (self.rho_s * self.t * omega ** 2 / self.D) ** 0.25
        return 2.0 * math.pi / k_b


def aluminium(t, **kwargs):
    """Aluminium shell of thickness ``t`` [m]: E = 70 GPa, nu = 0.33, 2700 kg/m^3."""
    return Structure(t=t, E=70e9, nu=0.33, rho_s=2700.0, name="aluminium",
                     **kwargs)


def steel(t, **kwargs):
    """Steel shell of thickness ``t`` [m]: E = 210 GPa, nu = 0.30, 7850 kg/m^3."""
    return Structure(t=t, E=210e9, nu=0.30, rho_s=7850.0, name="steel",
                     **kwargs)
