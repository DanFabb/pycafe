"""
Closed forms a coupled model is checked against.

The cavity's own closed forms are :mod:`pycafe.analytical`; the plate's
are here, because they are structural — they take a
:class:`~pycafe_vibro.structure.Structure` and its bending stiffness.
"""

import numpy as np


def warburton_cccc(m, n, a, b, D, mass, nu):
    r"""
    Warburton (1954) frequency of a clamped plate, mode ``(m, n)``.

    The beam-function (Rayleigh) approximation for a C-C-C-C rectangular
    plate: the mode shape is taken as the product of two clamped-beam
    functions, so the frequency comes out a few percent high on the
    first modes and closer above them.

    Parameters
    ----------
    m, n : int
        Half-wave numbers along ``a`` and ``b``, from 1.
    a, b : float
        Sides of the plate [m].
    D : float
        Bending stiffness [N m].
    mass : float
        Mass per unit area [kg/m2].
    nu : float
        Poisson ratio.

    Returns
    -------
    float
        Frequency [Hz].

    References
    ----------
    G. B. Warburton, "The vibration of rectangular plates",
    Proceedings of the Institution of Mechanical Engineers 168 (1954)
    371-384.
    """
    def coefficients(k):
        if k == 1:
            return 1.506, 1.248, 1.248
        G = k + 0.5
        H = J = G ** 2 * (1.0 - 2.0 / (G * np.pi))
        return G, H, J

    Gx, Hx, Jx = coefficients(m)
    Gy, Hy, Jy = coefficients(n)
    term = (Gx ** 4 / a ** 4 + Gy ** 4 / b ** 4
            + 2.0 / (a ** 2 * b ** 2) * (nu * Hx * Hy + (1.0 - nu) * Jx * Jy))
    return np.pi ** 2 * np.sqrt(D / mass) * np.sqrt(term) / (2.0 * np.pi)


def clamped_plate_modes(a, b, structure, *, num=6, order=4):
    """
    The first Warburton frequencies of a clamped plate, sorted.

    Parameters
    ----------
    a, b : float
        Sides of the plate [m].
    structure : Structure
        The material, from :mod:`pycafe.core.model_spec`: its ``D``,
        ``rho_s * t`` and ``nu`` are what the formula needs.
    num : int, optional
        How many to return.
    order : int, optional
        Highest half-wave number tried along each side.

    Returns
    -------
    list of (float, tuple)
        ``(frequency [Hz], (m, n))``, sorted by frequency.
    """
    mass = structure.rho_s * structure.t + structure.nsm
    return sorted(
        (warburton_cccc(m, n, a, b, structure.D, mass, structure.nu), (m, n))
        for m in range(1, order) for n in range(1, order)
    )[:num]
