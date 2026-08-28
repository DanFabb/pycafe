r"""
The closed forms a finite element model can be checked against.

Two of them, and both are quoted where a validation notebook needs a
number that owes nothing to the mesh:

* :func:`rectangular_cavity_modes` — the modes of a hard-walled box,
  exact, with :func:`cavity_mode_shape` for the shape that goes with
  each one and :func:`identify_cavity_modes` to give a computed mode
  the ``(l, m, n)`` it correlates with;
* :func:`clamped_plate_modes` — Warburton's (1954) beam-function
  frequencies of a plate clamped on all four edges, an approximate
  Rayleigh solution good to a few percent.

Neither is a substitute for the element validation itself (see
``examples/validation/``); they are what an example run prints its own
frequencies next to.
"""

import numpy as np

__all__ = ["rectangular_cavity_modes", "cavity_mode_shape",
           "identify_cavity_modes", "mac",
           "warburton_cccc", "clamped_plate_modes"]


def rectangular_cavity_modes(c0, Lx, Ly, Lz, *, num=5, skip_zero=True,
                             order=3):
    r"""
    Modes of a hard-walled rectangular cavity.

    .. math::

        f_{lmn} = \frac{c_0}{2}\sqrt{(l/L_x)^2 + (m/L_y)^2 + (n/L_z)^2}

    Parameters
    ----------
    c0 : float
        Speed of sound [m/s].
    Lx, Ly, Lz : float
        Sides of the box [m].
    num : int, optional
        How many modes to return.
    skip_zero : bool, optional
        Leave out ``(0, 0, 0)``, the uniform pressure at 0 Hz. Default
        True.
    order : int, optional
        Highest index tried along each side. The default covers the
        first few dozen modes of a box of ordinary proportions.

    Returns
    -------
    list of (float, tuple)
        ``(frequency [Hz], (l, m, n))``, sorted by frequency.

    Examples
    --------
    >>> modes = rectangular_cavity_modes(343.0, 0.8, 0.6, 0.5, num=2)
    >>> [(round(float(f), 1), lmn) for f, lmn in modes]
    [(214.4, (1, 0, 0)), (285.8, (0, 1, 0))]
    """
    modes = sorted(
        (c0 / 2 * np.sqrt((l / Lx) ** 2 + (m / Ly) ** 2 + (n / Lz) ** 2),
         (l, m, n))
        for l in range(order) for m in range(order) for n in range(order)
    )
    if skip_zero:
        modes = [entry for entry in modes if entry[1] != (0, 0, 0)]
    return modes[:num]


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


def cavity_mode_shape(coordinates, sides, index, origin=None):
    r"""
    Nodal values of one hard-walled cavity mode shape.

    .. math::

        p_{lmn}(x, y, z) = \cos\frac{l\pi x}{L_x}\,
                           \cos\frac{m\pi y}{L_y}\,
                           \cos\frac{n\pi z}{L_z}

    written about the corner of the box, so the coordinates are measured
    from ``origin`` rather than from wherever the mesh happens to sit.

    Parameters
    ----------
    coordinates : ndarray (N, 3)
        Node coordinates [m].
    sides : sequence of float
        ``(Lx, Ly, Lz)`` [m].
    index : tuple of int
        ``(l, m, n)``.
    origin : sequence of float, optional
        The corner the formula is written about. Default: the smallest
        coordinate along each axis.

    Returns
    -------
    ndarray (N,)
    """
    coordinates = np.asarray(coordinates, dtype=float)
    origin = (coordinates.min(axis=0) if origin is None
              else np.asarray(origin, dtype=float))
    local = coordinates - origin
    shape = np.ones(local.shape[0])
    for axis, (k, length) in enumerate(zip(index, sides)):
        shape = shape * np.cos(k * np.pi * local[:, axis] / length)
    return shape


def mac(a, b):
    r"""
    Modal assurance criterion between two shapes.

    .. math::

        \mathrm{MAC}(\mathbf a, \mathbf b) =
        \frac{(\mathbf a^{T}\mathbf b)^{2}}
             {(\mathbf a^{T}\mathbf a)(\mathbf b^{T}\mathbf b)}

    One for two shapes that differ by a scale factor, zero for two that
    are orthogonal.

    Parameters
    ----------
    a, b : ndarray (N,)

    Returns
    -------
    float
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    return float((a @ b) ** 2 / ((a @ a) * (b @ b)))


def identify_cavity_modes(coordinates, modes, frequencies, c0, sides, *,
                          origin=None, order=5):
    """
    Give every computed cavity mode the ``(l, m, n)`` it correlates with.

    The pairing is by **shape**, not by order: two analytical
    frequencies can sit a few hertz apart, and reading the labels off
    the sorted list would swap them and blame the solver for it. Each
    computed mode is compared with every candidate shape and keeps the
    best correlated one; that triple then gives the frequency it should
    have had.

    Parameters
    ----------
    coordinates : ndarray (N, 3)
        Node coordinates of the cavity mesh [m].
    modes : ndarray (N, k)
        Mode shapes, one nodal pressure per column, as
        :func:`~pycafe.solver.solver_modale.solve_modal_acoustic_reduced`
        returns them.
    frequencies : array-like (k,)
        Their frequencies [Hz].
    c0 : float
        Speed of sound [m/s].
    sides : sequence of float
        ``(Lx, Ly, Lz)`` of the box [m].
    origin : sequence of float, optional
        The corner the analytical shapes are written about. Default: the
        corner the mesh turned out to have.
    order : int, optional
        Highest index tried along each side.

    Returns
    -------
    dict
        ``labels`` (list of ``(l, m, n)``), ``exact`` [Hz],
        ``computed`` [Hz], ``error`` [%], ``mac`` (k x k, every computed
        mode against every label it was given) and ``shapes``, the
        paired analytical shapes on the nodes.

    Notes
    -----
    A diagonal of ones in ``mac`` is what the pairing was chosen to
    produce; what the matrix adds is everything off it. Near-zero
    off-diagonal entries say each computed mode resembles its own label
    and no other, so the labels are the only ones they could have had.
    """
    coordinates = np.asarray(coordinates, dtype=float)
    modes = np.asarray(modes, dtype=float)
    frequencies = np.asarray(frequencies, dtype=float)
    sides = np.asarray(sides, dtype=float)

    candidates = [(l, m, n)
                  for l in range(order) for m in range(order)
                  for n in range(order) if (l, m, n) != (0, 0, 0)]
    library = np.array([cavity_mode_shape(coordinates, sides, index, origin)
                        for index in candidates])

    labels = [candidates[int(np.argmax([mac(shape, modes[:, i])
                                        for shape in library]))]
              for i in range(modes.shape[1])]
    shapes = np.array([cavity_mode_shape(coordinates, sides, index, origin)
                       for index in labels])
    exact = np.array([c0 / 2 * np.sqrt(np.sum((np.array(index) / sides) ** 2))
                      for index in labels])

    matrix = np.array([[mac(shapes[j], modes[:, i])
                        for j in range(len(labels))]
                       for i in range(len(labels))])

    return {
        "labels": labels,
        "exact": exact,
        "computed": frequencies,
        "error": (frequencies - exact) / exact * 100.0,
        "mac": matrix,
        "shapes": shapes,
    }
