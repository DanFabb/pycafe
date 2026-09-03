# element_tetra4.py
#
# 4-node linear tetrahedral acoustic element (CTETRA4, 1 pressure DOF
# per node). Shape functions are the tetrahedral coordinates themselves,
# N_i = zeta_i, following Felippa, "Advanced Finite Element Methods",
# Chapter 15 ("The Linear Tetrahedron"); the equation numbers quoted
# below are his.
#
# Node ordering follows the Gmsh "Tetrahedron 4" convention, which is
# the numbering rule of §15.2.1: nodes 1-2-3 counterclockwise seen from
# node 4, so that the signed volume (15.2) comes out positive.
#
# Both element matrices are exact in closed form — no quadrature. The
# shape function gradients are constant over the element (15.10), so the
# stiffness is a single product; the mass matrix follows from the
# integration rule (15.35). This is the same acoustic convention as the
# other elements: no 1/rho factor in K, and M = integral(N N^T)/c^2.
import numpy as np


#  GEOMETRY
def tetra4_geometry(x_e):
    """
    Volume and shape function gradients of a linear tetrahedron.

    Implements (15.7)-(15.10) of Felippa Ch. 15: the coefficients
    ``a_i``, ``b_i``, ``c_i`` are the adjoints of the coordinate
    transformation (15.5), and

    .. math:: 6 \\mathcal{V} \\, \\partial \\zeta_i / \\partial x = a_i ,

    and likewise ``b_i`` for ``y`` and ``c_i`` for ``z``. Because the
    shape functions are the tetrahedral coordinates, these are also the
    gradients of ``N_i``, and they are constant over the element.

    Parameters
    ----------
    x_e : ndarray of shape (4, 3)
        Physical coordinates of the four corners, in Gmsh
        "Tetrahedron 4" ordering.

    Returns
    -------
    volume : float
        Element volume, positive.
    grad_N : ndarray of shape (3, 4)
        Rows ``dN/dx``, ``dN/dy``, ``dN/dz``; columns are the nodes.

    Raises
    ------
    ValueError
        If the four corners are coplanar or numbered so that the signed
        volume is not positive.
    """
    x_e = np.asarray(x_e, dtype=float)
    if x_e.shape != (4, 3):
        raise ValueError(f"x_e must have shape (4, 3), got {x_e.shape}")

    x, y, z = x_e[:, 0], x_e[:, 1], x_e[:, 2]

    # Differences x_ij = x_i - x_j, in the 1-based notation of (15.7);
    # the index shift to 0-based happens here once.
    def d(v, i, j):
        return v[i - 1] - v[j - 1]

    # 6V, equation (15.8).
    six_v = (d(x, 2, 1) * (d(y, 3, 1) * d(z, 4, 1) - d(y, 4, 1) * d(z, 3, 1))
             + d(y, 2, 1) * (d(x, 4, 1) * d(z, 3, 1) - d(x, 3, 1) * d(z, 4, 1))
             + d(z, 2, 1) * (d(x, 3, 1) * d(y, 4, 1) - d(x, 4, 1) * d(y, 3, 1)))

    if six_v <= 0.0:
        raise ValueError(
            f"Non-positive tetrahedron volume (6V = {six_v:.3e}): the "
            "corners are coplanar, or the node ordering does not follow "
            "the Gmsh 'Tetrahedron 4' convention (nodes 1-2-3 "
            "counterclockwise seen from node 4)."
        )

    # Equation (15.7).
    a = np.array([
        y[1] * d(z, 4, 3) - y[2] * d(z, 4, 2) + y[3] * d(z, 3, 2),
        -y[0] * d(z, 4, 3) + y[2] * d(z, 4, 1) - y[3] * d(z, 3, 1),
        y[0] * d(z, 4, 2) - y[1] * d(z, 4, 1) + y[3] * d(z, 2, 1),
        -y[0] * d(z, 3, 2) + y[1] * d(z, 3, 1) - y[2] * d(z, 2, 1),
    ])
    b = np.array([
        -x[1] * d(z, 4, 3) + x[2] * d(z, 4, 2) - x[3] * d(z, 3, 2),
        x[0] * d(z, 4, 3) - x[2] * d(z, 4, 1) + x[3] * d(z, 3, 1),
        -x[0] * d(z, 4, 2) + x[1] * d(z, 4, 1) - x[3] * d(z, 2, 1),
        x[0] * d(z, 3, 2) - x[1] * d(z, 3, 1) + x[2] * d(z, 2, 1),
    ])
    c = np.array([
        x[1] * d(y, 4, 3) - x[2] * d(y, 4, 2) + x[3] * d(y, 3, 2),
        -x[0] * d(y, 4, 3) + x[2] * d(y, 4, 1) - x[3] * d(y, 3, 1),
        x[0] * d(y, 4, 2) - x[1] * d(y, 4, 1) + x[3] * d(y, 2, 1),
        -x[0] * d(y, 3, 2) + x[1] * d(y, 3, 1) - x[2] * d(y, 2, 1),
    ])

    grad_N = np.vstack([a, b, c]) / six_v
    return six_v / 6.0, grad_N


#  SHAPE FUNCTIONS
def tetra4_shape(zeta):
    """
    Shape functions of the linear tetrahedron at a point.

    They are the tetrahedral coordinates themselves, ``N_i = zeta_i``
    (§15.3), so this is little more than a named identity — it exists so
    that code needing ``N`` at a point (a source term, a probe) does not
    have to know that.

    Parameters
    ----------
    zeta : array_like of shape (4,)
        Tetrahedral coordinates, summing to one (15.3).

    Returns
    -------
    N : ndarray of shape (4,)

    Raises
    ------
    ValueError
        If the coordinates do not sum to one.
    """
    zeta = np.asarray(zeta, dtype=float)
    if zeta.shape != (4,):
        raise ValueError(f"zeta must have shape (4,), got {zeta.shape}")
    if not np.isclose(zeta.sum(), 1.0):
        raise ValueError(
            f"Tetrahedral coordinates must sum to 1 (15.3), got "
            f"{zeta.sum():.6g}."
        )
    return zeta.copy()


#  ELEMENT MATRICES (K_e, M_e)
# Consistent mass, from (15.35): the integral of zeta_i zeta_j over the
# element is V/10 on the diagonal and V/20 off it, i.e. V/20 times this
# matrix.
_MASS_PATTERN = np.ones((4, 4)) + np.eye(4)


def element_matrices_tetra4(x_e, c):
    """
    Element stiffness and mass matrices of a CTETRA4 acoustic element.

    .. math::
        \\mathbf K_e = \\int_{V_e} \\nabla N \\, \\nabla N^T \\, dV
        = \\mathcal{V} \\, \\mathbf G^T \\mathbf G ,
        \\qquad
        \\mathbf M_e = \\frac{1}{c^2} \\int_{V_e} N N^T dV ,

    both exact: the gradients ``G`` are constant over the element
    (15.10) and the mass integral is (15.35). No quadrature is
    involved, so unlike :func:`element_matrices_hex8` this kernel takes
    no ``quad_rule``.

    Parameters
    ----------
    x_e : ndarray of shape (4, 3)
        Physical coordinates of the element nodes, in Gmsh
        "Tetrahedron 4" ordering.
    c : float
        Speed of sound in the fluid.

    Returns
    -------
    K_e : ndarray of shape (4, 4)
        Element acoustic stiffness matrix.
    M_e : ndarray of shape (4, 4)
        Element acoustic mass matrix.

    Notes
    -----
    The shape function gradients are constant over the element (15.10),
    so the pressure gradient — and with it the particle velocity and the
    intensity — is constant inside each tetrahedron. Felippa notes the
    consequence for stress analysis in §15.2.

    See Also
    --------
    tetra4_geometry : Volume and shape function gradients.
    element_matrices_hex8 : The trilinear brick.
    """
    volume, grad_N = tetra4_geometry(x_e)

    K_e = volume * (grad_N.T @ grad_N)
    M_e = (volume / 20.0) * _MASS_PATTERN / c**2

    return K_e, M_e
