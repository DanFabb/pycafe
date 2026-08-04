# element_hex8.py
#
# 8-node trilinear hexahedral acoustic element (CHEXA8, 1 pressure
# DOF per node). Shape functions: standard linear brick
# N_i = 1/8 (1 + xi*xi_i)(1 + eta*eta_i)(1 + zeta*zeta_i)
# (Altair Radioss theory manual, "Linear 8-node brick element").
# Node-to-corner assignment follows the GMSH "Hexahedron 8"
# convention, since element connectivity comes from the Gmsh reader.
import numpy as np

# Coordinate naturali dei vertici nell'ordinamento Gmsh "Hexahedron 8"
_XI_I = np.array([-1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0])
_ETA_I = np.array([-1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0])
_ZETA_I = np.array([-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0])


# ------------------------------------------------------------
#  GAUSS QUADRATURE
# ------------------------------------------------------------
def gauss_rule_hex_2x2x2():
    """
    2x2x2 Gauss quadrature rule for hexahedral elements.

    Returns
    -------
    xi_pts, eta_pts, zeta_pts : ndarray of shape (8,)
        Integration point coordinates (+-1/sqrt(3)).
    w_pts : ndarray of shape (8,)
        Integration weights (all 1).
    """
    g = 1.0 / np.sqrt(3.0)
    pts = np.array([
        [-g, -g, -g],
        [ g, -g, -g],
        [ g,  g, -g],
        [-g,  g, -g],
        [-g, -g,  g],
        [ g, -g,  g],
        [ g,  g,  g],
        [-g,  g,  g],
    ])
    w = np.ones(8)
    return pts[:, 0], pts[:, 1], pts[:, 2], w


# ------------------------------------------------------------
#  SHAPE FUNCTIONS
# ------------------------------------------------------------
def hex8_shape(xi, eta, zeta):
    """
    Trilinear shape functions and derivatives for the HEXA8 element.

    N_i = 1/8 (1 + xi*xi_i)(1 + eta*eta_i)(1 + zeta*zeta_i),
    with node corners ordered per the Gmsh "Hexahedron 8" convention.

    Returns
    -------
    N, dN_dxi, dN_deta, dN_dzeta : ndarray of shape (8,)
    """
    fx = 1.0 + xi * _XI_I
    fy = 1.0 + eta * _ETA_I
    fz = 1.0 + zeta * _ZETA_I

    N = 0.125 * fx * fy * fz
    dN_dxi = 0.125 * _XI_I * fy * fz
    dN_deta = 0.125 * fx * _ETA_I * fz
    dN_dzeta = 0.125 * fx * fy * _ZETA_I

    return N, dN_dxi, dN_deta, dN_dzeta


# ------------------------------------------------------------
#  JACOBIAN
# ------------------------------------------------------------
def jacobian_3d(dN_dxi, dN_deta, dN_dzeta, x3d):
    """
    Jacobian of the isoparametric mapping for a 3D element.

    Parameters
    ----------
    dN_dxi, dN_deta, dN_dzeta : ndarray of shape (8,)
        Shape function derivatives in natural coordinates.
    x3d : ndarray of shape (8, 3)
        Physical coordinates of the element nodes.

    Returns
    -------
    J : ndarray (3, 3)
    detJ : float
    invJ : ndarray (3, 3)
    """
    grad = np.vstack([dN_dxi, dN_deta, dN_dzeta])   # (3, 8)
    J = grad @ x3d                                   # (3, 3)
    detJ = np.linalg.det(J)
    if detJ <= 0.0:
        raise ValueError(
            f"Non-positive Jacobian determinant (detJ = {detJ:.3e}): "
            "element is inverted or node ordering does not follow the "
            "Gmsh 'Hexahedron 8' convention."
        )
    invJ = np.linalg.inv(J)
    return J, detJ, invJ


# ------------------------------------------------------------
#  B OPERATORS
# ------------------------------------------------------------
def B_stiffness_3d(dN_dxi, dN_deta, dN_dzeta, invJ):
    """
    Gradient operator in physical coordinates: (3, 8) matrix of
    [dN/dx; dN/dy; dN/dz].
    """
    grad_nat = np.vstack([dN_dxi, dN_deta, dN_dzeta])   # (3, 8)
    return invJ @ grad_nat


def B_mass(N):
    """Shape function row vector (1, 8) for the mass matrix."""
    return N.reshape(1, -1)


# ------------------------------------------------------------
#  ELEMENT MATRICES (K_e, M_e)
# ------------------------------------------------------------
def element_matrices_hex8(x_e, c, quad_rule=gauss_rule_hex_2x2x2):
    """
    Compute the element stiffness and mass matrices for a CHEXA8 element.

    Same acoustic convention as the 2D elements: no 1/rho factor in K
    (density is folded into the impedance/velocity boundary terms) and
    M = integral(N N^T)/c^2.

    Parameters
    ----------
    x_e : ndarray of shape (8, 3)
        Physical coordinates of the element nodes, in Gmsh
        "Hexahedron 8" ordering.
    c : float
        Speed of sound in the fluid.
    quad_rule : callable, optional
        Function returning ``(xi_pts, eta_pts, zeta_pts, w_pts)``.
        Default is the 2x2x2 Gauss rule.

    Returns
    -------
    K_e : ndarray of shape (8, 8)
        Element acoustic stiffness matrix.
    M_e : ndarray of shape (8, 8)
        Element acoustic mass matrix.

    See Also
    --------
    hex8_shape : Shape functions and derivatives.
    jacobian_3d : Jacobian of the isoparametric mapping.
    gauss_rule_hex_2x2x2 : Default Gauss quadrature rule.
    """
    x_e = np.asarray(x_e, dtype=float)
    if x_e.shape != (8, 3):
        raise ValueError(f"x_e must have shape (8, 3), got {x_e.shape}")

    xi_pts, eta_pts, zeta_pts, w_pts = quad_rule()

    K_e = np.zeros((8, 8), dtype=float)
    M_e = np.zeros((8, 8), dtype=float)

    for xi, eta, zeta, w in zip(xi_pts, eta_pts, zeta_pts, w_pts):
        N, dN_dxi, dN_deta, dN_dzeta = hex8_shape(xi, eta, zeta)
        J, detJ, invJ = jacobian_3d(dN_dxi, dN_deta, dN_dzeta, x_e)
        B = B_stiffness_3d(dN_dxi, dN_deta, dN_dzeta, invJ)
        Bm = B_mass(N)

        K_e += (B.T @ B) * detJ * w
        M_e += (1.0 / c**2) * (Bm.T @ Bm) * detJ * w

    return K_e, M_e
