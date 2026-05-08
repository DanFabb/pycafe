
import numpy as np


#  GAUSS RULES
def gauss_rule_quad_2x2():
    GAUSS_POINT = 1/np.sqrt(3.0)

    xi = GAUSS_POINT * np.array(
        [-1, 1, 1, -1],
        dtype=float
    )

    eta = GAUSS_POINT * np.array(
        [-1, -1, 1, 1],
        dtype=float
    )

    w = 1.0

    return xi, eta, w * np.ones_like(xi, dtype=float)

def gauss_rule_quad_3x3():
    GAUSS_POINT = np.sqrt(3.0 / 5.0)  # = sqrt(0.6) ≈ 0.7746

    xi = GAUSS_POINT * np.array(
        [-1, 0, 1,
         -1, 0, 1,
         -1, 0, 1],
        dtype=float
    )

    eta = GAUSS_POINT * np.array(
        [-1, -1, -1,
          0,  0,  0,
          1,  1,  1],
        dtype=float
    )

    w = np.array(
        [25/81, 40/81, 25/81,
         40/81, 64/81, 40/81,
         25/81, 40/81, 25/81],
        dtype=float
    )

    return xi, eta, w


#  SHAPE FUNCTIONS
def cquad8_shape(xi, eta):
   

    # Corner nodes
    N1 = 0.25 * ((1 - xi) * (1 - eta) * (-xi - eta - 1))
    N2 = 0.25 * ((1 + xi) * (1 - eta) * ( xi - eta - 1))
    N3 = 0.25 * ((1 + xi) * (1 + eta) * ( xi + eta - 1))
    N4 = 0.25 * ((1 - xi) * (1 + eta) * (-xi + eta - 1))

    # Mid-side nodes
    N5 = 0.5 * ((1 - xi**2) * (1 - eta))
    N6 = 0.5 * ((1 + xi)    * (1 - eta**2))
    N7 = 0.5 * ((1 - xi**2) * (1 + eta))
    N8 = 0.5 * ((1 - xi)    * (1 - eta**2))

    N = np.array([N1, N2, N3, N4, N5, N6, N7, N8], dtype=float)

    # Derivate rispetto a xi (tradotte 1:1 dal MATLAB)
    dN_dxi = np.array([
        -0.25*(1-eta)*(1-xi) - 0.25*(1-eta)*(-1-eta-xi),
         0.25*(1-eta)*(1+xi) + 0.25*(1-eta)*(-1-eta+xi),
         0.25*(1+eta)*(1+xi) + 0.25*(1+eta)*(-1+eta+xi),
        -0.25*(1+eta)*(1-xi) - 0.25*(1+eta)*(-1+eta-xi),
        -xi*(1-eta),
         0.5*(1-eta**2),
        -xi*(1+eta),
        -0.5*(1-eta**2)
    ], dtype=float)

    # Derivate rispetto a eta
    dN_deta = np.array([
        -0.25*(1-eta)*(1-xi) - 0.25*(1-xi)*(-1-eta-xi),
        -0.25*(1-eta)*(1+xi) - 0.25*(1+xi)*(-1-eta+xi),
         0.25*(1+eta)*(1+xi) + 0.25*(1+xi)*(-1+eta+xi),
         0.25*(1+eta)*(1-xi) + 0.25*(1-xi)*(-1+eta-xi),
        -0.5*(1 - xi**2),
        -(1 + xi) * eta,
         0.5*(1 - xi**2),
        -(1 - xi) * eta
    ], dtype=float)

    return N, dN_dxi, dN_deta


def linear_shape_1d(xi):
 
    N = 0.5 * np.array([1.0 - xi, 1.0 + xi], dtype=float)
    dN_dxi = 0.5 * np.array([-1.0, 1.0], dtype=float)
    return N, dN_dxi


def B_stiffness(dN_dxi, dN_deta, invJ):

    grad_nat = np.vstack([dN_dxi, dN_deta])   # (2,8)
    dN_dxy   = invJ @ grad_nat                # (2,8)
    dN_dx    = dN_dxy[0, :]
    dN_dy    = dN_dxy[1, :]
    B        = np.vstack([dN_dx, dN_dy])      # (2,8)
    return B


def B_mass(N):

    return N.reshape(1, -1)


def B_damping(N):
 
    return np.array([[N[0], N[1]]], dtype=float)


def B_imposed_velocity(N):
 
    return np.array([[N[0]]], dtype=float)


def B_normal_velocity(N):
 
    return np.array([
        [N[0], 0.0,   N[1], 0.0],
        [0.0,   N[0], 0.0,  N[1]]
    ], dtype=float)


#  JACOBIAN
def jacobian_2d(dN_dxi, dN_deta, x2d):
 
    grad = np.vstack([dN_dxi, dN_deta])      # (2,8)
    # in MATLAB: J = [dNdxi; dNdeta] * x2D_e.'
    # qui: (2,8) @ (8,2) = (2,2)
    J = grad @ x2d
    detJ = np.linalg.det(J)
    invJ = np.linalg.inv(J)
    return J, detJ, invJ


def jacobian_1d(dN_dxi, x1D_e):
    
    x1D_e = np.asarray(x1D_e, dtype=float).ravel()
    J = dN_dxi @ x1D_e    # prodotto riga x1D_e
    detJ = J
    invJ = 1.0 / J
    return J, detJ, invJ


def calcola_line(x_start, x_end):
 
    x_start = np.asarray(x_start, dtype=float).ravel()
    x_end   = np.asarray(x_end,   dtype=float).ravel()

    # punto medio
    x0 = 0.5 * (x_start + x_end)

    # direzione locale
    x_dir = x_end - x_start
    x_dir = x_dir / np.linalg.norm(x_dir)

    # matrice di trasformazione 1x3
    T_1D = x_dir[np.newaxis, :]  # (1,3)

    # coord locali dei due nodi
    pts = np.column_stack([x_start, x_end])   # (3,2)
    x1D_e = T_1D @ (pts - x0[:, None])        # (1,2)

    return T_1D, x0, x1D_e


def calcola_cquad8(x1, x2, x3, x4, x5, x6, x7, x8):
 
    # porta tutto a vettori 1D
    x1 = np.asarray(x1, dtype=float).ravel()
    x2 = np.asarray(x2, dtype=float).ravel()
    x3 = np.asarray(x3, dtype=float).ravel()
    x4 = np.asarray(x4, dtype=float).ravel()
    x5 = np.asarray(x5, dtype=float).ravel()
    x6 = np.asarray(x6, dtype=float).ravel()
    x7 = np.asarray(x7, dtype=float).ravel()
    x8 = np.asarray(x8, dtype=float).ravel()

    nodi = np.vstack([x1, x2, x3, x4, x5, x6, x7, x8])  # (8,3)
    baricentro = np.mean(nodi, axis=0)
    nodi_traslati = nodi - baricentro

    A = nodi_traslati[1, :] - nodi_traslati[0, :]
    B = nodi_traslati[2, :] - nodi_traslati[1, :]
    normale = np.cross(A, B)

    # vettore z globale
    z_glob = np.array([0.0, 0.0, 1.0], dtype=float)

    # se la normale è allineata a z_glob → nessuna rotazione
    if np.allclose(np.cross(normale, z_glob), 0.0):
        T_e0, x0 = _calcola_matrice_senza_rotazione(nodi_traslati, normale, baricentro)
    else:
        nodi_ruotati, R = _ruota_nodi(normale, nodi_traslati)
        T_e0, x0 = _calcola_matrice_con_rotazione(nodi_ruotati, normale, R, baricentro)

    return T_e0, x0


def _calcola_matrice_senza_rotazione(nodi_traslati, normale, baricentro):
    x0_traslato = _calcola_intersezione_diagonali(nodi_traslati)
    intersections_traslato = _calcola_punto_E(nodi_traslati, x0_traslato)
    x0 = x0_traslato + baricentro
    intersections = intersections_traslato + baricentro
    T_e0 = _calcola_matrice(intersections, normale, x0)
    return T_e0, x0


def _calcola_matrice_con_rotazione(nodi_ruotati, normale, R, baricentro):
    x0_ruotato = _calcola_intersezione_diagonali(nodi_ruotati)
    intersections_ruotato = _calcola_punto_E(nodi_ruotati, x0_ruotato)

    R_inversa = R.T
    x0_traslato = (R_inversa @ x0_ruotato)
    intersections_traslato = (R_inversa @ intersections_ruotato)

    x0 = x0_traslato + baricentro
    intersections = intersections_traslato + baricentro
    T_e0 = _calcola_matrice(intersections, normale, x0)
    return T_e0, x0


def _calcola_intersezione_diagonali(nodi_traslati):

    x = nodi_traslati[:, 0]
    y = nodi_traslati[:, 1]

    deltax_13 = x[2] - x[0]
    deltax_24 = x[3] - x[1]

    if deltax_13 == 0.0:
        m1 = 1.0
    else:
        m1 = (y[2] - y[0]) / deltax_13

    if deltax_24 == 0.0:
        m2 = 1.0
    else:
        m2 = (y[3] - y[1]) / deltax_24

    b1 = y[0] - m1 * x[0]
    b2 = y[1] - m2 * x[1]

    # x_intersect = (b2 - b1) / (m1 - m2)
    denom = (m1 - m2)
    if abs(denom) < 1e-14:
        # diagonali quasi parallele -> fallback al baricentro
        x_intersect = np.mean(x)
        y_intersect = np.mean(y)
    else:
        x_intersect = (b2 - b1) / denom
        y_intersect = m1 * x_intersect + b1

    return np.array([x_intersect, y_intersect, 0.0], dtype=float)


def _calcola_punto_E(nodi_traslati, x0_traslato):

    x = nodi_traslati[:, 0]
    y = nodi_traslati[:, 1]

    lato_02 = np.sqrt((x[1] - x0_traslato[0])**2 + (y[1] - x0_traslato[1])**2)
    lato_03 = np.sqrt((x[2] - x0_traslato[0])**2 + (y[2] - x0_traslato[1])**2)

    denom = (lato_02 + lato_03)
    if denom < 1e-14:
        # degenerato: usa direttamente x0
        return x0_traslato.copy()

    X = (x[1] * lato_03 + x[2] * lato_02) / denom
    Y = (y[1] * lato_03 + y[2] * lato_02) / denom
    return np.array([X, Y, 0.0], dtype=float)


def _calcola_matrice(intersections, normale, x0):

    xe = intersections - x0
    xe = xe / np.linalg.norm(xe)

    ze = normale / np.linalg.norm(normale)
    ye = np.cross(ze, xe)
    ye = ye / np.linalg.norm(ye)

    T_e0 = np.vstack([xe, ye, ze])  # (3,3)
    return T_e0


def _ruota_nodi(normale, nodi_traslati):

    normale = np.asarray(normale, dtype=float)
    n = normale / np.linalg.norm(normale)

    z_glob = np.array([0.0, 0.0, 1.0], dtype=float)
    cos_theta = np.clip(np.dot(n, z_glob), -1.0, 1.0)
    theta = np.arccos(cos_theta)

    k = np.cross(n, z_glob)
    norm_k = np.linalg.norm(k)
    if norm_k < 1e-14:
        # già allineato
        R = np.eye(3)
    else:
        k = k / norm_k
        K = np.array([
            [0.0,   -k[2],  k[1]],
            [k[2],  0.0,   -k[0]],
            [-k[1], k[0],   0.0]
        ], dtype=float)
        R = np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)

    nodi_traslati = np.asarray(nodi_traslati, dtype=float)
    nodi_ruotati = (R @ nodi_traslati.T).T
    return nodi_ruotati, R


# ------------------------------------------------------------
#  ELEMENT MATRICES (K_e, M_e)
# ------------------------------------------------------------
def element_matrices_cquad8(x_e, c, quad_rule=gauss_rule_quad_3x3):
    """
    Compute the element stiffness and mass matrices for a CQUAD8 element.

    This function computes the local acoustic stiffness matrix (K_e)
    and mass matrix (M_e) for an 8-node quadrilateral (CQUAD8) element
    using numerical integration.

    Parameters
    ----------
    x_e : ndarray of shape (8, 2) or (8, 3)
        Physical coordinates of the element nodes.
        For 2D acoustic problems, only the (x, y) coordinates are used.
        If 3D coordinates are provided, the z-component is ignored.
    c : float
        Speed of sound in the fluid.
    quad_rule : tuple, optional
        Quadrature rule defined as ``(xi_pts, eta_pts, w_pts)``,
        where ``xi_pts`` and ``eta_pts`` are the integration points
        and ``w_pts`` are the corresponding weights.
        Default is a Gauss quadrature rule suitable for CQUAD8 elements.

    Returns
    -------
    K_e : ndarray of shape (8, 8)
        Element acoustic stiffness matrix.
    M_e : ndarray of shape (8, 8)
        Element acoustic mass matrix.

    Notes
    -----
    The element matrices are computed by numerical integration
    in the physical domain using isoparametric mapping.
    The formulation assumes quadratic acoustic pressure interpolation
    over the CQUAD8 element.

    See Also
    --------
    cquad8_shape : Shape functions and derivatives for CQUAD8 elements.
    jacobian_2d : Compute the Jacobian of the isoparametric mapping.
    gauss_rule_quad_3x3 : Default Gauss quadrature rule for CQUAD8 elements.
    """

    x_e = np.asarray(x_e, dtype=float)
    if x_e.shape[1] == 3:
        # se abbiamo 3D, prendiamo solo x,y per integrare nel piano
        x2d = x_e[:, :2]
    else:
        x2d = x_e  # (8,2)

    xi_pts, eta_pts, w_pts = quad_rule()

    K_e = np.zeros((8, 8), dtype=float)
    M_e = np.zeros((8, 8), dtype=float)

    for xi, eta, w in zip(xi_pts, eta_pts, w_pts):
        N, dN_dxi, dN_deta = cquad8_shape(xi, eta)
        J, detJ, invJ = jacobian_2d(dN_dxi, dN_deta, x2d)
        B  = B_stiffness(dN_dxi, dN_deta, invJ)
        Bm = B_mass(N)

        K_e += (B.T @ B) * detJ * w
        M_e += (1.0 / c**2) * (Bm.T @ Bm) * detJ * w

    return K_e, M_e
