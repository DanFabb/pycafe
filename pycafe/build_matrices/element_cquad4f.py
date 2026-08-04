# element_cquad4f.py
#
# CQUAD4F — 4-node structural shell element, 6 DOF/node
# [ux, uy, uz, rx, ry, rz]. Flat-facet Mindlin plate + membrane with
# drilling, in the MacNeal / MSC.Nastran style:
#
#   1. element frame taken at the intersection of the diagonals, with
#      the x axis along the bisector of the diagonals (MacNeal 1978);
#   2. selective reduced integration: direct membrane and bending terms
#      at 2x2 Gauss, in-plane shear and twist terms at the centre;
#   3. transverse shear introduced through MacNeal's residual bending
#      flexibility correction, sampled at the four edge midpoints,
#      rather than through a displacement-based shear strain (no
#      shear locking in the thin limit);
#   4. drilling stiffness coupled to the membrane DOFs (Zienkiewicz
#      type, PARAM,K6ROT).
#
# This implementation is derived from py-polifemo, a finite element
# solver that will shortly be released as open source.
#
# References
# ----------
# R. H. MacNeal, "A simple quadrilateral shell element",
#     Computers & Structures 8 (1978) 175-183.
# R. H. MacNeal, "The evolution of lower order plate and shell elements
#     in MSC/NASTRAN", FE in Analysis and Design 5 (1989) 197-222.

import numpy as np

GAUSS_POINT = 1.0 / np.sqrt(3.0)

# 0-based DOF indices inside the 24-DOF element vector
MEMBRANE_DOF = np.array([0, 1, 6, 7, 12, 13, 18, 19])
DRILLING_DOF = np.array([5, 11, 17, 23])
PLATE_DOF = np.array([2, 3, 4, 8, 9, 10, 14, 15, 16, 20, 21, 22])

# MacNeal's residual-bending-flexibility blending parameter.
# EPSILON_DEFAULT is the value used by the reference implementation;
# EPSILON_MINDLIN is its counterpart when the shear compliance Zs is
# added to the residual bending flexibility (`rbf_mindlin=True`).
EPSILON_DEFAULT = 0.0227
EPSILON_MINDLIN = 0.02642


# ------------------------------------------------------------------
#  SHAPE FUNCTIONS / JACOBIAN
# ------------------------------------------------------------------
def evaluate_shape_functions(xi, eta):
    """Bilinear shape functions and natural derivatives, (4,) each."""
    N = 0.25 * np.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
                         (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)])
    dNdxi = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)])
    dNdeta = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])
    return N, dNdxi, dNdeta


def calculate_jacobian_2d(dNdxi, dNdeta, x2D_e):
    """
    Parameters
    ----------
    x2D_e : ndarray (2, 4)
        In-plane node coordinates in the element frame (columns = nodes).

    Returns
    -------
    J, detJ, invJ
    """
    J = np.vstack([dNdxi, dNdeta]) @ x2D_e.T
    detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
    invJ = np.array([[J[1, 1], -J[0, 1]],
                     [-J[1, 0], J[0, 0]]]) / detJ
    return J, detJ, invJ


def calculate_membrane_B(dNdxi, dNdeta, invJ):
    """Membrane strain-displacement matrix (3, 8), DOF [u1,v1,...,u4,v4]."""
    dNdxy = invJ @ np.vstack([dNdxi, dNdeta])
    dNdx, dNdy = dNdxy[0], dNdxy[1]
    B = np.zeros((3, 8))
    B[0, 0::2] = dNdx
    B[1, 1::2] = dNdy
    B[2, 0::2] = dNdy
    B[2, 1::2] = dNdx
    return B


def calculate_curvature_B(dNdxi, dNdeta, invJ):
    """
    Plate curvature-displacement matrix (3, 12), DOF [w, rx, ry] per node.

    Rows 0 and 1 hold the two direct curvatures (swapped and sign
    flipped with respect to the usual [kxx, kyy] ordering, which is
    immaterial in B' E B because the direct 2x2 block of E is symmetric
    under that permutation); row 2 holds the twist.
    """
    dNdxy = invJ @ np.vstack([dNdxi, dNdeta])
    dNdx, dNdy = dNdxy[0], dNdxy[1]
    B = np.zeros((3, 12))
    for j in range(4):
        c = 3 * j
        B[0, c + 1] = dNdy[j]
        B[1, c + 2] = -dNdx[j]
        B[2, c + 1] = -dNdx[j]
        B[2, c + 2] = dNdy[j]
    return B


# ------------------------------------------------------------------
#  ELEMENT FRAME
# ------------------------------------------------------------------
def _diagonal_intersection(nodes):
    """Intersection of the diagonals 1-3 and 2-4 (in-plane)."""
    dx13 = nodes[2, 0] - nodes[0, 0]
    dx24 = nodes[3, 0] - nodes[1, 0]
    m1 = 1.0 if dx13 == 0.0 else (nodes[2, 1] - nodes[0, 1]) / dx13
    b1 = nodes[0, 1] - m1 * nodes[0, 0]
    m2 = 1.0 if dx24 == 0.0 else (nodes[3, 1] - nodes[1, 1]) / dx24
    b2 = nodes[1, 1] - m2 * nodes[1, 0]
    xi = (b2 - b1) / (m1 - m2)
    return np.array([xi, m1 * xi + b1, 0.0])


def _bisector_point(nodes, x0):
    """
    Point where the bisector of the angle 2-x0-3 meets the edge 2-3.
    It defines the element x axis (MacNeal).
    """
    l02 = np.hypot(nodes[1, 0] - x0[0], nodes[1, 1] - x0[1])
    l03 = np.hypot(nodes[2, 0] - x0[0], nodes[2, 1] - x0[1])
    E = np.zeros(3)
    E[0] = (nodes[1, 0] * l03 + nodes[2, 0] * l02) / (l02 + l03)
    E[1] = (nodes[1, 1] * l03 + nodes[2, 1] * l02) / (l02 + l03)
    return E


def element_frame(x_e):
    """
    Element reference frame of the CQUAD4F.

    Parameters
    ----------
    x_e : ndarray (4, 3)
        Global node coordinates (Gmsh/Nastran CQUAD4 ordering).

    Returns
    -------
    T_e0 : ndarray (3, 3)
        Basic -> element rotation (rows = element axes in global
        components), so that x_element = T_e0 @ (x_global - x0).
    x0 : ndarray (3,)
        Element origin (diagonal intersection) in global coordinates.
    """
    x_e = np.asarray(x_e, dtype=float)
    centroid = x_e.mean(axis=0)
    nodes = x_e - centroid
    normal = np.cross(nodes[1] - nodes[0], nodes[2] - nodes[0])

    n = normal / np.linalg.norm(normal)
    if np.linalg.norm(np.cross(n, [0.0, 0.0, 1.0])) < 1e-12:
        # already in a plane parallel to xy
        x0_t = _diagonal_intersection(nodes)
        E_t = _bisector_point(nodes, x0_t)
    else:
        # rotate onto the xy plane, locate the axes there, rotate back
        theta = np.arccos(np.clip(n @ np.array([0.0, 0.0, 1.0]), -1.0, 1.0))
        k = np.cross(n, [0.0, 0.0, 1.0])
        k /= np.linalg.norm(k)
        K = np.array([[0.0, -k[2], k[1]],
                      [k[2], 0.0, -k[0]],
                      [-k[1], k[0], 0.0]])
        R = np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)
        nodes_rot = nodes @ R.T
        x0_r = _diagonal_intersection(nodes_rot)
        E_r = _bisector_point(nodes_rot, x0_r)
        x0_t = R.T @ x0_r
        E_t = R.T @ E_r

    x0 = x0_t + centroid
    E = E_t + centroid

    xe = E - x0
    xe /= np.linalg.norm(xe)
    ze = normal / np.linalg.norm(normal)
    ye = np.cross(ze, xe)
    ye /= np.linalg.norm(ye)
    return np.vstack([xe, ye, ze]), x0


# ------------------------------------------------------------------
#  RESIDUAL BENDING FLEXIBILITY (MacNeal)
# ------------------------------------------------------------------
def residual_bending_flexibility(x2D_e, tNodes, E2dMembrane, E2dShear,
                                 area, I0, epsilon, mindlin=False):
    """
    MacNeal's transverse shear correction, a (24, 24) stiffness
    contribution acting on the plate DOFs only.

    The element is sampled at the four edge midpoints (at the Gauss
    positions along each edge); at each of them the transverse shear
    strain is written as dw/dn combined with the corresponding
    rotation. The resulting 4-term strain is weighted by the inverse of
    the *residual* flexibility Zb, i.e. the difference between the
    exact bending flexibility of a beam strip and the one the
    discretised element already provides, so that the element
    reproduces the exact tip rotation of the strip.

    `epsilon` blends the two in-plane directions according to the
    element aspect ratio; `mindlin=True` adds the true shear
    compliance Zs to Zb, which makes the element sensitive to the
    thickness (Mindlin behaviour) instead of behaving as a thin
    (Kirchhoff) plate.
    """
    pts = np.array([[0.0, -GAUSS_POINT],     # a: eta-, samples d/dx
                    [0.0, GAUSS_POINT],      # b: eta+, samples d/dx
                    [-GAUSS_POINT, 0.0],     # c: xi-,  samples d/dy
                    [GAUSS_POINT, 0.0]])     # d: xi+,  samples d/dy
    N_, dNdx_, dNdy_, detJ_ = [], [], [], []
    for xi, eta in pts:
        N, dxi, deta = evaluate_shape_functions(xi, eta)
        _, detJ, invJ = calculate_jacobian_2d(dxi, deta, x2D_e)
        dN = invJ @ np.vstack([dxi, deta])
        N_.append(N); dNdx_.append(dN[0]); dNdy_.append(dN[1])
        detJ_.append(detJ)
    Na, Nb, Nc, Nd = N_

    # shear compliance at the sampling points
    Vs = np.diag(np.sqrt(2.0 * np.array(detJ_) * np.asarray(tNodes)))
    g11, g12, g22 = E2dShear[0, 0], E2dShear[0, 1], E2dShear[1, 1]
    Gs = 6.0 / 5.0 * np.array([
        [g11, 0.0, 0.5 * g12, 0.5 * g12],
        [0.0, g11, 0.5 * g12, 0.5 * g12],
        [0.5 * g12, 0.5 * g12, g22, 0.0],
        [0.5 * g12, 0.5 * g12, 0.0, g22]])
    Zs = np.linalg.inv(Vs) @ np.linalg.inv(Gs) @ np.linalg.inv(Vs)

    # residual bending flexibility
    deltax = 0.5 * (x2D_e[0, 1] + x2D_e[0, 2] - x2D_e[0, 0] - x2D_e[0, 3])
    deltay = 0.5 * (x2D_e[1, 2] + x2D_e[1, 3] - x2D_e[1, 0] - x2D_e[1, 1])
    a = epsilon / (epsilon + (1 - epsilon) * (deltax**2 / deltay**2))
    b = epsilon / (epsilon + (1 - epsilon) * (deltay**2 / deltax**2))
    e11, e22 = E2dMembrane[0, 0], E2dMembrane[1, 1]
    Zb = 1.0 / (12.0 * area * I0) * np.array([
        [(1 + a) * deltax**2 / e11, (1 - a) * deltax**2 / e11, 0.0, 0.0],
        [(1 - a) * deltax**2 / e11, (1 + a) * deltax**2 / e11, 0.0, 0.0],
        [0.0, 0.0, (1 + b) * deltay**2 / e22, (1 - b) * deltay**2 / e22],
        [0.0, 0.0, (1 - b) * deltay**2 / e22, (1 + b) * deltay**2 / e22]])

    Zinv = np.linalg.inv(Zb + Zs) if mindlin else np.linalg.inv(Zb)

    def rows(j):
        # each row pairs a physical derivative of w with a rotation, so
        # that it vanishes on rigid-body motion of any quadrilateral
        return np.array([
            [-dNdy_[2][j], Nc[j], 0.0],
            [-dNdy_[3][j], Nd[j], 0.0],
            [-dNdx_[0][j], 0.0, -Na[j]],
            [-dNdx_[1][j], 0.0, -Nb[j]]])

    Kadd = np.zeros((24, 24))
    Rj = [rows(j) for j in range(4)]
    for ii in range(4):
        for jj in range(4):
            Kadd[6 * ii + 2:6 * ii + 5, 6 * jj + 2:6 * jj + 5] = \
                Rj[ii].T @ Zinv @ Rj[jj]
    return Kadd


# ------------------------------------------------------------------
#  ELEMENT MATRICES
# ------------------------------------------------------------------
def element_matrices_cquad4f(x_e, t, rho, E, nu, nsm=0.0,
                             epsilon=EPSILON_DEFAULT, k6rot=1000.0,
                             rbf_mindlin=False):
    """
    Stiffness and mass matrices of the CQUAD4F shell element.

    Parameters
    ----------
    x_e : ndarray (4, 3)
        Global node coordinates, Gmsh/Nastran CQUAD4 ordering.
    t : float or array_like (4,)
        Thickness, constant or one value per node.
    rho, E, nu : float
        Density, Young's modulus, Poisson ratio.
    nsm : float, optional
        Non-structural mass per unit area.
    epsilon : float, optional
        MacNeal's residual-bending-flexibility parameter. The default
        is the value of the reference implementation; calibrating it
        against MSC.Nastran gives ~0.034 (see
        examples/cquad4_plate_validation.ipynb).
    k6rot : float, optional
        Drilling stiffness parameter, equivalent to Nastran's
        PARAM,K6ROT.
    rbf_mindlin : bool, optional
        Add the shear compliance Zs to the residual bending
        flexibility. False (default) is the thin-plate limit, in which
        the element is thickness independent; True makes it sensitive
        to transverse shear and is recommended for a/t < 50.

    Returns
    -------
    K_e, M_e : ndarray (24, 24)
        Global-frame stiffness and consistent mass (translational
        only, no rotary inertia), DOF order [ux,uy,uz,rx,ry,rz] per
        node.

    Notes
    -----
    K has all six rigid-body modes in its null space, except for a
    residual of order 1e-4 on the rotation about the element normal:
    the drilling penalty assembles the drilling-drilling and
    drilling-membrane blocks (both identical to the Nastran element
    matrix) but not the membrane-membrane block that would complete the
    quadratic form.
    """
    x_e = np.asarray(x_e, dtype=float)
    if x_e.shape != (4, 3):
        raise ValueError(f"x_e must have shape (4, 3), got {x_e.shape}")
    tNodes = np.full(4, float(t)) if np.isscalar(t) else np.asarray(t, float)

    T_e0, x0 = element_frame(x_e)
    x_loc = T_e0 @ (x_e - x0).T          # (3, 4)
    x2D_e = x_loc[:2, :]                 # (2, 4)

    E2dMembrane = E / (1 - nu**2) * np.array([[1.0, nu, 0.0],
                                              [nu, 1.0, 0.0],
                                              [0.0, 0.0, (1 - nu) / 2]])
    E2dBending = E2dMembrane.copy()
    G = E / (2 * (1 + nu))
    E2dShear = (5.0 / 6.0) * G * np.eye(2)
    I0 = np.mean(tNodes) ** 3 / 12.0

    # --- 2x2 Gauss loop: direct membrane and bending terms, mass, area
    Xi = GAUSS_POINT * np.array([-1.0, 1.0, 1.0, -1.0])
    Eta = GAUSS_POINT * np.array([-1.0, -1.0, 1.0, 1.0])
    membrane_k = np.zeros((8, 8))
    bending_k = np.zeros((12, 12))
    consistent_mass = np.zeros((4, 4))
    area = 0.0
    for i in range(4):
        N, dNdxi, dNdeta = evaluate_shape_functions(Xi[i], Eta[i])
        _, detJ, invJ = calculate_jacobian_2d(dNdxi, dNdeta, x2D_e)
        tGauss = N @ tNodes
        consistent_mass += np.outer(N, N) * (nsm + rho * tGauss) * detJ
        Bm = calculate_membrane_B(dNdxi, dNdeta, invJ)
        membrane_k += Bm[:2].T @ E2dMembrane[:2, :2] @ Bm[:2] * detJ * tGauss
        Bp = calculate_curvature_B(dNdxi, dNdeta, invJ)
        bending_k += (tGauss**3 / 12.0) * Bp[:2].T @ E2dBending[:2, :2] \
            @ Bp[:2] * detJ
        area += detJ

    # --- centre point: in-plane shear and twist (reduced integration)
    N, dNdxi, dNdeta = evaluate_shape_functions(0.0, 0.0)
    _, detJ0, invJ0 = calculate_jacobian_2d(dNdxi, dNdeta, x2D_e)
    centerT = N @ tNodes
    centerBp = calculate_curvature_B(dNdxi, dNdeta, invJ0)
    centerBm = calculate_membrane_B(dNdxi, dNdeta, invJ0)
    membrane_shear_k = 4.0 * np.outer(centerBm[2], centerBm[2]) \
        * E2dMembrane[2, 2] * detJ0 * centerT
    bending_k += 4.0 * np.outer(centerBp[2], centerBp[2]) \
        * E2dBending[2, 2] * detJ0 * (centerT**3 / 12.0)

    # --- transverse shear: MacNeal residual bending flexibility
    Kadd = residual_bending_flexibility(x2D_e, tNodes, E2dMembrane, E2dShear,
                                        area, I0, epsilon, rbf_mindlin)

    # --- assembly in the element frame
    k_e = np.zeros((24, 24))
    k_e[np.ix_(MEMBRANE_DOF, MEMBRANE_DOF)] = membrane_k + membrane_shear_k
    k_e[np.ix_(PLATE_DOF, PLATE_DOF)] = bending_k

    krot = k6rot / 1e7 * E2dMembrane[2, 2] * centerT
    k_e[np.ix_(DRILLING_DOF, DRILLING_DOF)] = detJ0 * krot * 4.0 * np.eye(4)
    coupling = detJ0 * krot * invJ0[0, 0] * np.array([
        [-1.0, 1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, -1.0, -1.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 1.0]])
    k_e[np.ix_(DRILLING_DOF, MEMBRANE_DOF)] = coupling
    k_e[np.ix_(MEMBRANE_DOF, DRILLING_DOF)] = coupling.T

    k_e += Kadd

    m_e = np.zeros((24, 24))
    for d in range(3):
        m_e[np.ix_(np.arange(d, 24, 6), np.arange(d, 24, 6))] = consistent_mass

    R = np.kron(np.eye(8), T_e0)
    return R.T @ k_e @ R, R.T @ m_e @ R
