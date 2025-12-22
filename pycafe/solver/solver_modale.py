import numpy as np
from scipy.linalg import eigh
import scipy.sparse as sp

def solve_modal_acoustic_reduced(
    K_red,
    M_red,
    num_modes=1,
):
    """
    Modal analysis for acoustic FEM (REDUCED SYSTEM).

    Solves:
        K_red φ = λ M_red φ
        λ = ω²

    Parameters
    ----------
    K_red, M_red : (Nr,Nr) ndarray
        Reduced stiffness and mass matrices (Dirichlet already applied).
    num_modes : int
        Number of physical modes to return.
    tol_eig : float
        Threshold to discard zero / spurious eigenvalues.

    Returns
    -------
    freqs : (num_modes,) ndarray
        Natural frequencies [Hz].
    modes_red : (Nr, num_modes) ndarray
        Mode shapes on the REDUCED DOFs.
    """
    # --------------------------------------------------
    # 0. Convert to dense REAL matrices (MANDATORY)
    # --------------------------------------------------
    tol_eig=1e-10

    if sp.issparse(K_red):
        K_red = K_red.toarray()
    if sp.issparse(M_red):
        M_red = M_red.toarray()

    K_red = np.asarray(K_red, dtype=float)
    M_red = np.asarray(M_red, dtype=float)


    # --------------------------------------------------
    # 1. Safety: enforce symmetry (numerical)
    # --------------------------------------------------
    K_red = 0.5 * (K_red + K_red.T)
    M_red = 0.5 * (M_red + M_red.T)

    # --------------------------------------------------
    # 2. Solve generalized eigenproblem
    # --------------------------------------------------
    eigvals, eigvecs = eigh(K_red, M_red)

    eigvals = np.real(eigvals)

    # --------------------------------------------------
    # 3. Remove non-physical / null modes
    # --------------------------------------------------
    valid = eigvals > tol_eig
    eigvals = eigvals[valid]
    eigvecs = eigvecs[:, valid]

    if eigvals.size == 0:
        raise RuntimeError("No physical eigenvalues found (all λ ≈ 0).")

    # --------------------------------------------------
    # 4. Frequencies
    # --------------------------------------------------
    omegas = np.sqrt(eigvals)
    freqs = omegas / (2.0 * np.pi)

    # Sort and truncate
    order = np.argsort(freqs)
    freqs = freqs[order][:num_modes]
    modes_red = eigvecs[:, order][:, :num_modes]

    return freqs, modes_red
