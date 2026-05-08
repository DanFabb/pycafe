import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh


def solve_modal_acoustic_reduced(
    K_red,
    M_red,
    num_modes=1,
):
    """
    Modal analysis for acoustic FEM (REDUCED SYSTEM).

    Solves the generalised Hermitian eigenproblem:
        K_red φ = λ M_red φ,  λ = ω²

    Uses ARPACK (``scipy.sparse.linalg.eigsh``) to compute only the
    ``num_modes`` smallest eigenvalues without converting the sparse
    matrices to dense form, giving O(N·k) cost instead of O(N³).

    Parameters
    ----------
    K_red, M_red : sparse or dense (Nr, Nr) real matrix
        Reduced acoustic stiffness and mass matrices after Dirichlet
        elimination.
    num_modes : int
        Number of physical modes to return.

    Returns
    -------
    freqs : (num_modes,) ndarray
        Natural frequencies [Hz], sorted in ascending order.
    modes_red : (Nr, num_modes) ndarray
        Corresponding mode shapes on the reduced DOFs.
    """
    # --------------------------------------------------
    # 1. Ensure sparse CSR format (eigsh works with sparse or dense)
    # --------------------------------------------------
    if not sp.issparse(K_red):
        K_red = sp.csr_matrix(np.asarray(K_red, dtype=float))
    else:
        K_red = K_red.tocsr().astype(float)

    if not sp.issparse(M_red):
        M_red = sp.csr_matrix(np.asarray(M_red, dtype=float))
    else:
        M_red = M_red.tocsr().astype(float)

    # --------------------------------------------------
    # 2. Enforce symmetry numerically
    # --------------------------------------------------
    K_red = 0.5 * (K_red + K_red.T)
    M_red = 0.5 * (M_red + M_red.T)

    # --------------------------------------------------
    # 3. Solve with ARPACK: only num_modes smallest eigenvalues
    #    sigma=0 → shift-invert mode, much faster for smallest λ
    # --------------------------------------------------
    n = K_red.shape[0]
    k = min(num_modes + 2, n - 1)  # request a few extra to handle near-zero modes

    eigvals, eigvecs = eigsh(K_red, k=k, M=M_red, which="LM", sigma=1.0)

    eigvals = np.real(eigvals)

    # --------------------------------------------------
    # 4. Discard non-physical (near-zero) modes
    # --------------------------------------------------
    LAMBDA_MIN = 1.0  # well below any acoustic mode; trivial mode has λ≈0
    valid = eigvals > LAMBDA_MIN
    eigvals = eigvals[valid]
    eigvecs = eigvecs[:, valid]

    if eigvals.size == 0:
        raise RuntimeError("No physical eigenvalues found above threshold.")

    # --------------------------------------------------
    # 5. Sort, truncate, convert to frequencies
    # --------------------------------------------------
    order = np.argsort(eigvals)
    eigvals = eigvals[order][:num_modes]
    eigvecs = eigvecs[:, order][:, :num_modes]

    freqs = np.sqrt(eigvals) / (2.0 * np.pi)

    return freqs, eigvecs
