# solver_modal_forced.py
#
# Modal solution method for the acoustic frequency response.
#
# Instead of solving the full system
#
#     (K + j omega C - omega^2 M) p = F(omega)
#
# at every frequency (direct method), the response is expanded on the
# acoustic modes of the rigid, undamped cavity:
#
#     p(omega) ~= Phi phi(omega),        m_a << n_a
#
# and the system is projected onto the basis:
#
#     (Kt + j omega Ct - omega^2 Mt) phi = Phi^T F(omega)
#
# with Kt = Phi^T K Phi, etc. Mass-normalized modes make Mt = I and
# Kt = diag(omega_m^2); with proportional (Rayleigh) damping or modal
# damping ratios the equations decouple into scalars
#
#     phi_m = Ft_m / (omega_m^2 - omega^2 + 2 j zeta_m omega_m omega).
#
# A non-proportional C (e.g. an impedance wall) couples the modes: the
# projected m_a x m_a system is then solved as a small dense complex
# system, which is still far cheaper than the full solve.
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh


def build_modal_basis(
    K_red,
    M_red,
    num_modes,
    include_rigid=True,
):
    """
    Compute a mass-normalized modal basis of the acoustic system.

    Solves the generalized eigenproblem ``K Phi = omega^2 M Phi`` of
    the rigid, undamped cavity and normalizes the modes so that
    ``Phi^T M Phi = I`` (hence ``Phi^T K Phi = diag(omega_m^2)``).

    Parameters
    ----------
    K_red, M_red : sparse or dense (Nr, Nr) real matrices
        Reduced acoustic stiffness and mass matrices.
    num_modes : int
        Number of modes m_a in the basis. The practical rule is to
        keep every mode with natural frequency below about twice the
        highest analysis frequency (f_m < 2 f_max).
    include_rigid : bool, optional
        Keep the constant-pressure mode at ``omega = 0`` present in a
        fully rigid cavity. It carries the quasi-static
        (compressibility) response and must stay in a forced-response
        basis; drop it only when reproducing a free-vibration mode
        count. Default True.

    Returns
    -------
    omegas : ndarray of shape (m,)
        Natural angular frequencies [rad/s], ascending (0 first when
        the rigid mode is present).
    Phi : ndarray of shape (Nr, m)
        Mass-normalized mode shapes.

    See Also
    --------
    solve_modal_frequency_sweep : Forced response on this basis.
    pycafe.solver.solver_modale.solve_modal_acoustic_reduced :
        Free-vibration modal analysis (drops the rigid mode).
    """
    if not sp.issparse(K_red):
        K = sp.csr_matrix(np.asarray(K_red, dtype=float))
    else:
        K = K_red.tocsr().astype(float)
    if not sp.issparse(M_red):
        M = sp.csr_matrix(np.asarray(M_red, dtype=float))
    else:
        M = M_red.tocsr().astype(float)

    K = 0.5 * (K + K.T)
    M = 0.5 * (M + M.T)

    n = K.shape[0]
    if n == 0:
        raise RuntimeError("Reduced modal system is empty.")

    if n <= max(num_modes + 2, 3):
        eigvals, eigvecs = la.eigh(K.toarray(), M.toarray())
    else:
        k = min(num_modes + 2, n - 1)
        # sigma < 0: the shifted matrix K - sigma*M stays definite even
        # when K is singular (rigid cavity), unlike sigma = 0.
        eigvals, eigvecs = eigsh(K, k=k, M=M, which="LM", sigma=-1.0)

    eigvals = np.real(eigvals)
    eigvals[eigvals < 0.0] = 0.0          # numerical noise on the rigid mode

    if not include_rigid:
        scale = max(1.0, float(eigvals.max())) if eigvals.size else 1.0
        keep = eigvals > 1e-10 * scale
        eigvals, eigvecs = eigvals[keep], eigvecs[:, keep]

    order = np.argsort(eigvals)
    eigvals = eigvals[order][:num_modes]
    eigvecs = eigvecs[:, order][:, :num_modes]

    # Mass-normalize: Phi^T M Phi = I.
    for m in range(eigvecs.shape[1]):
        mm = float(eigvecs[:, m] @ (M @ eigvecs[:, m]))
        eigvecs[:, m] /= np.sqrt(mm)

    return np.sqrt(eigvals), eigvecs


def modal_damping_matrix(
    omegas,
    *,
    rayleigh=None,
    modal_zeta=None,
):
    """
    Diagonal modal damping ``Ct = diag(2 zeta_m omega_m)``.

    Parameters
    ----------
    omegas : ndarray of shape (m,)
        Natural angular frequencies of the basis.
    rayleigh : tuple (alpha, beta), optional
        Proportional damping ``C = alpha K + beta M``, which projects
        to ``zeta_m = alpha omega_m / 2 + beta / (2 omega_m)``.
    modal_zeta : float or array-like, optional
        Damping ratio(s) assigned directly per mode; a scalar applies
        to every mode. Added to the Rayleigh contribution when both
        are given.

    Returns
    -------
    Ct : ndarray of shape (m, m)
        Diagonal modal damping matrix (the coefficient of ``j omega``).

    Notes
    -----
    For the rigid mode (``omega_m = 0``) the ``beta / (2 omega_m)``
    term of the Rayleigh formula is singular; its damping contribution
    is ``beta`` directly (``C phi = beta M phi``), which is what the
    projection gives without ever forming zeta.
    """
    omegas = np.asarray(omegas, dtype=float)
    c = np.zeros_like(omegas)

    if rayleigh is not None:
        alpha, beta = rayleigh
        # Project alpha*K + beta*M exactly: alpha*omega_m^2 + beta.
        c += alpha * omegas**2 + beta

    if modal_zeta is not None:
        zeta = np.broadcast_to(
            np.asarray(modal_zeta, dtype=float), omegas.shape
        )
        c += 2.0 * zeta * omegas

    return np.diag(c)


def solve_modal_frequency_sweep(
    K_red,
    M_red,
    frequencies,
    *,
    num_modes,
    C_red=None,
    impedance_operator=None,
    velocity_operator=None,
    source_operator=None,
    rayleigh=None,
    modal_zeta=None,
    include_rigid=True,
    basis=None,
    return_modal=False,
):
    """
    Modal frequency sweep of the forced acoustic response.

    Projects the system on ``num_modes`` rigid-cavity modes and solves
    the small modal system at each frequency:

    .. math::
        (\\tilde K + j \\omega \\tilde C(\\omega) - \\omega^2 \\tilde M)
        \\, \\boldsymbol\\phi = \\Phi^T F(\\omega) ,
        \\qquad p = \\Phi \\boldsymbol\\phi .

    Parameters
    ----------
    K_red, M_red : sparse matrices
        Reduced acoustic stiffness and mass matrices.
    frequencies : array-like of float
        Frequencies of the sweep [Hz].
    num_modes : int
        Size of the modal basis (rule of thumb: keep all modes with
        ``f_m < 2 f_max``). Ignored when ``basis`` is given.
    C_red : sparse matrix, optional
        Constant damping matrix to project (e.g. an assembled
        impedance matrix). Ignored when ``impedance_operator`` is
        given.
    impedance_operator : ImpedanceOperator, optional
        Impedance operator **already reduced** to the free DOFs; it is
        projected per frequency, so frequency-dependent liners work.
    velocity_operator : NormalVelocityOperator, optional
        Normal-velocity load operator, already reduced.
    source_operator : AcousticSourceOperator, optional
        Volumetric source (monopole) operator, already reduced.
    rayleigh : tuple (alpha, beta), optional
        Proportional damping ``C = alpha K + beta M``.
    modal_zeta : float or array-like, optional
        Modal damping ratio(s), scalar or per mode.
    include_rigid : bool, optional
        Keep the constant-pressure mode of a rigid cavity in the
        basis. Default True.
    basis : tuple (omegas, Phi), optional
        Reuse a basis from :func:`build_modal_basis` (e.g. across load
        cases) instead of recomputing it.
    return_modal : bool, optional
        Also return the modal data. Default False.

    Returns
    -------
    P_red : ndarray of complex, shape (N_red, N_freq)
        Reconstructed nodal pressures ``Phi phi(omega)``.
    modal : dict, only if ``return_modal``
        ``omegas`` (rad/s), ``Phi``, and ``participations`` of shape
        (m, N_freq) — the modal participation factors ``phi_m(omega)``.

    Raises
    ------
    ValueError
        If no excitation operator is given (the response would be
        identically zero), or ``basis``/``num_modes`` are inconsistent.

    Notes
    -----
    Prescribed non-zero pressures are not supported by the modal path:
    they are essential conditions, not loads. Use the direct solver
    (:func:`solve_helmholtz_frequency_sweep`) for those models.
    Zero-pressure boundaries are fine — they are already eliminated
    from the reduced system, and the basis is computed on it.

    The projected stiffness/mass are taken as exactly diagonal
    (``diag(omega_m^2)`` and ``I``); with a projected impedance the
    modal system is a small dense complex matrix, solved per frequency.

    See Also
    --------
    build_modal_basis, modal_damping_matrix
    solve_helmholtz_frequency_sweep : The direct counterpart.
    """
    if (velocity_operator is None and source_operator is None):
        raise ValueError(
            "No excitation: give velocity_operator and/or source_operator."
        )

    if basis is not None:
        omegas, Phi = basis
        omegas = np.asarray(omegas, dtype=float)
        Phi = np.asarray(Phi, dtype=float)
    else:
        omegas, Phi = build_modal_basis(
            K_red, M_red, num_modes, include_rigid=include_rigid,
        )

    m = omegas.size
    frequencies = np.atleast_1d(np.asarray(frequencies, dtype=float))
    n_freq = frequencies.size
    n_red = Phi.shape[0]

    # Projected (diagonal) stiffness and mass of the normalized basis.
    Kt = np.diag(omegas**2)
    Mt = np.eye(m)

    # Frequency-independent part of the modal damping.
    Ct0 = modal_damping_matrix(
        omegas, rayleigh=rayleigh, modal_zeta=modal_zeta,
    )

    # Projected impedance: constant C projects once; an operator that
    # depends on frequency is projected inside the loop.
    project_per_freq = False
    if impedance_operator is not None and not impedance_operator.is_empty:
        if impedance_operator.is_frequency_dependent:
            project_per_freq = True
        else:
            Ct0 = Ct0 + Phi.T @ (impedance_operator.at(0.0) @ Phi)
    elif C_red is not None:
        C_red = C_red.tocsr() if sp.issparse(C_red) else np.asarray(C_red)
        if sp.issparse(C_red):
            if C_red.nnz:
                Ct0 = Ct0 + Phi.T @ (C_red @ Phi)
        elif np.any(C_red):
            Ct0 = Ct0 + Phi.T @ (C_red @ Phi)

    P_red = np.zeros((n_red, n_freq), dtype=complex)
    participations = np.zeros((m, n_freq), dtype=complex)

    for i, f in enumerate(frequencies):
        omega = 2.0 * np.pi * f

        # Modal load Ft = Phi^T F(omega).
        F = np.zeros(n_red, dtype=complex)
        if velocity_operator is not None and not velocity_operator.is_empty:
            F += velocity_operator.at(omega)
        if source_operator is not None and not source_operator.is_empty:
            F += source_operator.at(omega)
        Ft = Phi.T @ F

        Ct = Ct0
        if project_per_freq:
            Ct = Ct0 + Phi.T @ (impedance_operator.at(omega) @ Phi)

        Dt = Kt + 1j * omega * Ct - omega**2 * Mt

        if np.count_nonzero(Dt - np.diag(np.diagonal(Dt))) == 0:
            # Uncoupled modes: m scalar divisions.
            phi = Ft / np.diagonal(Dt)
        else:
            phi = np.linalg.solve(Dt, Ft)

        participations[:, i] = phi
        P_red[:, i] = Phi @ phi

    if return_modal:
        return P_red, {
            "omegas": omegas,
            "Phi": Phi,
            "participations": participations,
        }
    return P_red
