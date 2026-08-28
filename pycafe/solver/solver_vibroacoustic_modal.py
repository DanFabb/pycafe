r"""
Modal reduction of the coupled vibroacoustic problem — two optional bases.

The reference solution stays the direct sweep
(:func:`~pycafe.solver.solver_vibroacoustic.solve_vibroacoustic_frequency_sweep`),
which factorizes the complex unsymmetric matrix ``A(omega)`` once per
frequency and is exact. What follows are two ways of trading some of
that accuracy for speed when the sweep is long: the response is expanded
on a small basis,

.. math::

   \mathbf q =
   \begin{Bmatrix}\mathbf w\\ \mathbf p\end{Bmatrix}
   \simeq \mathbf \Phi \boldsymbol\varphi ,
   \qquad m \ll n_s + n_a ,

and an ``m x m`` system is solved per frequency instead. Neither method
is required to run a coupled analysis; pick one when the direct solver
is too slow, and check it against the direct solver on a few lines.

Both bases feed the same solver,
:func:`solve_coupled_modal_frequency_sweep`.

1. Coupled modes — :func:`build_coupled_modal_basis`
----------------------------------------------------
The basis is made of the modes of the *whole* fluid-structure system:
each one carries a structural shape **and** the pressure field that goes
with it. It is the most efficient basis per vector — it already knows
about the coupling — and the most expensive to compute, because the
coupled eigenproblem is unsymmetric.

Left and right modes. The coupled matrices are unsymmetric (a
displacement unknown paired with a pressure unknown puts the two
coupling terms in different blocks), so the eigenproblem has distinct
right and left eigenvectors,

.. math::

   \mathbf K \mathbf \Phi^R_c = \omega_c^2 \mathbf M \mathbf \Phi^R_c ,
   \qquad
   \mathbf K^T \mathbf \Phi^L_c = \omega_c^2 \mathbf M^T \mathbf \Phi^L_c ,

and the projection must use both, ``(Phi^L)^T A Phi^R``. For this block
structure the left modes need **no second eigenvalue solve** — they
follow from the right ones,

.. math::

   \mathbf \Phi^L_c =
   \begin{Bmatrix}
     \rho_0\,\omega_c^2\, \mathbf \Phi_{s,c} \\ \mathbf \Phi_{a,c}
   \end{Bmatrix} :

same acoustic part, structural part scaled by
:math:`\rho_0\omega_c^2`. (Texts that divide the acoustic row by
:math:`\rho_0` write the same relation with the factor
:math:`\omega_c^2`; the ``rho_0`` here follows the scaling of
:func:`~pycafe.solver.solver_vibroacoustic.coupled_modal_matrices`.)

Right modes are not orthogonal to each other. What holds is
biorthogonality between the two families,
:math:`(\mathbf \Phi^L_i)^T \mathbf M \mathbf \Phi^R_j = 0` for
:math:`i \neq j`, so normalizing the diagonal to 1 makes the projected
mass the identity and the projected stiffness ``diag(omega_c^2)``.

2. Component Mode Synthesis — :func:`build_cms_basis`
------------------------------------------------------
CMS never touches the coupled eigenproblem. The two components are
solved separately, both **symmetric**,

.. math::

   \mathbf K_s \mathbf \Phi_{su} = \omega_{s}^2 \mathbf M_s \mathbf \Phi_{su} ,
   \qquad
   \mathbf K_a \mathbf \Phi_{au} = \omega_{a}^2 \mathbf M_a \mathbf \Phi_{au} ,

— the structure in vacuo (no pressure load on the interface) and the cavity
with **hard walls** — and the two bases are stacked block-diagonally,

.. math::

   \begin{Bmatrix}\mathbf w\\ \mathbf p\end{Bmatrix} =
   \begin{bmatrix}\mathbf \Phi_{su} & \mathbf 0\\
                  \mathbf 0 & \mathbf \Phi_{au}\end{bmatrix}
   \begin{Bmatrix}\boldsymbol\varphi_s\\ \boldsymbol\varphi_a\end{Bmatrix}.

The coupling comes back in the *reduced* matrices, and only there:

.. math::

   \tilde{\mathbf K} =
   \begin{bmatrix}\mathbf \Lambda_s & -\mathbf A\\
                  \mathbf 0 & \mathbf \Lambda_a\end{bmatrix},
   \qquad
   \tilde{\mathbf M} =
   \begin{bmatrix}\mathbf I_s & \mathbf 0\\
                  \rho_0 \mathbf A^T & \mathbf I_a\end{bmatrix},
   \qquad
   \mathbf A = \mathbf \Phi_{su}^T \mathbf K_c \mathbf \Phi_{au} .

``A`` is the coupling in modal coordinates, and it is worth looking at:
``A[i, j]`` measures how much structural mode ``i`` talks to acoustic
mode ``j`` — how well that plate shape pushes the fluid the way that
cavity mode wants to move. It is returned in the basis as
``modal_coupling``. Note that the reduced system stays unsymmetric and
coupled: normalizing each component's modes buys diagonal ``Lambda``
blocks, not independence.

The trade. Building the basis is much cheaper — two symmetric
eigenproblems instead of one large unsymmetric one — but the basis is
less efficient, so more vectors are needed for the same accuracy. The
reason is the hard-wall assumption: every acoustic mode has
:math:`u_{f,n} = 0` on the interface, while the true condition there is
:math:`u_{f,n} = u_{s,n}`. Near the flexible wall the field a vibrating
plate produces is largely evanescent, with strong normal gradients, and
global hard-wall modes reconstruct it slowly — so the pressure *near
the plate* is what converges last.

Refinements exist for that: acoustic modes computed with an impedance
condition on the interface instead of a hard wall (efficient, but
choosing ``Z`` is an open question), or fixed-interface modes
(``p = 0`` on the interface) plus one constraint mode per interface DOF
— exact in principle, but ``N_Gamma`` extra vectors on a finely meshed
interface leaves little reduction. Neither is implemented here.

Choosing
--------
================  ======================  ==========================
Basis             Cost to build           Efficiency per vector
================  ======================  ==========================
Coupled modes     high (unsymmetric)      high
CMS               low (two symmetric)     lower, needs more vectors
================  ======================  ==========================

Truncation, either way: keep every mode up to about **twice** the
highest frequency of interest. It is a guideline, not a law — point
loads and near-field results need more, since the modes left out still
carry a quasi-static flexibility that a concentrated load excites — so
convergence with respect to the basis size is the only real check, and
the direct solver is the reference to check against.

See Also
--------
pycafe.solver.solver_vibroacoustic : Direct coupled sweep and modes.
pycafe.solver.solver_modal_forced : The same idea on the uncoupled fluid.
"""

import warnings

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, eigs, splu

from pycafe.solver.solver_modal_forced import (
    build_modal_basis,
    modal_damping_matrix,
)
from pycafe.solver.solver_vibroacoustic import (
    _as_sweep,
    build_coupled_blocks,
    coupled_modal_matrices,
)


def _refuse_frequency_dependent_impedance(blocks):
    """
    A liner that changes with frequency has no place in a modal basis.

    The reduction projects one damping matrix onto the modes and reuses
    it at every frequency, so an admittance that depends on omega would
    be frozen at whichever value happened to be projected. That is the
    wrong answer rather than a coarse one, so it raises.
    """
    if blocks.get("impedance") is not None:
        raise ValueError(
            "These blocks carry a frequency-dependent impedance, which no "
            "single modal damping matrix can stand for. Solve this model "
            "with solve_vibroacoustic_frequency_sweep, or state the liner "
            "as a constant admittance."
        )


def _blocks_for_basis(system, blocks, pressure_zero_nodes0, C_a):
    """Resolve and validate the coupled blocks a basis is built on."""
    if blocks is None:
        if system is None:
            raise ValueError("Give either 'system' or 'blocks'.")
        blocks = build_coupled_blocks(
            system, pressure_zero_nodes0=pressure_zero_nodes0, C_a=C_a,
        )
    _refuse_frequency_dependent_impedance(blocks)
    if np.issubdtype(blocks["Ks"].dtype, np.complexfloating):
        raise ValueError(
            "The coupled blocks carry a complex stiffness: build the basis "
            "on the undamped system (eta_s=0) and pass the loss factor to "
            "the forced solution instead."
        )
    return blocks


def left_modes_from_right(Phi_R, omegas, n_s, rho0):
    r"""
    Left modes of the coupled eigenproblem, from the right ones.

    Applies :math:`\mathbf \Phi^L = [\rho_0\omega_c^2 \mathbf \Phi_s;
    \mathbf \Phi_a]`, the relation that holds for the block structure of
    :func:`~pycafe.solver.solver_vibroacoustic.coupled_modal_matrices`.

    Parameters
    ----------
    Phi_R : ndarray (n_s + n_a, m)
        Right modes, one per column.
    omegas : ndarray (m,)
        Natural angular frequencies [rad/s].
    n_s : int
        Number of structural unknowns (first block).
    rho0 : float
        Fluid density.

    Returns
    -------
    Phi_L : ndarray (n_s + n_a, m)
    """
    Phi_R = np.asarray(Phi_R)
    omegas = np.asarray(omegas, dtype=float)
    Phi_L = Phi_R.copy()
    Phi_L[:n_s, :] = rho0 * omegas ** 2 * Phi_R[:n_s, :]
    return Phi_L


def build_coupled_modal_basis(
    system=None,
    num_modes=10,
    *,
    blocks=None,
    pressure_zero_nodes0=None,
    C_a=None,
    sigma=1.0,
    check=True,
):
    r"""
    Basis of coupled modes, biorthonormalized (method 1).

    Solves the undamped coupled eigenproblem, builds the left modes from
    the right ones (no second eigensolve) and scales them so that
    ``(Phi^L)^T M Phi^R = I``, hence ``(Phi^L)^T K Phi^R =
    diag(omega_c^2)``.

    Parameters
    ----------
    system : dict, optional
        Prepared vibroacoustic system, coupling included. Not needed
        when ``blocks`` is given.
    num_modes : int, optional
        Size ``m_c`` of the basis. Rule of thumb: cover every mode up to
        about twice the highest frequency of the sweep.
    blocks : dict, optional
        Pre-reduced blocks from
        :func:`~pycafe.solver.solver_vibroacoustic.build_coupled_blocks`.
        Reuse them to keep an acoustic damping matrix across calls.
    pressure_zero_nodes0, C_a : optional
        Passed to ``build_coupled_blocks`` when ``blocks`` is not given.
        ``C_a`` never enters the eigenproblem — it is carried along for
        the forced solution.
    sigma : float, optional
        Shift of the shift-invert, in ``omega^2``.
    check : bool, optional
        Measure the biorthogonality residual of the basis (cheap, and
        the honest way to know the modes are usable). Default True.

    Returns
    -------
    basis : dict
        ``omegas`` (rad/s), ``freqs`` (Hz), ``Phi_R``, ``Phi_L``,
        ``blocks``, ``n_s``, ``n_a``, ``idx_s``, ``idx_a``, ``kind``
        (``"coupled"``) and, when ``check``, ``biorthogonality`` — the
        largest off-diagonal term of ``(Phi^L)^T M Phi^R`` after
        normalization, which should be many orders below 1.

    Raises
    ------
    ValueError
        If neither ``system`` nor ``blocks`` is given, if the blocks
        carry a complex stiffness (a structural loss factor belongs to
        the forced solution, not to the eigenproblem), or if a mode
        cannot be normalized because its biorthogonality product
        vanishes.

    Notes
    -----
    The zero-frequency mode of a sealed cavity (uniform pressure) is
    kept: it carries the quasi-static part of the pressure response,
    exactly as the rigid mode does in the uncoupled acoustic basis. Pin
    one pressure DOF through ``pressure_zero_nodes0`` to remove it.
    """
    blocks = _blocks_for_basis(system, blocks, pressure_zero_nodes0, C_a)

    K, M = coupled_modal_matrices(blocks)
    n_s, n_a = blocks["n_s"], blocks["n_a"]
    n_total = n_s + n_a
    m = min(int(num_modes), n_total - 2)

    # Shift-invert by hand: ARPACK's M argument assumes a symmetric
    # positive definite mass matrix, which the coupled M is not.
    lu = splu((K - sigma * M).tocsc())
    OP = LinearOperator(
        (n_total, n_total), matvec=lambda v: lu.solve(M @ v), dtype=float,
    )
    mu, vecs = eigs(OP, k=m, which="LM")

    omega2 = np.real(sigma + 1.0 / mu)
    order = np.argsort(omega2)
    omega2 = np.clip(omega2[order], 0.0, None)
    omegas = np.sqrt(omega2)
    Phi_R = np.real_if_close(vecs[:, order])
    if np.iscomplexobj(Phi_R):
        # Complex pairs mean the reduction below is no longer real; keep
        # them, but say so -- they usually signal a defective or badly
        # scaled coupled problem.
        warnings.warn(
            "Complex coupled modes were returned by the eigensolver; the "
            "modal basis is complex.", RuntimeWarning,
        )

    Phi_L = left_modes_from_right(Phi_R, omegas, n_s, blocks["rho0"])

    # Biorthonormalization: scaling Phi_R by s scales Phi_L by s too, so
    # the product scales by s^2.
    prod = np.einsum("ic,ic->c", Phi_L.conj(), M @ Phi_R)
    if np.any(np.abs(prod) < 1e-30):
        raise ValueError(
            "A coupled mode has a vanishing biorthogonality product and "
            "cannot be normalized; the basis is degenerate."
        )
    scale = 1.0 / np.sqrt(np.abs(prod))
    Phi_R = Phi_R * scale
    Phi_L = Phi_L * (scale * np.sign(np.real(prod)))

    basis = {
        "kind": "coupled",
        "omegas": omegas,
        "freqs": omegas / (2.0 * np.pi),
        "Phi_R": Phi_R,
        "Phi_L": Phi_L,
        "blocks": blocks,
        "n_s": n_s,
        "n_a": n_a,
        "idx_s": blocks["idx_s"],
        "idx_a": blocks["idx_a"],
    }

    if check:
        G = Phi_L.conj().T @ (M @ Phi_R)
        basis["biorthogonality"] = float(
            np.abs(G - np.diag(np.diag(G))).max()
        )

    return basis


def build_cms_basis(
    system=None,
    *,
    num_structural=10,
    num_acoustic=20,
    blocks=None,
    pressure_zero_nodes0=None,
    C_a=None,
    include_rigid=True,
):
    r"""
    Component Mode Synthesis basis: structure in vacuo + hard-wall cavity (method 2).

    Solves the two **symmetric** component eigenproblems separately and
    stacks their mass-normalized modes block-diagonally. The coupling is
    reintroduced only when the system is projected, through
    ``A = Phi_su^T Kc Phi_au``.

    Parameters
    ----------
    system : dict, optional
        Prepared vibroacoustic system. Not needed when ``blocks`` is
        given.
    num_structural : int, optional
        Number of in vacuo structural modes ``m_s``.
    num_acoustic : int, optional
        Number of hard-wall acoustic modes ``m_a``. Expect to need more
        of these than of the structural ones: hard-wall modes represent
        the field near the flexible wall poorly.
    blocks, pressure_zero_nodes0, C_a : optional
        As in :func:`build_coupled_modal_basis`.
    include_rigid : bool, optional
        Keep the uniform-pressure mode of a sealed cavity
        (``omega = 0``). It carries the compressibility of the trapped
        air, i.e. the air-spring effect, and belongs in a forced-response
        basis. Default True.

    Returns
    -------
    basis : dict
        Same keys as :func:`build_coupled_modal_basis`, with
        ``kind = "cms"``, plus ``omegas_s`` / ``omegas_a`` (the component
        frequencies), ``num_structural`` / ``num_acoustic``, and
        ``modal_coupling`` — the matrix ``A`` above, whose entry
        ``(i, j)`` says how strongly structural mode ``i`` and acoustic
        mode ``j`` drive each other.

    Notes
    -----
    The test space is the basis itself (plain Galerkin): with
    block-diagonal ``Phi`` the projection reproduces exactly the CMS
    reduced matrices ``[[Lambda_s, -A], [0, Lambda_a]]`` and
    ``[[I, 0], [rho0 A^T, I]]``, which are still coupled and still
    unsymmetric — normalizing each component only diagonalizes its own
    block.

    A structural mode with no net volume displacement has a whole row of
    ``A`` near zero: it neither compresses the air nor is loaded by it.
    That is the modal-coordinate version of a mode that a centred point
    force cannot excite.
    """
    blocks = _blocks_for_basis(system, blocks, pressure_zero_nodes0, C_a)
    n_s, n_a = blocks["n_s"], blocks["n_a"]

    omega_s, Phi_su = build_modal_basis(
        blocks["Ks"], blocks["Ms"], int(num_structural), include_rigid=True,
    )
    omega_a, Phi_au = build_modal_basis(
        blocks["Ka"], blocks["Ma"], int(num_acoustic),
        include_rigid=include_rigid,
    )

    m_s, m_a = omega_s.size, omega_a.size
    Phi = np.zeros((n_s + n_a, m_s + m_a))
    Phi[:n_s, :m_s] = Phi_su
    Phi[n_s:, m_s:] = Phi_au

    return {
        "kind": "cms",
        "omegas": np.concatenate([omega_s, omega_a]),
        "freqs": np.concatenate([omega_s, omega_a]) / (2.0 * np.pi),
        "omegas_s": omega_s,
        "omegas_a": omega_a,
        "Phi_R": Phi,
        "Phi_L": Phi,                       # Galerkin: same test space
        "blocks": blocks,
        "n_s": n_s,
        "n_a": n_a,
        "num_structural": m_s,
        "num_acoustic": m_a,
        "modal_coupling": Phi_su.T @ (blocks["Kc"] @ Phi_au),
        "idx_s": blocks["idx_s"],
        "idx_a": blocks["idx_a"],
    }


def project_coupled_system(basis, *, eta_s=0.0, rayleigh=None, modal_zeta=None):
    r"""
    Projected matrices of the reduced coupled system.

    Returns the three ``m x m`` matrices of

    .. math::
        \left(\tilde{\mathbf K} + j\omega\tilde{\mathbf C}
              - \omega^2 \tilde{\mathbf M}\right)\boldsymbol\varphi
        = (\mathbf \Phi^L)^T \mathbf F ,

    for either basis. The structural loss factor enters ``Kt`` as
    ``j eta_s`` times the projected structural stiffness — not as
    ``j eta_s Kt``, since only the structure is damped that way.

    Parameters
    ----------
    basis : dict
        Output of :func:`build_coupled_modal_basis` or
        :func:`build_cms_basis`.
    eta_s : float, optional
        Structural loss factor: ``Ks (1 + j eta_s)``.
    rayleigh : tuple (alpha, beta), optional
        Proportional damping applied to the modal equations.
    modal_zeta : float or array-like, optional
        Modal damping ratio(s), scalar or one per basis vector.

    Returns
    -------
    Kt, Ct, Mt : ndarray (m, m)
        ``Ct`` is the coefficient of ``j omega``: the projected acoustic
        damping plus the diagonal modal damping. It is generally **not**
        diagonal — an impedance boundary couples the modes.

    Notes
    -----
    ``Kt`` and ``Mt`` are *computed*, not assumed equal to their
    theoretical forms. For a clean coupled basis they are
    ``diag(omega_c^2)`` and ``I`` to round-off; for CMS they come out as
    the block matrices of the module header. Doing the products for real
    costs two matrix multiplications and keeps the reduction exact
    (Petrov-Galerkin) whatever the eigensolver returned — which matters
    with the repeated eigenvalues of a box cavity, where the vectors
    inside a cluster are arbitrary.
    """
    blocks = basis["blocks"]
    n_s = basis["n_s"]
    omegas = basis["omegas"]
    Phi_R, Phi_L = basis["Phi_R"], basis["Phi_L"]

    K, M = coupled_modal_matrices(blocks)
    Kt = (Phi_L.conj().T @ (K @ Phi_R)).astype(complex)
    Mt = Phi_L.conj().T @ (M @ Phi_R)

    if eta_s:
        Ks_t = Phi_L[:n_s].conj().T @ (blocks["Ks"] @ Phi_R[:n_s])
        Kt = Kt + 1j * float(eta_s) * Ks_t

    Ct = modal_damping_matrix(
        omegas, rayleigh=rayleigh, modal_zeta=modal_zeta,
    ).astype(complex)

    _refuse_frequency_dependent_impedance(blocks)

    Ca = blocks["Ca"]
    if sp.issparse(Ca) and Ca.nnz:
        Ct = Ct + Phi_L[n_s:].conj().T @ (Ca @ Phi_R[n_s:])

    return Kt, Ct, Mt


def solve_coupled_modal_frequency_sweep(
    basis,
    frequencies,
    *,
    F_s=None,
    F_a=None,
    eta_s=0.0,
    rayleigh=None,
    modal_zeta=None,
    verbose=False,
    return_modal=False,
):
    r"""
    Frequency response of the coupled problem on a reduced basis.

    Works with either basis — coupled modes or CMS. Solves the projected
    ``m x m`` system at every frequency and expands the result back onto
    the physical unknowns, ``q = Phi^R phi(omega)``.

    Parameters
    ----------
    basis : dict
        Output of :func:`build_coupled_modal_basis` or
        :func:`build_cms_basis`.
    frequencies : array-like (N_freq,)
        Frequencies [Hz].
    F_s : ndarray, optional
        Mechanical load on the free structural DOFs, ``(n_s,)`` or
        ``(n_s, N_freq)``.
    F_a : ndarray, optional
        Acoustic source on the retained fluid nodes, same shapes.
    eta_s, rayleigh, modal_zeta : optional
        Damping, see :func:`project_coupled_system`.
    verbose : bool, optional
        Print the progress of the sweep.
    return_modal : bool, optional
        Also return the modal participations.

    Returns
    -------
    result : dict
        ``w`` (n_s, N_freq), ``p`` (n_a, N_freq), ``idx_s``, ``idx_a``,
        ``frequencies`` — the same keys as the direct sweep, so the two
        can be swapped in post-processing — plus ``basis`` and, when
        ``return_modal``, ``participations`` (m, N_freq).

    Raises
    ------
    ValueError
        If no load is given (the response would be identically zero).

    Warns
    -----
    RuntimeWarning
        If the basis does not reach twice the highest frequency of the
        sweep, the usual truncation rule.

    Notes
    -----
    Truncation shows up as a missing quasi-static tail: the vectors left
    out still carry a static flexibility, which is why a point load —
    broad in space — converges more slowly than a smooth one, and why
    the driven point is the worst place to judge convergence.
    """
    if F_s is None and F_a is None:
        raise ValueError("No excitation: give F_s and/or F_a.")

    frequencies = np.atleast_1d(np.asarray(frequencies, dtype=float))
    n_f = frequencies.size
    n_s, n_a = basis["n_s"], basis["n_a"]
    Phi_R, Phi_L = basis["Phi_R"], basis["Phi_L"]
    omegas = basis["omegas"]

    f_top = frequencies.max()
    f_basis = omegas.max() / (2.0 * np.pi)
    if f_basis < 2.0 * f_top:
        warnings.warn(
            f"The basis reaches {f_basis:.1f} Hz, below the usual "
            f"2 x f_max = {2 * f_top:.1f} Hz rule; the response near the "
            "top of the sweep is likely truncated.",
            RuntimeWarning,
        )

    Kt, Ct, Mt = project_coupled_system(
        basis, eta_s=eta_s, rayleigh=rayleigh, modal_zeta=modal_zeta,
    )

    f_s = _as_sweep(F_s, n_s, n_f, "F_s")
    f_a = _as_sweep(F_a, n_a, n_f, "F_a")
    # Modal loads: the left modes are the ones that project the forces.
    Ft = Phi_L[:n_s].conj().T @ f_s + Phi_L[n_s:].conj().T @ f_a

    phi = np.zeros((omegas.size, n_f), dtype=complex)
    for i, freq in enumerate(frequencies):
        omega = 2.0 * np.pi * freq
        At = Kt + 1j * omega * Ct - omega ** 2 * Mt
        phi[:, i] = np.linalg.solve(At, Ft[:, i])
        if verbose:
            print(f"  [{i + 1}/{n_f}] f = {freq:8.2f} Hz  "
                  f"|phi|max = {np.abs(phi[:, i]).max():.3e}")

    q = Phi_R @ phi
    result = {
        "w": q[:n_s, :],
        "p": q[n_s:, :],
        "idx_s": basis["idx_s"],
        "idx_a": basis["idx_a"],
        "frequencies": frequencies,
        "basis": basis,
    }
    if return_modal:
        result["participations"] = phi
    return result


def reduced_model_error(basis, frequencies, reference, *, F_s=None, F_a=None,
                        eta_s=0.0, dof=None):
    """
    Solve a sweep on a reduced basis and measure it against the direct one.

    Parameters
    ----------
    basis : dict
        From :func:`build_coupled_modal_basis` or
        :func:`build_cms_basis`.
    frequencies : array-like (N_freq,)
        The frequencies of the reference sweep [Hz].
    reference : dict
        The direct solution to compare with, as returned by
        :func:`~pycafe.solver.solver_vibroacoustic.solve_vibroacoustic_frequency_sweep`
        — its ``"w"`` and ``"p"``.
    F_s, F_a : ndarray, optional
        The same loads the reference was solved with.
    eta_s : float, optional
        The same structural loss factor.
    dof : int, optional
        A reduced structural DOF to report the error at, typically the
        driven one, where truncation shows worst.

    Returns
    -------
    dict
        ``err_w``, ``err_p`` (relative 2-norms over the whole sweep),
        ``err_at_dof`` (median relative error there, or None),
        ``time`` [s] of the reduced sweep, and ``result``.
    """
    import time as _time

    started = _time.time()
    result = solve_coupled_modal_frequency_sweep(
        basis, frequencies, F_s=F_s, F_a=F_a, eta_s=eta_s,
    )
    elapsed = _time.time() - started

    err_w = (np.linalg.norm(result["w"] - reference["w"])
             / np.linalg.norm(reference["w"]))
    err_p = (np.linalg.norm(result["p"] - reference["p"])
             / np.linalg.norm(reference["p"]))
    at_dof = None
    if dof is not None:
        at_dof = float(np.median(
            np.abs(result["w"][dof] - reference["w"][dof])
            / np.abs(reference["w"][dof])
        ))
    return {"err_w": float(err_w), "err_p": float(err_p),
            "err_at_dof": at_dof, "time": elapsed, "result": result}
