r"""
Fully coupled vibroacoustic solvers, Eulerian :math:`(\mathbf w,
\mathbf p)` formulation.

The unknowns of the two domains are stacked in a single vector
:math:`\mathbf x = \{\mathbf w;\ \mathbf p\}`, with :math:`\mathbf w`
the free structural DOFs (6 per node, the support already removed) and
:math:`\mathbf p` the acoustic pressures of the fluid nodes. The
coupled dynamic stiffness is the unsymmetric block matrix

.. math::

   \mathbf A(\omega) =
   \begin{bmatrix}
   \mathbf K_s (1 + j\eta_s) - \omega^2 \mathbf M_s & -\mathbf K_c \\
   -\rho_0 \omega^2 \mathbf K_c^T &
       \mathbf K_a + j\omega \mathbf C_a - \omega^2 \mathbf M_a
   \end{bmatrix}

built from the coupling matrix of
:mod:`pycafe.build_matrices.coupling`, whose module documentation
carries the sign convention. Both interactions are present at once --
the pressure loads the structure, the structural acceleration drives
the fluid -- so the problem is genuinely two-way coupled, not a
sequential one-way chain.

Examples
--------
Coupled natural frequencies and a forced response of a cavity closed by
a plate::

    system = prepare_vibroacoustic_system(
        nodes=nodes, groups=groups, rho0=1.204, c0=343.0,
        t=0.002, rho_s=7800.0, E=210e9, nu=0.3,
    )

    freqs, modes = solve_vibroacoustic_modal(system, num_modes=6)

    blocks = build_coupled_blocks(system, eta_s=0.02)
    F_s = np.zeros(blocks["n_s"])
    F_s[dof_of_interest] = 1.0
    result = solve_vibroacoustic_frequency_sweep(
        system, np.arange(20.0, 120.0, 1.0), F_s=F_s, blocks=blocks,
    )
    p_full = expand_pressure(result["p"], result["idx_a"], len(nodes))

Notes
-----
A sealed cavity has no pressure constraint, so the coupled
eigenproblem carries one zero eigenvalue: the uniform-pressure mode of
the fluid, the acoustic counterpart of a rigid-body motion. It is not a
floating structure -- a clamped plate has no rigid-body motion at all,
and in that mode it merely sits at its static deflection under the
uniform pressure. Pin one pressure DOF through ``pressure_zero_nodes0``
to remove it.

See Also
--------
pycafe.build_matrices.coupling : Coupling matrix and sign convention.
pycafe.core.prepare_vibroacoustic_system : Builds the coupled system.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from pycafe.build_matrices.coupling import build_coupling_matrix


def build_coupled_blocks(
    system,
    *,
    pressure_zero_nodes0=None,
    C_a=None,
    eta_s=0.0,
):
    """
    Reduce the two domains and the coupling to the coupled unknowns.

    Parameters
    ----------
    system : dict
        Output of
        :func:`pycafe.core.prepare_vibroacoustic_system.prepare_vibroacoustic_system`,
        with its ``coupling`` entry already filled.
    pressure_zero_nodes0 : array-like of int, optional
        Fluid nodes where ``p = 0`` is imposed; removed from the
        acoustic unknowns.
    C_a : sparse matrix, optional
        Acoustic damping (impedance) matrix on the **full** node
        numbering, e.g. from an anechoic or lined boundary.
    eta_s : float, optional
        Structural loss factor: the stiffness becomes ``Ks (1 + j eta)``.

    Returns
    -------
    blocks : dict
        ``Ks, Ms, Ka, Ma, Ca, Kc`` reduced to the coupled unknowns,
        plus ``idx_s`` (global structural DOF indices), ``idx_a``
        (0-based fluid node indices), ``n_s``, ``n_a`` and ``rho0``.

    Raises
    ------
    RuntimeError
        If the coupling matrix has not been built.
    """
    if system.get("coupling") is None:
        raise RuntimeError(
            "system['coupling'] is empty: build it with "
            "prepare_vibroacoustic_system(..., build_coupling=True) or "
            "pycafe.build_matrices.coupling.build_coupling_matrix."
        )

    idx_s = np.asarray(system["structural"]["idx_free"], dtype=int)
    idx_a = np.asarray(system["acoustic"]["nodes0"], dtype=int)

    if pressure_zero_nodes0 is not None and len(pressure_zero_nodes0):
        idx_a = np.setdiff1d(idx_a, np.asarray(pressure_zero_nodes0, dtype=int))

    Kc = system["coupling"]["Kc"]
    rho0 = float(system["props"]["rho0"])

    Ks = system["structural"]["K"][np.ix_(idx_s, idx_s)]
    Ms = system["structural"]["M"][np.ix_(idx_s, idx_s)]
    Ka = system["acoustic"]["K"][np.ix_(idx_a, idx_a)]
    Ma = system["acoustic"]["M"][np.ix_(idx_a, idx_a)]
    Ca = (C_a[np.ix_(idx_a, idx_a)] if C_a is not None
          else sp.csr_matrix((idx_a.size, idx_a.size)))

    if eta_s:
        Ks = Ks.astype(complex) * (1.0 + 1j * float(eta_s))

    return {
        "Ks": Ks.tocsr(),
        "Ms": Ms.tocsr(),
        "Ka": Ka.tocsr(),
        "Ma": Ma.tocsr(),
        "Ca": Ca.tocsr(),
        "Kc": Kc[np.ix_(idx_s, idx_a)].tocsr(),
        "idx_s": idx_s,
        "idx_a": idx_a,
        "n_s": idx_s.size,
        "n_a": idx_a.size,
        "rho0": rho0,
    }


def coupled_dynamic_stiffness(blocks, omega):
    """
    Assemble ``A(omega)`` of the coupled system at one frequency.

    Parameters
    ----------
    blocks : dict
        Output of :func:`build_coupled_blocks`.
    omega : float
        Angular frequency [rad/s].

    Returns
    -------
    A : scipy.sparse.csr_matrix of complex, shape (n_s + n_a, n_s + n_a)
    """
    w2 = omega ** 2
    A_ss = blocks["Ks"].astype(complex) - w2 * blocks["Ms"]
    A_aa = (blocks["Ka"].astype(complex)
            + 1j * omega * blocks["Ca"]
            - w2 * blocks["Ma"])

    # Off-diagonal blocks: pressure -> structural force, and structural
    # acceleration -> acoustic source. They are transposes of each other
    # up to -rho0 omega^2, never equal: that is the asymmetry of the
    # (w, p) formulation.
    A_sa = -blocks["Kc"].astype(complex)
    A_as = -blocks["rho0"] * w2 * blocks["Kc"].T.astype(complex)

    return sp.bmat([[A_ss, A_sa], [A_as, A_aa]], format="csr")


def solve_vibroacoustic_frequency_sweep(
    system,
    frequencies,
    *,
    F_s=None,
    F_a=None,
    pressure_zero_nodes0=None,
    C_a=None,
    eta_s=0.0,
    blocks=None,
    verbose=True,
):
    """
    Direct frequency response of the fully coupled problem.

    Solves ``A(omega) x = f`` at every frequency of the sweep, with the
    structure and the fluid resolved simultaneously.

    Parameters
    ----------
    system : dict
        Prepared vibroacoustic system, coupling included.
    frequencies : array-like (N_freq,)
        Frequencies [Hz].
    F_s : ndarray, optional
        Mechanical load on the free structural DOFs. Either
        ``(n_s,)``, applied at every frequency, or ``(n_s, N_freq)``.
    F_a : ndarray, optional
        Acoustic source on the retained fluid nodes, same shapes.
    pressure_zero_nodes0, C_a, eta_s : optional
        Passed to :func:`build_coupled_blocks`.
    blocks : dict, optional
        Pre-reduced blocks, to skip the reduction on repeated sweeps.
    verbose : bool, optional
        Print the progress of the sweep.

    Returns
    -------
    result : dict
        ``w`` (n_s, N_freq) complex structural displacements,
        ``p`` (n_a, N_freq) complex nodal pressures,
        ``idx_s``, ``idx_a`` index sets, ``frequencies``.

    Notes
    -----
    The matrix is complex and unsymmetric, so each frequency needs its
    own factorization: the cost grows linearly with the number of
    frequencies.
    """
    blocks = blocks or build_coupled_blocks(
        system,
        pressure_zero_nodes0=pressure_zero_nodes0,
        C_a=C_a,
        eta_s=eta_s,
    )
    n_s, n_a = blocks["n_s"], blocks["n_a"]
    frequencies = np.atleast_1d(np.asarray(frequencies, dtype=float))
    n_f = frequencies.size

    f_s = _as_sweep(F_s, n_s, n_f, "F_s")
    f_a = _as_sweep(F_a, n_a, n_f, "F_a")

    x = np.zeros((n_s + n_a, n_f), dtype=complex)
    for i, freq in enumerate(frequencies):
        A = coupled_dynamic_stiffness(blocks, 2.0 * np.pi * freq)
        rhs = np.concatenate([f_s[:, i], f_a[:, i]])
        x[:, i] = spsolve(A.tocsc(), rhs)
        if verbose:
            print(f"  [{i + 1}/{n_f}] f = {freq:8.2f} Hz  "
                  f"|w|max = {np.abs(x[:n_s, i]).max():.3e}  "
                  f"|p|max = {np.abs(x[n_s:, i]).max():.3e}")

    return {
        "w": x[:n_s, :],
        "p": x[n_s:, :],
        "idx_s": blocks["idx_s"],
        "idx_a": blocks["idx_a"],
        "frequencies": frequencies,
        "blocks": blocks,
    }


def _as_sweep(F, n, n_f, name):
    """Broadcast a load to (n, n_f), accepting None, (n,) or (n, n_f)."""
    if F is None:
        return np.zeros((n, n_f), dtype=complex)
    F = np.asarray(F, dtype=complex)
    if F.ndim == 1:
        if F.size != n:
            raise ValueError(f"{name} has size {F.size}, expected {n}.")
        return np.repeat(F[:, None], n_f, axis=1)
    if F.shape != (n, n_f):
        raise ValueError(
            f"{name} has shape {F.shape}, expected ({n},) or ({n}, {n_f})."
        )
    return F


def solve_vibroacoustic_modal(
    system,
    num_modes=6,
    *,
    pressure_zero_nodes0=None,
    eta_s=0.0,
    blocks=None,
    sigma=1.0,
):
    """
    Coupled natural frequencies and mode shapes.

    The undamped coupled problem is the generalized eigenproblem
    ``K x = omega^2 M x`` with

        K = [[Ks, -Kc],  M = [[Ms,          0 ],
             [0 ,  Ka]]       [rho0 Kc^T,  Ma ]]

    which is **unsymmetric**: the fluid acts on the structure through
    the pressure (a stiffness-block term) and the structure acts on the
    fluid through its acceleration (a mass-block term). The eigenvalues
    stay real and positive for a lossless problem; a small imaginary
    residue from the unsymmetric solver is discarded.

    Parameters
    ----------
    system : dict
        Prepared vibroacoustic system, coupling included.
    num_modes : int, optional
        Number of modes returned, in increasing frequency.
    pressure_zero_nodes0, eta_s, blocks : optional
        See :func:`build_coupled_blocks`.
    sigma : float, optional
        Shift of the shift-invert, in ``omega^2``; the default targets
        the lowest modes.

    Returns
    -------
    freqs : ndarray (num_modes,)
        Coupled natural frequencies [Hz].
    modes : ndarray (n_s + n_a, num_modes)
        Coupled mode shapes, structural DOFs stacked over pressures.

    Notes
    -----
    A zero eigenvalue exists whenever the fluid is fully enclosed by
    rigid and flexible walls without any pressure constraint: it is the
    uniform-pressure mode of a closed cavity, the acoustic counterpart
    of a rigid-body mode. It is returned like the others; pin one
    pressure DOF through ``pressure_zero_nodes0`` to remove it.

    The shift-invert is applied by hand rather than through the ``M``
    argument of ARPACK, which assumes a symmetric positive definite
    mass matrix -- an assumption the coupled ``M`` violates, and which
    silently returns wrong eigenvalues when ignored.
    """
    from scipy.sparse.linalg import LinearOperator, eigs, splu

    blocks = blocks or build_coupled_blocks(
        system, pressure_zero_nodes0=pressure_zero_nodes0, eta_s=eta_s,
    )
    K, M = coupled_modal_matrices(blocks)

    n_total = K.shape[0]
    k = min(int(num_modes), n_total - 2)

    # Standard shift-invert: eigenvalues mu of (K - sigma M)^-1 M map
    # back as omega^2 = sigma + 1/mu.
    lu = splu((K - sigma * M).tocsc())
    OP = LinearOperator(
        (n_total, n_total),
        matvec=lambda v: lu.solve(M @ v),
        dtype=float,
    )
    mu, vecs = eigs(OP, k=k, which="LM")

    omega2 = np.real(sigma + 1.0 / mu)
    order = np.argsort(omega2)
    omega2 = np.clip(omega2[order], 0.0, None)

    freqs = np.sqrt(omega2) / (2.0 * np.pi)
    modes = np.real_if_close(vecs[:, order])
    return freqs, modes


def coupled_modal_matrices(blocks):
    """
    Stiffness and mass of the undamped coupled eigenproblem.

    Returns the pair ``(K, M)`` of ``K x = omega^2 M x`` described in
    :func:`solve_vibroacoustic_modal`: the fluid-to-structure coupling
    sits in ``K``, the structure-to-fluid one in ``M``.

    Returns
    -------
    K, M : scipy.sparse.csr_matrix, shape (n_s + n_a, n_s + n_a)
    """
    n_s, n_a = blocks["n_s"], blocks["n_a"]
    rho0, Kc = blocks["rho0"], blocks["Kc"]

    zero_as = sp.csr_matrix((n_a, n_s))
    zero_sa = sp.csr_matrix((n_s, n_a))

    K = sp.bmat([[blocks["Ks"], -Kc],
                 [zero_as, blocks["Ka"]]], format="csr")
    M = sp.bmat([[blocks["Ms"], zero_sa],
                 [rho0 * Kc.T, blocks["Ma"]]], format="csr")
    return K, M


def split_solution(x, blocks):
    """
    Split a stacked coupled vector into its structural and acoustic parts.

    Returns
    -------
    w, p : ndarray
        Structural DOFs and nodal pressures of the reduced unknowns.
    """
    n_s = blocks["n_s"]
    return x[:n_s], x[n_s:]


def expand_pressure(p_red, idx_a, num_nodes):
    """
    Scatter the reduced pressures back onto the full mesh numbering.

    Nodes outside the fluid, and constrained ones, are left at zero, so
    the result can be handed straight to the 3D post-processing.

    Parameters
    ----------
    p_red : ndarray (n_a,) or (n_a, N_freq)
    idx_a : ndarray (n_a,)
        0-based node indices of the retained fluid nodes.
    num_nodes : int
        Total number of mesh nodes.

    Returns
    -------
    p_full : ndarray (num_nodes,) or (num_nodes, N_freq)
    """
    p_red = np.asarray(p_red)
    shape = ((num_nodes,) if p_red.ndim == 1
             else (num_nodes, p_red.shape[1]))
    p_full = np.zeros(shape, dtype=complex)
    p_full[idx_a] = p_red
    return p_full


def structural_displacement_field(w_red, idx_s, num_nodes, dofs_per_node=6):
    """
    Scatter reduced structural DOFs onto ``(num_nodes, dofs_per_node)``.

    Clamped and non-structural nodes stay at zero. Column 2 is the
    transverse translation ``uz``, the one usually plotted.

    Parameters
    ----------
    w_red : ndarray (n_s,)
    idx_s : ndarray (n_s,)
        Global structural DOF indices of the reduced system.
    num_nodes : int
    dofs_per_node : int, optional

    Returns
    -------
    w_full : ndarray (num_nodes, dofs_per_node) complex
    """
    w_full = np.zeros(num_nodes * dofs_per_node, dtype=complex)
    w_full[np.asarray(idx_s, dtype=int)] = np.asarray(w_red)
    return w_full.reshape(num_nodes, dofs_per_node)
