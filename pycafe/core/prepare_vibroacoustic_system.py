"""
Preparation of the vibroacoustic system.

Builds the two domains of a conforming coupled mesh and, unless asked
otherwise, the coupling between them:

- acoustic domain (``fluid`` physical group): K_a, M_a on the fluid
  nodes, 1 pressure DOF/node;
- structural domain (``plate``/``structure`` group): K_s, M_s with
  6 DOF/node numbered over all mesh nodes, reduced to the active
  (plate) DOFs with the clamp boundary applied;
- interface data (shared faces and nodes);
- coupling blocks ``Kc`` and ``Mc = -rho0 Kc^T``, see
  :mod:`pycafe.build_matrices.coupling` for the sign convention.

The coupled system is then solved by
:mod:`pycafe.solver.solver_vibroacoustic`.
"""

import numpy as np

from pycafe.build_matrices.coupling import (
    acoustic_coupling_matrix,
    build_coupling_matrix,
    interface_area_vector,
)
from pycafe.build_matrices.domains import (
    identify_domains,
    build_KM_acoustic_domain,
    build_KM_structural_domain,
)

from pycafe.create_geom.conventions import CLAMP_GROUP_NAMES

DEFAULT_CLAMP_GROUPS = CLAMP_GROUP_NAMES

# Local DOF blocked on the support line, per support type. Local DOFs are
# [ux, uy, uz, rx, ry, rz]; a simple support holds the edge in place but
# lets the plate rotate about it.
SUPPORT_DOFS = {
    "clamped": (0, 1, 2, 3, 4, 5),
    "simply_supported": (0, 1, 2),
    "free": (),
}


def _blocked_dofs(support, clamp_dofs):
    """Local DOF indices blocked on the support, from name or explicit list."""
    if clamp_dofs is not None:
        blocked = np.unique(np.asarray(clamp_dofs, dtype=int))
        if blocked.size and (blocked.min() < 0 or blocked.max() > 5):
            raise ValueError("clamp_dofs must hold local DOF indices 0..5.")
        return blocked

    key = str(support).strip().lower()
    aliases = {"cccc": "clamped", "clamp": "clamped", "encastre": "clamped",
               "ssss": "simply_supported", "simple": "simply_supported",
               "pinned": "simply_supported"}
    key = aliases.get(key, key)
    if key not in SUPPORT_DOFS:
        raise ValueError(
            f"Unknown support '{support}'. Use one of "
            f"{sorted(SUPPORT_DOFS)}, or give 'clamp_dofs' explicitly."
        )
    return np.array(SUPPORT_DOFS[key], dtype=int)


def prepare_vibroacoustic_system(
    *,
    nodes,
    groups,
    rho0,
    c0,
    t,
    rho_s,
    E,
    nu,
    nsm=0.0,
    clamp_group=None,
    support="clamped",
    clamp_dofs=None,
    build_coupling=True,
    interface_sign=None,
    nonconforming="auto",
):
    """
    Assemble the two separate domains of a vibroacoustic problem.

    Parameters
    ----------
    nodes : ndarray (N, 3)
        Mesh node coordinates.
    groups : dict
        Physical groups with connectivity, from
        :func:`pycafe.create_geom.visualize_mesh.load_mesh_with_groups`.
    rho0, c0 : float
        Fluid density and speed of sound.
    t, rho_s, E, nu : float
        Shell thickness, density, Young's modulus, Poisson ratio.
    nsm : float, optional
        Non-structural mass per unit area of the shell.
    clamp_group : str, optional
        Name of the physical group holding the constrained structural
        nodes. Default: first match among
        ``("plate_clamp", "clamp", "fixed")``; no support if none found.
    support : {"clamped", "simply_supported", "free"}, optional
        What is blocked on the nodes of ``clamp_group``:

        - ``"clamped"`` (C-C-C-C): all 6 DOF, translations and
          rotations. The edge cannot move nor rotate.
        - ``"simply_supported"`` (S-S-S-S): the 3 translations only,
          the rotations stay free, so the plate can hinge on its edge.
        - ``"free"``: nothing, the group is ignored.
    clamp_dofs : sequence of int, optional
        Explicit list of the blocked local DOFs (0..5 =
        ``ux, uy, uz, rx, ry, rz``), overriding ``support``.
    build_coupling : bool, optional
        Assemble the coupling matrix over the interface faces. Set it
        to ``False`` to obtain the two uncoupled domains only.
    interface_sign : {+1, -1}, optional
        Force the orientation of the interface normals instead of
        deducing it from the fluid elements; see
        :func:`pycafe.build_matrices.coupling.interface_normals`.
    nonconforming : {"auto", False} or dict, optional
        What to do when the structural elements are not faces of the
        fluid elements. ``"auto"`` builds the coupling by the
        interpolation method of
        :mod:`pycafe.build_matrices.coupling_nonconforming` and says so
        in ``system["interface"]``; ``False`` refuses. A conforming
        interface is unaffected either way.

    Returns
    -------
    system : dict
        ``system["domains"]``    — output of :func:`identify_domains`;
        ``system["acoustic"]``   — ``K, M`` (n_nodes x n_nodes CSR),
        ``elem_type``, ``nodes0`` (0-based fluid node indices);
        ``system["structural"]`` — ``K, M`` full (6N x 6N),
        ``K_red, M_red`` on the free DOFs, ``idx_free`` (global DOF
        indices of the reduced system), ``nodes0`` (plate nodes,
        0-based), ``clamped_nodes0``, ``elem_type``;
        ``system["interface"]``  — ``conn0`` (plate faces, 0-based,
        shared with the fluid hexa faces), ``nodes0``, ``normals``
        (unit normals out of the fluid, one per face);
        ``system["coupling"]``   — ``{"Kc", "Mc", "area_vector"}``, or
        ``None`` when ``build_coupling=False``. ``Kc`` has shape
        ``(6*N, N)``: structural DOFs numbered over all mesh nodes by
        acoustic pressures, to be reduced with the index sets of the
        two domains (done by
        :func:`pycafe.solver.solver_vibroacoustic.build_coupled_blocks`).
        ``system["props"]``      — material/fluid properties used.

    See Also
    --------
    pycafe.build_matrices.coupling : Coupling matrix and sign convention.
    pycafe.solver.solver_vibroacoustic : Coupled solvers.
    """
    domains = identify_domains(groups)
    if "structure" not in domains:
        raise RuntimeError(
            "Vibroacoustic preparation requires a structural domain "
            "(physical group named 'plate', 'structure' or 'shell')."
        )

    n = nodes.shape[0]

    # Acoustics
    K_a, M_a, _, elem_a = build_KM_acoustic_domain(nodes, domains, c0)
    fluid_nodes0 = np.asarray(domains["fluid"]["nodes"], dtype=int) - 1
    # Structural
    K_s, M_s, _, elem_s = build_KM_structural_domain(
        nodes, domains, t, rho_s, E, nu, nsm
    )
    plate_nodes0 = np.asarray(domains["structure"]["nodes"], dtype=int) - 1

    # clamped
    if clamp_group is None:
        clamp_group = next(
            (g for g in groups if g.lower() in DEFAULT_CLAMP_GROUPS), None
        )
    if clamp_group is not None and clamp_group in groups:
        clamped_nodes0 = (
            np.asarray(groups[clamp_group]["nodes"], dtype=int) - 1
        )
    else:
        clamped_nodes0 = np.array([], dtype=int)

    blocked = _blocked_dofs(support, clamp_dofs)
    all_dofs = (plate_nodes0[:, None] * 6 + np.arange(6)).ravel()
    constrained = (clamped_nodes0[:, None] * 6 + blocked).ravel()
    idx_free_s = np.setdiff1d(all_dofs, constrained)

    K_s_red = K_s[np.ix_(idx_free_s, idx_free_s)]
    M_s_red = M_s[np.ix_(idx_free_s, idx_free_s)]

    # Interface
    
    iface_conn0 = domains["structure"]["conn0"]
    fluid_conn0 = domains["fluid"]["conn0"]

    interface = {
        "conn0": iface_conn0,
        "nodes0": plate_nodes0,
        "normals": None,
    }

    # Coupling 
    
    coupling = None
    if build_coupling:
        from pycafe.build_matrices.coupling import interface_normals

        report = {}
        Kc = build_coupling_matrix(
            nodes, iface_conn0, fluid_conn0,
            sign=interface_sign,
            num_nodes=n,
            dofs_per_node=domains["structure"]["spec"].dofs_per_node,
            nonconforming=nonconforming,
            report=report,
        )
        # On a non-conforming interface the normals come from the fluid
        # boundary the structure faces, not from the element behind it,
        # and the fallback has already resolved them.
        interface["conforming"] = report.get("conforming", True)
        interface["normals"] = report.get("normals")
        if interface["normals"] is None:
            interface["normals"] = interface_normals(
                nodes, iface_conn0, fluid_conn0, sign=interface_sign
            )
        coupling = {
            "Kc": Kc,
            "Mc": acoustic_coupling_matrix(Kc, rho0),
            "area_vector": interface_area_vector(Kc),
        }
        if not interface["conforming"]:
            coupling["nonconforming"] = {
                k: v for k, v in report.items() if k != "normals"
            }

    system = {
        "domains": domains,
        "acoustic": {
            "K": K_a,
            "M": M_a,
            "elem_type": elem_a,
            "nodes0": fluid_nodes0,
        },
        "structural": {
            "K": K_s,
            "M": M_s,
            "K_red": K_s_red,
            "M_red": M_s_red,
            "idx_free": idx_free_s,
            "nodes0": plate_nodes0,
            "clamped_nodes0": clamped_nodes0,
            "blocked_dofs": blocked,
            "support": support if clamp_dofs is None else "custom",
            "elem_type": elem_s,
        },
        "interface": interface,
        "coupling": coupling,
        "props": {
            "rho0": rho0, "c0": c0,
            "t": t, "rho_s": rho_s, "E": E, "nu": nu, "nsm": nsm,
        },
    }
    return system
