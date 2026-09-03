# domains.py
#
# Assembly of the structural domain of a coupled mesh.
#
# The fluid side of this — which physical group is which, and the
# acoustic assembly over it — is `pycafe.build_matrices.domains`, which
# an acoustic-only install has on its own. Only the shell assembly,
# which needs the plate properties and the CQUAD4F kernel, lives here.
from pycafe.build_matrices.assembly import assemble_KM
from pycafe.create_geom.conventions import STRUCTURE_GROUP_NAMES


def build_KM_structural_domain(nodes, domains, t, rho, E, nu, nsm=0.0,
                               **kwargs):
    """
    Assemble the structural K, M from the structure domain.

    Global structural DOFs are numbered over ALL mesh nodes
    (``6 * num_nodes``): nodes not belonging to the structure carry
    zero rows/columns. This keeps the node numbering shared with the
    fluid — which is what the vibroacoustic coupling matrix needs on
    a conforming mesh — and the empty DOFs are removed later by the
    solver-side reduction.

    Returns
    -------
    K, M : scipy.sparse.csr_matrix, shape (6*num_nodes, 6*num_nodes)
    debug_data : dict or None
    elem_type : str
    """
    if "structure" not in domains:
        raise RuntimeError(
            "No structural domain found: expected a physical group "
            f"named one of {STRUCTURE_GROUP_NAMES}."
        )
    dom = domains["structure"]
    kernel = dom["spec"].kernel
    if kernel is None:
        raise NotImplementedError(
            f"Element type '{dom['elem_type']}' has no kernel."
        )
    K, M, debug_data = assemble_KM(
        nodes, dom["conn0"], kernel,
        dofs_per_node=dom["spec"].dofs_per_node,
        kernel_args=(t, rho, E, nu, nsm),
        **kwargs,
    )
    return K, M, debug_data, dom["elem_type"]
