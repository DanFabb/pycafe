# assembly.py
#
# Generic sparse assembly of global FEM matrices. Works for any element
# type registered in element_registry: the element kernel computes the
# local matrices, this module scatters them into global COO triplets.
import numpy as np
from scipy.sparse import coo_matrix


def assemble_KM(
    nodes,
    conn0,
    kernel,
    c0=None,
    debug=False,
    debug_elem_id=0,
    store_all_elements=False,
    dofs_per_node=1,
    kernel_args=None,
):
    """
    Assemble global stiffness and mass matrices from an element kernel.

    Element-local matrices are computed by ``kernel`` and accumulated
    into preallocated COO triplet arrays; the global matrices are built
    in a single ``coo_matrix`` call (duplicate entries are summed) and
    returned in CSR format. This replaces the per-entry insertion into
    ``lil_matrix`` used by the previous per-element-type assembly
    routines and scales to large (3D) meshes.

    Parameters
    ----------
    nodes : ndarray of shape (N, 2) or (N, 3)
        Coordinates of the mesh nodes.
    conn0 : ndarray of int, shape (n_elem, n_nodes_per_elem)
        Element connectivity, 0-based.
    kernel : callable
        ``kernel(x_e, *kernel_args) -> (K_e, M_e)`` computing the
        element matrices from the element node coordinates
        ``x_e = nodes[conn0[e]]``. For acoustic elements
        ``kernel_args`` defaults to ``(c0,)``.
    c0 : float, optional
        Speed of sound in the fluid (acoustic kernels). Ignored when
        ``kernel_args`` is given explicitly.
    debug : bool, optional
        If True, store element-level data for ``debug_elem_id``.
    debug_elem_id : int, optional
        Element index for which debug data is collected.
    store_all_elements : bool, optional
        If True, store element-level matrices for all elements
        under ``debug_data["all_elements"]``.
    dofs_per_node : int, optional
        Degrees of freedom per node (1 for acoustic pressure, 6 for
        shell elements). Global DOF of node ``n``, component ``d`` is
        ``n * dofs_per_node + d``; element matrices are
        ``(n_en * dofs_per_node)`` square with node-major DOF order.
    kernel_args : tuple, optional
        Extra arguments passed to ``kernel`` after ``x_e``.
        Defaults to ``(c0,)``.

    Returns
    -------
    K_global : scipy.sparse.csr_matrix
        Global stiffness matrix.
    M_global : scipy.sparse.csr_matrix
        Global mass matrix.
    debug_data : dict or None
        Element-level debug information, or None.
    """
    conn0 = np.asarray(conn0, dtype=int)
    n_elem, n_en = conn0.shape
    if kernel_args is None:
        kernel_args = (c0,)
    n_ed = n_en * dofs_per_node          # DOF per elemento
    num_dofs = nodes.shape[0] * dofs_per_node
    block = n_ed * n_ed

    rows = np.empty(n_elem * block, dtype=np.int64)
    cols = np.empty(n_elem * block, dtype=np.int64)
    k_vals = np.empty(n_elem * block, dtype=float)
    m_vals = np.empty(n_elem * block, dtype=float)

    debug_data = None
    all_elements = []

    for ie, elem in enumerate(conn0):
        coords = nodes[elem]
        K_e, M_e = kernel(coords, *kernel_args)

        if dofs_per_node == 1:
            edofs = elem
        else:
            edofs = (elem[:, None] * dofs_per_node
                     + np.arange(dofs_per_node)).ravel()

        s = ie * block
        rows[s:s + block] = np.repeat(edofs, n_ed)
        cols[s:s + block] = np.tile(edofs, n_ed)
        k_vals[s:s + block] = K_e.ravel()
        m_vals[s:s + block] = M_e.ravel()

        if store_all_elements:
            all_elements.append({
                "elem_id": ie,
                "nodes": elem.copy(),
                "coords": coords.copy(),
                "K_e": K_e.copy(),
                "M_e": M_e.copy(),
            })

        if debug and ie == debug_elem_id:
            debug_data = {
                "elem_id": ie,
                "nodes": elem.copy(),
                "coords": coords.copy(),
                "K_e": K_e.copy(),
                "M_e": M_e.copy(),
            }

    shape = (num_dofs, num_dofs)
    K_global = coo_matrix((k_vals, (rows, cols)), shape=shape).tocsr()
    M_global = coo_matrix((m_vals, (rows, cols)), shape=shape).tocsr()

    if store_all_elements:
        debug_data = debug_data or {}
        debug_data["all_elements"] = all_elements

    return K_global, M_global, debug_data
