"""
Faces of the fluid mesh.

Which element owns which face is a property of the acoustic mesh
alone, so it is answered here: the mesh report uses it to check an
interface, and :mod:`pycafe_vibro.coupling` uses it to orient one.
"""

import numpy as np


_VOLUME_FACES = {
    8: [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
        (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)],   # Hexahedron 8
    4: [(0, 1, 3), (0, 2, 1), (0, 3, 2), (1, 2, 3)],  # Tetrahedron 4
}


def _fluid_face_owners(fluid_conn0):
    """
    Map each face of the fluid mesh to the elements it belongs to.

    Returns
    -------
    owners : dict
        ``{sorted tuple of face nodes: [element indices]}``. A face on
        the boundary of the fluid has exactly one owner, which is what
        fixes the outward direction; an interior face has two, one on
        each side, and no outward normal can be deduced from it.
    """
    conn = np.asarray(fluid_conn0, dtype=int)
    faces = _VOLUME_FACES.get(conn.shape[1])
    if faces is None:
        raise ValueError(
            f"Fluid elements with {conn.shape[1]} nodes are not supported "
            "for the interface orientation; expected 8 (hexa) or 4 (tetra)."
        )

    owners = {}
    for e, row in enumerate(conn):
        for face in faces:
            owners.setdefault(
                tuple(sorted(row[list(face)].tolist())), []
            ).append(e)
    return owners
