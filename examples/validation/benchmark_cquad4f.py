"""
CQUAD4F benchmark: single-element response and plate convergence.

Two single-element test cases, both taken from the MSC.Nastran reference
model whose stiffness matrices are in ``examples/data/*.pch``:

    single CQUAD4, t = 0.1 mm, E = 200000 MPa, nu = 0.3
    nodes 1-2-3 clamped (123456), FORCE Fz = 1 N on node 4

    "square": (0,0) (2,0) (2,2) (0,2)
    "skew"  : (0,0) (2,0) (1.5,2) (0,2)

followed by a mesh-convergence check of a simply supported square plate
against the analytical Kirchhoff solution.

For the full validation, including the term-by-term comparison of the
element stiffness matrix, see ``examples/validation/cquad4_plate_validation.ipynb``.

Usage:  python -m examples.validation.benchmark_cquad4f
"""

from functools import partial

import numpy as np
from scipy.sparse.linalg import eigsh

from pycafe.build_matrices.assembly import assemble_KM
from pycafe_vibro.element_cquad4f import element_matrices_cquad4f

E, NU, T, RHO = 200e3, 0.3, 0.1, 7.8e-9

# MSC.Nastran .f06, displacement and rotations of the loaded node
CASES = {
    "square": dict(
        x=np.array([[0., 0., 0.], [2., 0., 0.], [2., 2., 0.], [0., 2., 0.]]),
        nastran=(0.050742708, 0.047409551, 0.047409551)),
    "skew": dict(
        x=np.array([[0., 0., 0.], [2., 0., 0.], [1.5, 2., 0.], [0., 2., 0.]]),
        nastran=(0.03734316, 0.03541586, 0.04646765)),
}


def corner_response(K, node=3):
    """T3, R1, R2 of the loaded node, the other three being clamped."""
    dof = np.arange(6 * node, 6 * node + 6)
    f = np.zeros(6)
    f[2] = 1.0
    u = np.linalg.solve(K[np.ix_(dof, dof)], f)
    return np.array([u[2], u[3], u[4]])


def single_element_benchmark():
    print("=" * 92)
    print("SINGLE CQUAD4F — displacement and rotations of the loaded node "
          "(deviation % from Nastran)")
    print("=" * 92)
    for name, case in CASES.items():
        ref = np.array(case["nastran"])
        print(f"\n### {name} element")
        print(f"  {'':22s} {'T3':>12s} {'R1':>12s} {'R2':>12s}"
              f" | {'err T3':>7s} {'err R1':>7s} {'err R2':>7s} | {'mean':>7s}")
        print(f"  {'Nastran (reference)':22s} {ref[0]:12.6g} {ref[1]:12.6g}"
              f" {ref[2]:12.6g} |")
        for eps in (0.0227, 0.0336, 0.04, 0.1):
            K, _ = element_matrices_cquad4f(case["x"], T, RHO, E, NU,
                                            epsilon=eps)
            u = corner_response(K)
            err = np.abs(u - ref) / np.abs(ref)
            print(f"  {'CQUAD4F eps=' + str(eps):22s} {u[0]:12.6g} {u[1]:12.6g}"
                  f" {u[2]:12.6g} | {err[0]*100:7.2f} {err[1]*100:7.2f}"
                  f" {err[2]*100:7.2f} | {err.mean()*100:7.2f}")


# ------------------------------------------------- simply supported plate
def plate_mesh(L, n):
    v = np.linspace(0.0, L, n + 1)
    X, Y = np.meshgrid(v, v, indexing="ij")
    nodes = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])
    conn = [[i * (n + 1) + j, (i + 1) * (n + 1) + j,
             (i + 1) * (n + 1) + j + 1, i * (n + 1) + j + 1]
            for i in range(n) for j in range(n)]
    return nodes, np.array(conn, dtype=int)


def plate_frequency(n, L=1.0, t=0.01, rho=7800.0, e=210e9, nu=0.3, **kw):
    nodes, conn = plate_mesh(L, n)
    kernel = element_matrices_cquad4f
    if kw:
        kernel = partial(kernel, **kw)
    K, M, _ = assemble_KM(nodes, conn, kernel, dofs_per_node=6,
                          kernel_args=(t, rho, e, nu))
    nn = nodes.shape[0]
    fixed = set()
    for k in range(nn):
        xk, yk = nodes[k, :2]
        if (np.isclose(xk, 0) or np.isclose(xk, L)
                or np.isclose(yk, 0) or np.isclose(yk, L)):
            fixed.update(6 * k + d for d in (0, 1, 2))
    free = np.array(sorted(set(range(6 * nn)) - fixed), dtype=int)
    vals, _ = eigsh(K[np.ix_(free, free)].tocsc(), k=1,
                    M=M[np.ix_(free, free)].tocsc(), sigma=0.0, which="LM")
    return np.sqrt(np.abs(vals[0])) / (2 * np.pi)


def plate_benchmark():
    L, t, rho, e, nu = 1.0, 0.01, 7800.0, 210e9, 0.3
    D = e * t**3 / (12 * (1 - nu**2))
    f11 = np.pi * np.sqrt(D / (rho * t)) / L**2
    print("\n" + "=" * 92)
    print(f"SIMPLY SUPPORTED SQUARE PLATE — Kirchhoff f11 = {f11:.4f} Hz")
    print("=" * 92)
    print(f"  {'mesh':>7s} {'f11 [Hz]':>12s} {'error %':>10s}")
    for n in (4, 8, 12, 16):
        f = plate_frequency(n, L, t, rho, e, nu)
        print(f"  {n:3d}x{n:<3d} {f:12.4f} {abs(f - f11) / f11 * 100:10.3f}")


if __name__ == "__main__":
    single_element_benchmark()
    plate_benchmark()
