import numpy as np
import pycafe


def test_import():
    assert pycafe is not None


def test_version():
    assert hasattr(pycafe, "__version__")
    assert isinstance(pycafe.__version__, str)


def test_basic_acoustic_workflow():
    """
    Smoke test for a minimal acoustic FEM workflow.
    """

    # Fluid
    fluid = pycafe.load_fluid("air")

    # Nodes
    nodes = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ])

    # 🔥 IMPORTANT: element group name MUST match pyCAFE expectations
    elements = {
        "Quadrilateral 4": np.array([
            [0, 1, 2, 3]
        ])
    }

    boundaries = {}

    # Boundary conditions (non-interactive)
    bc = pycafe.assign_boundary_conditions(
        boundaries=boundaries,
        nodes=nodes,
        interactive=False,
    )

    system = pycafe.prepare_acoustic_system(
        nodes=nodes,
        elements=elements,
        boundaries=boundaries,
        rho=fluid["rho"],
        c0=fluid["c0"],
        bc=bc,
    )

    results = pycafe.run_analysis(
        system=system,
        nodes=nodes,
        boundaries=boundaries,
        rho=fluid["rho"],
        interactive=False,
    )

    assert isinstance(results, dict)
    assert results["type"] == "direct"
