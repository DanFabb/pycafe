def load_mesh(
    msh_file,
    *,
    show_info=True,
    show_plot=True,
):
    """
    Load a mesh and optionally visualize it.
    """
    from pycafe.create_geom.visualize_mesh import (
        load_mesh_and_elements,
        plot_2d_mesh,
        print_available_boundaries,
    )

    nodes, elements, boundaries = load_mesh_and_elements(msh_file)

    if show_info:
        print_available_boundaries(boundaries)

    if show_plot:
        plot_2d_mesh(nodes, elements)

    return nodes, elements, boundaries
