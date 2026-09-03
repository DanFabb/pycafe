r"""
The MAC matrix as a picture: what the pairing is worth.

A table that pairs every computed mode with an analytical one says
nothing about how *safely* it did so — a mode that correlated 0.6 with
its label and 0.5 with another would be read exactly the same way. The
whole matrix does say it, and it is worth looking at precisely because
the diagonal is not the interesting part: a diagonal of ones is what the
pairing was chosen to produce, and everything off it is the answer.

.. code-block:: python

    paired = identify_cavity_modes(nodes, modes, freqs, c0, sides)
    plot_mac_matrix(paired["mac"], paired["labels"], freqs)
"""

import numpy as np

__all__ = ["plot_mac_matrix"]


def plot_mac_matrix(matrix, labels, frequencies=None, *, ax=None,
                    cmap="Blues", gamma=0.45, show=False,
                    xlabel="analytical mode $(l,m,n)$", ylabel="pyCAFE mode",
                    title=None):
    """
    Draw a MAC matrix, scaled so that the small numbers are readable.

    The colour scale is stretched near zero: a linear one leaves every
    off-diagonal cell the same shade of white, which is the half of the
    picture worth reading.

    Parameters
    ----------
    matrix : ndarray (k, k)
        MAC between every computed mode (rows) and every label
        (columns), as ``identify_cavity_modes`` returns it.
    labels : sequence
        What each column is, printed under it — ``(l, m, n)`` triples,
        or any object with a short ``str``.
    frequencies : array-like (k,), optional
        Frequencies of the computed modes [Hz], added to the row
        labels.
    ax : matplotlib axes, optional
    cmap : str, optional
    gamma : float, optional
        Exponent of the power-law colour scale; below one it stretches
        the low end.
    show : bool, optional
    xlabel, ylabel, title : str, optional
        The title defaults to what the matrix says: one on the diagonal
        and the largest value off it.

    Returns
    -------
    matplotlib axes
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import PowerNorm
    from matplotlib.patches import Rectangle

    matrix = np.asarray(matrix, dtype=float)
    k = matrix.shape[0]
    off_diagonal = matrix - np.diag(np.diag(matrix))

    if ax is None:
        _, ax = plt.subplots(figsize=(7.0, 5.8))
    norm = PowerNorm(gamma=gamma, vmin=0.0, vmax=1.0)
    image = ax.imshow(matrix, cmap=cmap, norm=norm)

    rows = ([f"{i + 1}  ({frequencies[i]:.0f} Hz)" for i in range(k)]
            if frequencies is not None else [str(i + 1) for i in range(k)])
    ax.set_xticks(range(k), [str(v) for v in labels], rotation=45,
                  ha="right", fontsize=8)
    ax.set_yticks(range(k), rows, fontsize=8)
    ax.set_xticks(np.arange(-0.5, k, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, k, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.1)
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for i in range(k):
        ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                               edgecolor="black", linewidth=1.0))
        for j in range(k):
            value = matrix[i, j]
            if value >= 0.005:
                ax.text(j, i, f"{value:.2f}".lstrip("0") if value < 0.995
                        else "1.00", ha="center", va="center", fontsize=7.5,
                        color="white" if norm(value) > 0.55 else "0.25",
                        fontweight="bold" if i == j else "normal")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title or ("Every mode against every label: "
                           f"{np.diag(matrix).min():.2f} on the diagonal, "
                           f"at most {off_diagonal.max():.3f} off it"),
                 fontsize=10)
    bar = ax.figure.colorbar(image, ax=ax, shrink=0.86,
                             ticks=[0, 0.01, 0.1, 0.5, 1.0])
    bar.set_label("MAC")
    bar.ax.set_yticklabels(["0", "0.01", "0.1", "0.5", "1"])
    ax.figure.tight_layout()
    if show:
        plt.show()
    return ax
