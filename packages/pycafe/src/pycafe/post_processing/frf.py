r"""
Reading a frequency response: where its peaks are.

A sweep is a table of numbers until someone asks where the resonances
are, and the answer is not simply "at the largest sample": with a step
of half a hertz on a peak two hertz wide, the largest sample can sit
half a step off. :func:`find_response_peaks` fits a parabola through the
three points around each local maximum, which recovers the position to
well under the step.

.. code-block:: python

    peaks = find_response_peaks(frequencies, {"w": w_centre, "p": p_mic})
    for f, seen in peaks:
        print(f, "seen in", "+".join(sorted(seen)))
"""

import numpy as np


def find_response_peaks(frequencies, response, *, merge_within=None):
    """
    Local maxima of one or several responses, refined below the step.

    Parameters
    ----------
    frequencies : array-like (N,)
        The frequencies of the sweep, evenly spaced [Hz].
    response : array-like or dict
        One response, or ``{name: response}`` for several signals of the
        same sweep — a displacement and a microphone, say. Complex
        input is read through its modulus.
    merge_within : float, optional
        How close two peaks of different signals have to be to count as
        the same resonance [Hz]. Default: two steps of the sweep. Only
        used when ``response`` is a dict.

    Returns
    -------
    list
        With one signal, the peak frequencies as an ndarray. With a
        dict, a list of ``(frequency, {names})`` sorted by frequency,
        one entry per resonance, saying which signals show it.

    Examples
    --------
    >>> find_response_peaks([1.0, 2.0, 3.0], [1.0, 3.0, 1.0])
    array([2.])
    """
    frequencies = np.asarray(frequencies, dtype=float)
    if frequencies.size < 3:
        return np.array([]) if not isinstance(response, dict) else []
    step = float(frequencies[1] - frequencies[0])

    if not isinstance(response, dict):
        return _peaks_of(frequencies, response, step)

    tolerance = 2.0 * step if merge_within is None else float(merge_within)
    rows = []
    for name, signal in response.items():
        for f in _peaks_of(frequencies, signal, step):
            near = [row for row in rows if abs(row[0] - f) < tolerance]
            if near:
                near[0][1].add(name)
            else:
                rows.append([f, {name}])
    rows.sort(key=lambda row: row[0])
    return [(f, names) for f, names in rows]


def _peaks_of(frequencies, signal, step):
    """Refined local maxima of one signal, on a log amplitude."""
    # On a log amplitude the parabola fits a resonance far better than on
    # the amplitude itself, where the peak is much sharper than a quadratic.
    a = np.log(np.abs(np.asarray(signal)) + 1e-300)
    out = []
    for i in range(1, frequencies.size - 1):
        if a[i] > a[i - 1] and a[i] >= a[i + 1]:
            denominator = a[i - 1] - 2 * a[i] + a[i + 1]
            shift = (0.5 * (a[i - 1] - a[i + 1]) / denominator
                     if denominator != 0 else 0.0)
            out.append(frequencies[i] + shift * step)
    return np.array(out)


def missing_from_response(peaks, resonances, *, tolerance=1.0):
    """
    Which resonances the sweep does not show.

    A load that sits on a node line of a mode cannot excite it, so the
    mode is missing from the response and nothing is wrong. This is the
    list to look at before saying so.

    Parameters
    ----------
    peaks : array-like
        Peak frequencies found in the sweep [Hz]; the list of
        :func:`find_response_peaks` is accepted as it is.
    resonances : array-like
        The frequencies to look for [Hz].
    tolerance : float, optional
        How far a peak may be and still count as that resonance [Hz].

    Returns
    -------
    ndarray
        The resonances with no peak near them.
    """
    peaks = np.asarray([p[0] if isinstance(p, tuple) else p for p in peaks],
                       dtype=float)
    resonances = np.asarray(resonances, dtype=float)
    if peaks.size == 0:
        return resonances
    return np.array([f for f in resonances
                     if np.min(np.abs(peaks - f)) > tolerance])
