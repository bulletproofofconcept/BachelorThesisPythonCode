from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike

"""
This code implements the change probability Hurst exponent estimator on the fBm
model
"""


def estimate_change_probability(profile: ArrayLike, stepsize: int) -> float:
    """
    Estimate the probability of a direction change using lagged triples.

    For every start index i such that i + 2*stepsize < n, consider the triple:
        a = x[i]
        b = x[i + stepsize]
        c = x[i + 2*stepsize]

    A "change" occurs if b is a local extremum relative to a and c:
        - local maximum: a < b and b >= c
        - local minimum: a >= b and b < c

        The estimate is:
        (#changes) / (#triples)

    Parameters
    ----------
    profile
        1D sequence of numeric values.
    stepsize
        Positive integer lag between the points in each triple.

    Returns
    -------
    float
        Estimated change probability in [0, 1].

    Raises
    ------
    ValueError
        If stepsize <= 0 or if the profile is too short for at least one triple.
    """
    if stepsize <= 0:
        raise ValueError("stepsize must be a positive integer")

    x = np.asarray(profile, dtype=float)
    n = x.size
    m = n - 2 * stepsize  # number of valid triples
    if m <= 0:
        raise ValueError("profile is too short for the given stepsize (need n > 2*stepsize)")

    a = x[:m]
    b = x[stepsize:stepsize + m]
    c = x[2 * stepsize:2 * stepsize + m]

    maxima = (a < b) & (b >= c)
    minima = (a >= b) & (b < c)

    changes = np.count_nonzero(maxima | minima)
    return changes / m


def hurst_from_change_probability(change_probability: float) -> float:
    """
    Estimate the Hurst exponent H (for an fBm model) from a change probability p.

        theta = pi * (1 - p) / 2
        H = 1 + log2(sin(theta))

    Parameters
    ----------
    change_probability
        Change probability p in [0, 1].

    Returns
    -------
    float
        Estimated Hurst exponent.

    Raises
    ------
    ValueError
        If change_probability is not in [0, 1].
    """
    p = float(change_probability)
    if not (0.0 <= p <= 1.0):
        raise ValueError("change_probability must be in [0, 1]")

    theta = np.pi * (1.0 - p) / 2.0
    s = np.sin(theta)

    # If p==1 => theta==0 => sin==0 => log2(0) = -inf (might be expected mathematically).
    # If you want to avoid -inf, uncomment the next line:
    # s = max(s, np.finfo(float).tiny)

    return 1.0 + np.log2(s)




