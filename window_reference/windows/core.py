"""Exact reference moments for complete-pair sliding square windows.

All public numerical results are fractions.  There are no third-party
dependencies, periodic boundary conditions, or random-number generators.
"""

from fractions import Fraction
from math import comb


def _integer(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}")


def pair_moments(q):
    """Return mean, variance, shared-endpoint covariance of |D-D'|.

    D and D' are independent absolute differences of two independent
    uniform intensities in {0, ..., q-1}.
    """
    _integer(q, "q", 2)
    mean = Fraction(4 * (q**4 - 1), 15 * q**3)
    variance = Fraction((q*q - 1) * (9*q**6 + 34*q**4 + 16*q*q + 16),
                        225*q**6)
    covariance = Fraction((q*q - 4) * (q*q - 1) * (3*q**4 + 20*q*q - 8),
                          450*q**6)
    return mean, variance, covariance


def line_overlap_sums(length, width):
    """Return sum of overlaps and sum of squared overlaps, ordered windows."""
    _integer(width, "width", 2)
    _integer(length, "length", width)
    starts = length - width + 1
    radius = min(width - 1, starts - 1)
    overlap = starts * width + 2 * sum(
        (starts - shift) * (width - shift) for shift in range(1, radius + 1))
    squared = starts * width**2 + 2 * sum(
        (starts - shift) * (width - shift)**2 for shift in range(1, radius + 1))
    return overlap, squared


def grid_counts(rows, columns, width):
    """Return exact pair-occurrence, repeated-pair and degree counts."""
    _integer(width, "width", 2)
    _integer(rows, "rows", width)
    _integer(columns, "columns", width)
    window_count = (rows - width + 1) * (columns - width + 1)
    pairs_per_window = comb(width**2, 2)
    gm, fm = line_overlap_sums(rows, width)
    gn, fn = line_overlap_sums(columns, width)
    square_numerator = fm*fn - gm*gn
    assert square_numerator % 2 == 0
    pair_squares = square_numerator // 2
    degree_squares = (width**2 - 1)**2 * gm*gn
    shared_numerator = degree_squares - 2*pair_squares
    assert shared_numerator % 2 == 0
    return {
        "rows": rows, "columns": columns, "width": width,
        "window_count": window_count, "pairs_per_window": pairs_per_window,
        "pair_occurrences": window_count * pairs_per_window,
        "pair_squares": pair_squares, "degree_squares": degree_squares,
        "shared_edge_weight": shared_numerator // 2,
        "row_overlap_sum": gm, "row_overlap_square_sum": fm,
        "column_overlap_sum": gn, "column_overlap_square_sum": fn,
    }


def window_moments(rows, columns, width, q=256):
    """Exact moments for the normalized complete-pair sliding-window score.

    The score averages all unordered pair distances within each width by
    width window, and then divides by q-1.  Width 2 is ordinary BACI.
    pair_independence_variance discards every inter-occurrence covariance.
    The reported ratio is exact variance divided by this approximation.
    """
    counts = grid_counts(rows, columns, width)
    mean, variance, covariance = pair_moments(q)
    occurrences = counts["pair_occurrences"]
    numerator = ((variance - 2*covariance) * counts["pair_squares"]
                 + covariance * counts["degree_squares"])
    exact_variance = numerator / (occurrences * (q-1))**2
    pair_variance = variance / (occurrences * (q-1)**2)
    return {**counts, "q": q, "mean": mean / (q-1),
            "variance": exact_variance,
            "pair_independence_variance": pair_variance,
            "variance_ratio": exact_variance / pair_variance}


def large_grid_ratio(width, q=256):
    """Limit of exact/pair-independence variance as BOTH dimensions grow."""
    _integer(width, "width", 2)
    _, variance, covariance = pair_moments(q)
    t = width**2
    return (Fraction(4*t-1, 9)
            + Fraction(2*(9*t*t-13*t+1), 9) * covariance / variance)


def large_alphabet_large_grid_ratio(width):
    """q-to-infinity limit after the fixed-width, two-dimension grid limit."""
    _integer(width, "width", 2)
    t = width**2
    return Fraction(9*t*t-t-2, 27)
