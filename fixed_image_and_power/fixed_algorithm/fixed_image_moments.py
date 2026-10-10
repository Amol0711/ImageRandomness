"""Exact conditional moments for fixed images and independent uniform comparisons.

BACI uses all overlapping 2x2 windows, six unordered distances per window,
and ordinary absolute differences. UACI is the mean absolute pixel difference.
All finite sums are evaluated with integers. ``exact=False`` converts only the
final rational answers to Python floats. Fractions are returned by default.

Requires NumPy. No sampling, fitted parameters, or Gaussian tail calculation
is performed. Independent whole-image repetition averages retain the mean and
divide every variance by ``repeats`` exactly once.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral
import sys

import numpy as np

Number = Fraction | float
_INT64_MAX = int(np.iinfo(np.int64).max)


@dataclass(frozen=True)
class FixedBACIMoments:
    mean: Number
    variance: Number
    pair_occurrence_variance: Number
    independent_edge_variance: Number
    vertex_projection_variance: Number
    edge_residual_variance: Number
    raw_mean: Number
    normalization: int
    pixels: int
    windows: int
    repeats: int
    reflected_classes: int
    arithmetic: str


@dataclass(frozen=True)
class FixedUACIMoments:
    mean: Number
    variance: Number
    pooled_variance: Number
    pooled_excess: Number
    pixels: int
    repeats: int


def _integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f'{name} must be an integer, not a boolean or float')
    value = int(value)
    if value < minimum:
        raise ValueError(f'{name} must be at least {minimum}')
    return value


def _arguments(image, q, repeats, exact, grid):
    q = _integer(q, 'q', 2)
    repeats = _integer(repeats, 'repeats', 1)
    if not isinstance(exact, bool):
        raise TypeError('exact must be bool')
    if q > sys.maxsize:
        raise OverflowError('the declared alphabet must fit a platform array index')
    a = np.asarray(image)
    if a.dtype.kind not in 'iu':
        raise TypeError('pixels must have an integer dtype; booleans, floats and objects are rejected')
    if grid:
        if a.ndim != 2 or min(a.shape, default=0) < 2:
            raise ValueError('BACI requires a 2D image with both dimensions at least 2')
    elif a.ndim not in (1, 2) or a.size == 0:
        raise ValueError('UACI requires a nonempty 1D or 2D image')
    if a.size == 0 or int(a.min()) < 0 or int(a.max()) >= q:
        raise ValueError('every pixel must belong to {0,...,q-1}')
    if int(a.max()) > _INT64_MAX:
        raise OverflowError('pixel values must fit signed 64-bit storage')
    return a.astype(np.int64, copy=False), q, repeats, exact


def _class_tables(image, q):
    """O(MN+q+tq+t^2q) exact operations, with reflected law reuse.

    N_a(x) counts the q uniform comparison values giving difference x.
    g_a(x) is sum_y N_a(y)*abs(x-y), computed by two prefix sums.
    Pair tables use integer numerators with denominator q^2 or q^4.
    """
    # Images a and q-1-a have identical conditional difference laws.
    reflected = np.minimum(image, q-1-image)
    present = np.flatnonzero(np.bincount(reflected.ravel(), minlength=q))
    lookup = np.full(q, -1, dtype=np.intp)
    lookup[present] = np.arange(len(present), dtype=np.intp)
    ids = lookup[reflected]
    t = len(present)
    # Every local pair-moment intermediate is bounded by q^4*(q-1)^2.
    # Each centered weighted vertex projection has |F_i(x)|<=12*q^2*(q-1).
    # Count rows are at most two, so 288*q^4*(q-1)^2 bounds each squared term.
    use_int64 = 288*q**4*(q-1)**2 <= _INT64_MAX
    dtype = np.int64 if use_int64 else object
    x = np.arange(q, dtype=dtype)
    counts = np.empty((t, q), dtype=dtype)
    g = np.empty_like(counts)
    first = np.empty(t, dtype=dtype)
    second = np.empty(t, dtype=dtype)
    for k, value in enumerate(present):
        a = int(value)
        row = ((x <= a).astype(np.int64) + (x <= q-1-a).astype(np.int64)).astype(dtype)
        row[0] = 1
        counts[k] = row
        first[k] = np.sum(row*x)
        second[k] = np.sum(row*x*x)
        mass = np.cumsum(row)
        first_prefix = np.cumsum(row*x)
        g[k] = x*(2*mass-q) + first[k] - 2*first_prefix
    pair_mean = counts @ g.T
    pair_second = q*(second[:, None]+second[None, :])-2*first[:, None]*first[None, :]
    pair_variance = q*q*pair_second-pair_mean*pair_mean
    endpoint_variance = q*(counts @ (g*g).T)-pair_mean*pair_mean
    residual = pair_variance-endpoint_variance-endpoint_variance.T
    if any(int(x) < 0 for x in residual.ravel()):
        raise ArithmeticError('negative exact residual variance')
    return ids, counts, g, pair_mean, pair_variance, residual, use_int64


def _sum_nonnegative(a):
    """Exact integer sum, including totals exceeding int64 capacity."""
    if a.dtype == object:
        return int(np.sum(a))
    largest = int(a.max(initial=0))
    if largest == 0:
        return 0
    # Each NumPy partial sum is provably within int64. Python combines them.
    length = max(1, _INT64_MAX // largest)
    flat = a.ravel()
    return sum(int(np.sum(flat[start:start+length], dtype=np.int64))
               for start in range(0, flat.size, length))


def _edge_counts(ids, t):
    """Weighted and squared-weight type counts from four edge orientations."""
    m, n = ids.shape
    if 16*m*n > _INT64_MAX:
        raise OverflowError('grid size exceeds the conservative int64 edge-count bound')
    weights = np.zeros((t, t), dtype=np.int64)
    squares = np.zeros_like(weights)
    orientations = ((0, 1), (1, -1), (1, 0), (1, 1))
    for dr, dc in orientations:
        r0, r1 = 0, m-dr
        c0, c1 = max(0, -dc), min(n, n-dc)
        a = ids[r0:r1, c0:c1]
        b = ids[r0+dr:r1+dr, c0+dc:c1+dc]
        key = (np.minimum(a, b)*t+np.maximum(a, b)).ravel()
        c = np.ones(a.shape, dtype=np.int64)
        if dr == 0:
            c[1:-1] = 2
        elif dc == 0:
            c[:, 1:-1] = 2
        # np.bincount(weights=...) uses floating accumulation, so count each
        # integer multiplicity separately with the unweighted integer routine.
        for multiplicity in (1, 2):
            count = np.bincount(key[c.ravel() == multiplicity], minlength=t*t).reshape(t, t)
            weights += multiplicity*count
            squares += multiplicity*multiplicity*count
    return weights, squares


def fixed_baci_moments(image, q=256, *, repeats=1, exact=True):
    """Conditional BACI moments for one fixed image and uniform comparison.

    Input must be a 2D integer image with both dimensions >=2 and values in
    {0,...,q-1}. Every comparison pixel is independent uniform on that alphabet.
    ``repeats`` means independent repetitions of the whole comparison image.

    ``variance`` is the exact full variance. ``pair_occurrence_variance``
    treats each of the 6K occurrences separately and omits every covariance.
    ``independent_edge_variance`` first combines repeated occurrences, keeps
    c_e^2 Var(h_e), and omits shared-endpoint covariances. Neither diagnostic is
    substituted for ``variance``. Vertex and residual fields partition the
    full variance. ``raw_mean`` is E[S], before normalization, unaffected by R.

    All six normalized moment/variance fields use fraction units, not percent.
    ``exact=False`` converts final exact fractions to float, with no change to
    the calculation. Integer arithmetic uses checked int64 local arrays when
    safe and arbitrary-precision Python integers otherwise. Graph totals use
    arbitrary precision in both paths. No alphabet-size upper bound beyond
    array-index capacity is imposed, but time and memory grow with q.

    With n=MN and t distinct reflected pixel classes, arithmetic complexity is
    O((n+t^2)q), and auxiliary memory is O(n+Nq+tq+t^2+q). This counts integer
    arithmetic operations, not bit complexity. One image row of projections
    is held at a time; no n-by-q projection matrix or adjacency list is stored.
    """
    a, q, repeats, exact = _arguments(image, q, repeats, exact, True)
    m, n = a.shape
    ids, counts, g, pair_mean, pair_var, residual, use_int64 = _class_tables(a, q)
    t = len(counts)
    weights, squares = _edge_counts(ids, t)
    raw_mean_numerator = sum(int(c)*int(x) for c, x in zip(weights.ravel(), pair_mean.ravel()))
    pair_numerator = sum(int(c)*int(x) for c, x in zip(weights.ravel(), pair_var.ravel()))
    edge_numerator = sum(int(c)*int(x) for c, x in zip(squares.ravel(), pair_var.ravel()))
    residual_numerator = sum(int(c)*int(x) for c, x in zip(squares.ravel(), residual.ravel()))
    vertex_numerator = 0
    columns = np.arange(n)
    for i in range(m):
        center = ids[i]
        projection = np.zeros((n, q), dtype=np.int64 if use_int64 else object)
        for dr in (-1, 0, 1):
            row = i+dr
            if row < 0 or row >= m:
                continue
            for dc in (-1, 0, 1):
                if dr == dc == 0:
                    continue
                c0, c1 = max(0, -dc), min(n, n-dc)
                own = center[c0:c1]
                neighbor = ids[row, c0+dc:c1+dc]
                if dr == 0:
                    w = 1 if i in (0, m-1) else 2
                elif dc == 0:
                    w = np.where((columns[c0:c1] == 0) | (columns[c0:c1] == n-1), 1, 2)[:, None]
                else:
                    w = 1
                projection[c0:c1] += w*(q*g[neighbor]-pair_mean[own, neighbor, None])
        terms = counts[center]*projection*projection
        vertex_numerator += _sum_nonnegative(terms)
    k = (m-1)*(n-1)
    scale = 6*k*(q-1)
    denominator = scale*scale*repeats
    vertex = Fraction(vertex_numerator, q**5*denominator)
    remainder = Fraction(residual_numerator, q**4*denominator)
    values = [Fraction(raw_mean_numerator, q*q*scale), vertex+remainder,
              Fraction(pair_numerator, q**4*denominator),
              Fraction(edge_numerator, q**4*denominator), vertex, remainder,
              Fraction(raw_mean_numerator, q*q)]
    if not exact:
        values = list(map(float, values))
    return FixedBACIMoments(*values, scale, m*n, k, repeats, t,
                           'checked int64 with Python integer totals' if use_int64 else 'Python integers')


def fixed_uaci_moments(image, q=256, *, repeats=1, exact=True):
    """Exact fixed-coordinate UACI moments and pooled-law diagnostic.

    The comparison pixels are independent uniform. ``pooled_variance`` samples
    the fixed intensity independently from its empirical histogram at every
    coordinate. It differs from the fixed-coordinate design by pooled_excess.
    Input may be a nonempty 1D or 2D integer array. Independent whole-image
    averaging divides variance, pooled_variance and pooled_excess by repeats.
    Complexity is O(n+q) operations and O(n+q) auxiliary storage for n
    pixels, including signed-int64 input conversion. If no conversion is
    needed, the additional storage is O(q).
    """
    a, q, repeats, exact = _arguments(image, q, repeats, exact, False)
    histogram = np.bincount(a.ravel(), minlength=q)
    total_mean = total_second = total_mean_square = 0
    for intensity in np.flatnonzero(histogram):
        v, count = int(intensity), int(histogram[intensity])
        # A=sum_u |v-u|, B=sum_u |v-u|^2. Python integers throughout.
        right = q-1-v
        first = (v*(v+1)+right*(right+1))//2
        second = (v*(v+1)*(2*v+1)+right*(right+1)*(2*right+1))//6
        total_mean += count*first
        total_second += count*second
        total_mean_square += count*first*first
    size = int(a.size)
    mean = Fraction(total_mean, size*q*(q-1))
    fixed = Fraction(q*total_second-total_mean_square, size*size*q*q*(q-1)**2*repeats)
    excess = Fraction(size*total_mean_square-total_mean*total_mean,
                      size**3*q*q*(q-1)**2*repeats)
    values = [mean, fixed, fixed+excess, excess]
    if not exact:
        values = list(map(float, values))
    return FixedUACIMoments(*values, size, repeats)
