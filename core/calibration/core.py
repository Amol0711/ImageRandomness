"""Transparent finite-law reference implementation.

The default BACI experiment uses independent, identically distributed uniform
image pixels U and V, with D=abs(U-V). Fixed-image routines instead condition on
the supplied image and make only the comparison pixels independent uniform.
R denotes independent repetitions of the *entire* statistic. Its mean is
unchanged and its variance is divided by R. No routine infers independence from
data, estimates nuisance parameters, or certifies cipher security.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from math import comb, isfinite, sqrt
from numbers import Integral, Rational, Real
from statistics import NormalDist
from typing import Mapping, Sequence

import numpy as np

Number = Fraction | float


@dataclass(frozen=True)
class PairConstants:
    """Unnormalized pair distance mean, variance A, and shared-vertex covariance C."""
    mean_distance: Number
    A: Number
    C: Number


@dataclass(frozen=True)
class BACIMoments:
    mean: Number
    variance: Number
    independent_pair_variance: Number
    within_block_only_variance: Number
    A: Number
    C: Number
    K: int
    S2: int
    W: int
    repeats: int


@dataclass(frozen=True)
class EdgeMoments:
    """Moments and orthogonal variance components for the specified edge sum."""
    mean: Number
    variance: Number
    vertex_projection_variance: Number
    edge_residual_variance: Number
    repeats: int = 1


@dataclass(frozen=True)
class UACIMoments:
    mean: Number
    variance: Number
    pooled_variance: Number
    pooled_excess: Number
    pixels: int
    repeats: int


@dataclass(frozen=True)
class Interval:
    """Unclipped Gaussian reference interval; not an exact finite-sample interval."""
    lower: float
    upper: float
    mean: float
    standard_error: float
    alpha: float
    method: str


def _integer(value, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f'{name} must be an integer, not a boolean or float')
    value = int(value)
    if value < minimum:
        raise ValueError(f'{name} must be at least {minimum}')
    return value


def _exact_flag(value) -> bool:
    if not isinstance(value, bool):
        raise TypeError('exact must be bool')
    return value


def _fraction(value, name: str) -> Fraction:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Rational):
        raise TypeError(f'{name} must be rational in exact mode; use Fraction or integers')
    return Fraction(int(value.numerator), int(value.denominator))


def _real(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f'{name} must be a finite real scalar')
    value = float(value)
    if not isfinite(value):
        raise ValueError(f'{name} must be finite')
    return value


def _image(image, q: int, *, grid: bool) -> np.ndarray:
    a = np.asarray(image)
    if a.dtype.kind not in 'iu':
        raise TypeError('pixels must have an integer dtype; floating, boolean, and object arrays are rejected')
    if grid:
        if a.ndim != 2 or min(a.shape) < 2:
            raise ValueError('BACI requires one two-dimensional image with both dimensions at least 2')
    elif a.ndim not in (1, 2) or a.size == 0:
        raise ValueError('a fixed UACI image must be a nonempty one- or two-dimensional integer array')
    if a.size == 0 or int(a.min()) < 0 or int(a.max()) >= q:
        raise ValueError('every pixel must belong to {0,...,q-1}')
    if int(a.max()) > np.iinfo(np.int64).max:
        raise OverflowError('pixel array arithmetic supports signed 64-bit integers')
    return a.astype(np.int64, copy=False)


def grid_edge_weights(m: int, n: int) -> dict[tuple[int, int], int]:
    """Literal all-window multiplicities with row-major vertices, including thin strips."""
    m, n = _integer(m, 'm', 2), _integer(n, 'n', 2)
    result = Counter()
    for i in range(m - 1):
        for j in range(n - 1):
            block = (i*n+j, i*n+j+1, (i+1)*n+j, (i+1)*n+j+1)
            result.update(combinations(block, 2))
    return dict(result)


def grid_coefficients(m: int, n: int) -> tuple[int, int]:
    """Return (sum c_e^2, sum_{e<f, e intersects f} c_e*c_f)."""
    m, n = _integer(m, 'm', 2), _integer(n, 'n', 2)
    return 10*m*n - 12*m - 12*n + 14, 62*m*n - 96*m - 96*n + 148


def baci_statistic(d, q: int = 256, *, method: str = 'weighted', exact: bool = False) -> Number:
    """BACI of an ordinary difference image, using every overlapping 2x2 window.

    The numerator is the sum of all six unordered pixel-pair distances in every
    window. Input is exactly one integer-valued 2D difference image. No batch
    dimension, wraparound subtraction, modulo difference, or broadcasting occurs.
    Integer accumulation is checked before use. Return units are [0,1].
    """
    q, exact = _integer(q, 'q', 2), _exact_flag(exact)
    a = _image(d, q, grid=True)
    if method not in ('weighted', 'literal'):
        raise ValueError("method must be 'weighted' or 'literal'")
    m, n = a.shape
    scale = 6*(m-1)*(n-1)*(q-1)
    if 6*(m-1)*(n-1)*int(a.max()) > np.iinfo(np.int64).max:
        raise OverflowError('the conservative integer numerator bound exceeds int64')
    if method == 'literal':
        corners = (a[:-1, :-1], a[:-1, 1:], a[1:, :-1], a[1:, 1:])
        numerator = sum(int(np.abs(u-v).sum(dtype=np.int64))
                        for u, v in combinations(corners, 2))
    else:
        h = np.abs(a[:, 1:] - a[:, :-1])
        v = np.abs(a[1:, :] - a[:-1, :])
        numerator = (2*int(h.sum(dtype=np.int64))
                     - int(h[0].sum(dtype=np.int64)) - int(h[-1].sum(dtype=np.int64))
                     + 2*int(v.sum(dtype=np.int64))
                     - int(v[:, 0].sum(dtype=np.int64)) - int(v[:, -1].sum(dtype=np.int64))
                     + int(np.abs(a[1:, 1:]-a[:-1, :-1]).sum(dtype=np.int64))
                     + int(np.abs(a[1:, :-1]-a[:-1, 1:]).sum(dtype=np.int64)))
    value = Fraction(numerator, scale)
    return value if exact else float(value)


def baci_from_images(u, v, q: int = 256, *, method: str = 'weighted', exact: bool = False) -> Number:
    """Form abs(U-V) with signed subtraction, then evaluate BACI; shapes must match."""
    q = _integer(q, 'q', 2)
    a, b = _image(u, q, grid=True), _image(v, q, grid=True)
    if a.shape != b.shape:
        raise ValueError('image shapes must match exactly; broadcasting is forbidden')
    return baci_statistic(np.abs(a-b), q, method=method, exact=exact)


def uaci_statistic(u, v, q: int = 256, *, exact: bool = False) -> Number:
    """Mean abs(U-V)/(q-1) over matched integer images; no percentage scaling."""
    q, exact = _integer(q, 'q', 2), _exact_flag(exact)
    a, b = _image(u, q, grid=False), _image(v, q, grid=False)
    if a.shape != b.shape:
        raise ValueError('image shapes must match exactly; broadcasting is forbidden')
    # Python integer summation avoids overflow without altering pixel subtraction.
    numerator = sum(map(int, np.abs(a-b).ravel()))
    value = Fraction(numerator, int(a.size)*(q-1))
    return value if exact else float(value)


def npcr_statistic(u, v, q: int = 256, *, exact: bool = False) -> Number:
    """Fraction of unequal positions in matched integer images."""
    q, exact = _integer(q, 'q', 2), _exact_flag(exact)
    a, b = _image(u, q, grid=False), _image(v, q, grid=False)
    if a.shape != b.shape:
        raise ValueError('image shapes must match exactly; broadcasting is forbidden')
    value = Fraction(int(np.count_nonzero(a != b)), int(a.size))
    return value if exact else float(value)


def count_entropy(counts) -> float:
    """Empirical Shannon entropy in bits from one nonempty integer count vector.

    Zero-count categories contribute zero. The vector represents the declared
    alphabet, and counts are data, not normalized probabilities.
    """
    a = np.asarray(counts)
    if a.ndim != 1 or a.size == 0:
        raise ValueError('counts must be a nonempty one-dimensional vector')
    if a.dtype.kind not in 'iu':
        raise TypeError('counts must have an integer dtype')
    if int(a.min()) < 0:
        raise ValueError('counts must be nonnegative')
    total = sum(map(int, a))
    if total == 0:
        raise ValueError('at least one count must be positive')
    p = np.asarray([int(value)/total for value in a if value > 0], dtype=float)
    return float(-np.dot(p, np.log2(p)))


def block_entropy(block, q: int = 256) -> float:
    """Empirical entropy in bits of one supplied integer pixel block."""
    q = _integer(q, 'q', 2)
    a = _image(block, q, grid=False)
    return count_entropy(np.bincount(a.ravel(), minlength=q))


def local_entropy_mean(blocks, q: int = 256) -> float:
    """Mean entropy of a predeclared collection of equal-sized supplied blocks.

    Input shape is (k,t) for flattened blocks or (k,h,w) for 2D blocks. This
    computes a statistic only. Calibration requires independent identically
    distributed blocks; callers retain and verify block-location provenance.
    Array values alone cannot establish that selections are disjoint. The
    function does not select blocks or infer sampling/replication protocols.
    """
    q = _integer(q, 'q', 2)
    a = np.asarray(blocks)
    if a.ndim not in (2, 3) or min(a.shape) < 1:
        raise ValueError('blocks must have shape (k,t) or (k,h,w), with all dimensions positive')
    return sum(block_entropy(block, q) for block in a)/a.shape[0]


def local_entropy_from_counts(count_rows) -> float:
    """Mean block entropy from equal-total count rows with a common alphabet.

    This is equivalent to local_entropy_mean after per-block histogramming.
    Independence and sampling provenance remain caller-supplied assumptions.
    """
    a = np.asarray(count_rows)
    if a.ndim != 2 or min(a.shape) < 1:
        raise ValueError('count_rows must have nonempty shape (blocks,alphabet)')
    if a.dtype.kind not in 'iu':
        raise TypeError('counts must have an integer dtype')
    totals = [sum(map(int, row)) for row in a]
    if len(set(totals)) != 1:
        raise ValueError('all blocks must have the same number of pixels')
    return sum(count_entropy(row) for row in a)/a.shape[0]


def triangular_difference_law(q: int = 256, *, exact: bool = True) -> tuple[Number, ...]:
    """Law of abs(U-V) for two independent uniform q-alphabet variables."""
    q, exact = _integer(q, 'q', 2), _exact_flag(exact)
    p = [Fraction(1, q)] + [Fraction(2*(q-d), q*q) for d in range(1, q)]
    return tuple(p if exact else map(float, p))


def fixed_difference_law(value: int, q: int = 256, *, exact: bool = True) -> tuple[Number, ...]:
    """Law of abs(value-U), conditional on one fixed integer pixel."""
    q, value, exact = _integer(q, 'q', 2), _integer(value, 'value', 0), _exact_flag(exact)
    if value >= q:
        raise ValueError('value must be smaller than q')
    counts = Counter(abs(value-u) for u in range(q))
    p = [Fraction(counts[d], q) for d in range(q)]
    return tuple(p if exact else map(float, p))


def _power_sums(n: int, upto: int) -> list[Fraction]:
    out = [Fraction(n)]
    for r in range(1, upto + 1):
        out.append((Fraction((n+1)**(r+1)-1)
                    - sum((comb(r+1, t)*out[t] for t in range(r)), Fraction()))/(r+1))
    return out


def _abs_expectation(law):
    """O(q) prefix-sum expectations g(x)=E|x-X|; works for Fraction or float."""
    q = len(law)
    total_mean = sum(i*p for i, p in enumerate(law))
    mass = first = law[0]*0
    out = []
    for x, p in enumerate(law):
        mass += p
        first += x*p
        out.append(x*(2*mass-1) + total_mean - 2*first)
    return out


def iid_pair_constants(q: int = 256, *, exact: bool = True,
                       method: str = 'polynomial') -> PairConstants:
    """Exact constants for h(D1,D2)=abs(D1-D2) under the triangular law.

    polynomial uses seven finite power sums and has no alphabet-sized loop.
    sum independently uses the discrete triangular law and O(q) prefix sums.
    Both compute rational results before any requested float conversion.
    """
    q, exact = _integer(q, 'q', 2), _exact_flag(exact)
    if method == 'polynomial':
        mu = Fraction(4*(q**4-1), 15*q**3)
        A = Fraction(9*q**8+25*q**6-18*q**4-16, 225*q**6)
        powers = _power_sums(q-1, 7)
        raw = [Fraction(1)] + [2*(q*powers[r]-powers[r+1])/q**2 for r in range(1, 7)]
        coeff = [Fraction(q**3-q, 3*q*q), Fraction(-3*q*q+2, 3*q*q),
                 Fraction(2, q), Fraction(-2, 3*q*q)]
        C = sum((coeff[i]*coeff[j]*raw[i+j] for i in range(4) for j in range(4)), Fraction()) - mu*mu
    elif method == 'sum':
        p = triangular_difference_law(q)
        g = _abs_expectation(p)
        mu = sum((p[x]*g[x] for x in range(q)), Fraction())
        ed = sum((x*p[x] for x in range(q)), Fraction())
        ed2 = sum((x*x*p[x] for x in range(q)), Fraction())
        A = 2*(ed2-ed*ed)-mu*mu
        C = sum((p[x]*(g[x]-mu)**2 for x in range(q)), Fraction())
    else:
        raise ValueError("method must be 'polynomial' or 'sum'")
    if not exact:
        return PairConstants(float(mu), float(A), float(C))
    return PairConstants(mu, A, C)


def iid_baci_moments(m: int, n: int, q: int = 256, *, repeats: int = 1,
                     exact: bool = True) -> BACIMoments:
    """Exact all-window moments under the iid two-random-image null.

    independent_pair_variance is the documented six-independent-pairs comparator.
    within_block_only_variance retains within-window dependence but treats
    distinct windows as independent. Neither comparator replaces variance.
    repeats averages independent whole-image statistics with identical design.
    """
    m, n, q = _integer(m, 'm', 2), _integer(n, 'n', 2), _integer(q, 'q', 2)
    repeats, exact = _integer(repeats, 'repeats', 1), _exact_flag(exact)
    c = iid_pair_constants(q)
    K = (m-1)*(n-1)
    S2, W = grid_coefficients(m, n)
    values = [c.mean_distance/(q-1),
              (c.A*S2+2*c.C*W)/(36*K*K*(q-1)**2*repeats),
              c.A/(6*K*(q-1)**2*repeats),
              (c.A/6+Fraction(2, 3)*c.C)/(K*(q-1)**2*repeats), c.A, c.C]
    if not exact:
        values = list(map(float, values))
    return BACIMoments(*values, K, S2, W, repeats)


def _laws(laws, exact: bool):
    try:
        rows = [list(row) for row in laws]
    except TypeError as exc:
        raise TypeError('laws must be a nonempty sequence of finite weight rows') from exc
    if not rows or len(rows[0]) < 1 or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError('laws must have equal, nonzero support sizes')
    out = []
    for row in rows:
        vals = [(_fraction(x, 'law weight') if exact else _real(x, 'law weight')) for x in row]
        if any(x < 0 for x in vals):
            raise ValueError('law weights must be nonnegative')
        total = sum(vals)
        if total <= 0 or (not exact and not isfinite(total)):
            raise ValueError('each law must have a positive finite total weight')
        out.append(tuple(x/total for x in vals))
    return out


def _edges(edges, vertices: int, exact: bool):
    if not isinstance(edges, Mapping):
        raise TypeError('edges must map distinct undirected vertex pairs to finite weights')
    result = {}
    for key, weight in edges.items():
        if not isinstance(key, tuple) or len(key) != 2:
            raise TypeError('each edge key must be a two-integer tuple')
        u, v = (_integer(x, 'vertex index', 0) for x in key)
        if u == v or max(u, v) >= vertices:
            raise ValueError('edge vertices must be distinct valid row indices')
        canonical = (min(u, v), max(u, v))
        if canonical in result:
            raise ValueError('a reversed duplicate edge is forbidden; combine its weights explicitly')
        result[canonical] = _fraction(weight, 'edge weight') if exact else _real(weight, 'edge weight')
    return result


def _nonnegative_roundoff(value: float, scale: float) -> float:
    """Remove only cancellation-sized negative residuals; fail on material negatives."""
    tol = 128*np.finfo(float).eps*max(1.0, scale)
    if not isfinite(value):
        raise FloatingPointError('non-finite floating moment; use exact mode or rescale weights')
    if value < -tol:
        raise FloatingPointError('negative residual exceeds floating roundoff tolerance; use exact mode')
    return max(value, 0.0)


def _class_graph_moments(unique, ids, edges, exact: bool) -> EdgeMoments:
    """Class reuse keeps fixed-image storage independent of pixel-law duplication."""
    q, vertices = len(unique[0]), len(ids)
    zero = Fraction() if exact else 0.0
    gs = [_abs_expectation(p) for p in unique]
    first = [sum(x*p[x] for x in range(q)) for p in unique]
    second = [sum(x*x*p[x] for x in range(q)) for p in unique]
    if not exact:
        unique, gs = np.asarray(unique, dtype=float), np.asarray(gs, dtype=float)
        projection = np.zeros((vertices, q), dtype=float)
    else:
        projection = [[Fraction() for _ in range(q)] for _ in range(vertices)]
    mean, residual_total = zero, zero
    cache = {}
    for (u, v), weight in edges.items():
        a, b = ids[u], ids[v]
        pair = (min(a, b), max(a, b))
        if pair not in cache:
            i, j = pair
            if exact:
                mu = sum((unique[i][x]*gs[j][x] for x in range(q)), zero)
                vi = sum((unique[i][x]*(gs[j][x]-mu)**2 for x in range(q)), zero)
                vj = sum((unique[j][x]*(gs[i][x]-mu)**2 for x in range(q)), zero)
            else:
                mu = float(np.dot(unique[i], gs[j]))
                vi = float(np.dot(unique[i], (gs[j]-mu)**2))
                vj = float(np.dot(unique[j], (gs[i]-mu)**2))
            var_h = second[i]+second[j]-2*first[i]*first[j]-mu*mu
            residual = var_h-vi-vj
            if not exact:
                residual = _nonnegative_roundoff(float(residual), abs(float(var_h))+vi+vj)
            elif residual < 0:
                raise ArithmeticError('negative exact canonical residual variance')
            cache[pair] = mu, residual
        mu, residual = cache[pair]
        mean += weight*mu
        residual_total += weight*weight*residual
        if exact:
            for x in range(q):
                projection[u][x] += weight*(gs[b][x]-mu)
                projection[v][x] += weight*(gs[a][x]-mu)
        else:
            projection[u] += weight*(gs[b]-mu)
            projection[v] += weight*(gs[a]-mu)
    vertex_total = zero
    for v, law_id in enumerate(ids):
        if exact:
            vertex_total += sum((unique[law_id][x]*projection[v][x]**2 for x in range(q)), zero)
        else:
            vertex_total += float(np.dot(unique[law_id], projection[v]**2))
    variance = vertex_total+residual_total
    if not exact and not all(isfinite(x) for x in (mean, variance, vertex_total, residual_total)):
        raise FloatingPointError('non-finite floating moments; use exact mode or rescale weights')
    return EdgeMoments(mean, variance, vertex_total, residual_total)


def weighted_edge_moments(laws: Sequence[Sequence], edges: Mapping[tuple[int, int], Real],
                          *, exact: bool = True, repeats: int = 1) -> EdgeMoments:
    """Moments of sum_e w_e*abs(X_u-X_v) for independent finite-law vertices.

    Row entries are nonnegative weights on the *common* support {0,...,q-1};
    each row is normalized, so integer count rows and probability rows both work.
    Exact mode requires integer/Fraction inputs. Float mode accepts finite real
    inputs. Signed edge weights, isolated vertices, empty edge sets, and
    degenerate laws are supported. Self loops and reversed duplicates are errors.
    Variance uses vertex projections plus orthogonal edge residuals. Float mode
    uses O(V*q + E + J*q) storage, with J distinct law rows, plus a cache of
    scalar moments for distinct adjacent law pairs. It is not an asymptotic result.
    """
    exact, repeats = _exact_flag(exact), _integer(repeats, 'repeats', 1)
    rows = _laws(laws, exact)
    checked = _edges(edges, len(rows), exact)
    unique, lookup, ids = [], {}, []
    for law in rows:
        if law not in lookup:
            lookup[law] = len(unique)
            unique.append(law)
        ids.append(lookup[law])
    result = _class_graph_moments(unique, ids, checked, exact)
    return EdgeMoments(result.mean, result.variance/repeats,
                       result.vertex_projection_variance/repeats,
                       result.edge_residual_variance/repeats, repeats)


def fixed_baci_moments(image, q: int = 256, *, repeats: int = 1,
                       exact: bool = False) -> EdgeMoments:
    """BACI moments conditional on the fixed image, for independent uniform U.

    The law of D_v=abs(image_v-U_v) is position dependent. Float mode shares
    identical laws; exact mode is intended for small certified examples. The
    returned mean and variance components include BACI normalization and R.
    """
    q, repeats, exact = _integer(q, 'q', 2), _integer(repeats, 'repeats', 1), _exact_flag(exact)
    a = _image(image, q, grid=True)
    values, assignments = np.unique(a, return_inverse=True)
    laws = [fixed_difference_law(int(value), q, exact=exact) for value in values]
    edges = grid_edge_weights(*a.shape)
    result = _class_graph_moments(laws, assignments.ravel().tolist(), edges, exact)
    scale = 6*(a.shape[0]-1)*(a.shape[1]-1)*(q-1)
    return EdgeMoments(result.mean/scale, result.variance/(scale*scale*repeats),
                       result.vertex_projection_variance/(scale*scale*repeats),
                       result.edge_residual_variance/(scale*scale*repeats), repeats)


def fixed_uaci_moments(image, q: int = 256, *, repeats: int = 1,
                       exact: bool = True) -> UACIMoments:
    """Conditional UACI mean/variance and pooled-variance comparator.

    Uniform comparison pixels are independent. The positive pooled_excess is the
    between-position mean term in the documented pooled comparator. It vanishes
    whenever all position means agree, not only for a constant image. Input may
    be a one-dimensional list for small witnesses or a full two-dimensional image.
    """
    q, repeats, exact = _integer(q, 'q', 2), _integer(repeats, 'repeats', 1), _exact_flag(exact)
    a = _image(image, q, grid=False)
    counts = Counter(map(int, a.ravel()))
    pixels = int(a.size)
    means, variances = {}, {}
    for v in counts:
        # Exact sums over both sides of fixed v, with zero counted only once.
        left, right = v, q-1-v
        one = left*(left+1)//2 + right*(right+1)//2
        two = left*(left+1)*(2*left+1)//6 + right*(right+1)*(2*right+1)//6
        mu = Fraction(one, q*(q-1))
        means[v] = mu
        variances[v] = Fraction(two, q*(q-1)**2)-mu*mu
    mean = sum((count*means[v] for v, count in counts.items()), Fraction())/pixels
    variance = sum((count*variances[v] for v, count in counts.items()), Fraction())/(pixels*pixels*repeats)
    excess = sum((count*(means[v]-mean)**2 for v, count in counts.items()), Fraction())/(pixels*pixels*repeats)
    vals = [mean, variance, variance+excess, excess]
    if not exact:
        vals = list(map(float, vals))
    return UACIMoments(*vals, pixels, repeats)


def gaussian_interval(mean: Real, variance: Real, *, alpha: float = 0.05) -> Interval:
    """Unclipped normal-approximation acceptance interval for a scalar statistic.

    variance is the variance of the *reported statistic*, already accounting for
    its averaging design. A zero variance returns the singleton {mean}. This
    function neither claims exact level nor applies a continuity correction.
    """
    mean, variance, alpha = _real(mean, 'mean'), _real(variance, 'variance'), _real(alpha, 'alpha')
    if variance < 0:
        raise ValueError('variance must be nonnegative')
    if not 0 < alpha < 1:
        raise ValueError('alpha must lie strictly between 0 and 1')
    z = -NormalDist().inv_cdf(alpha/2)
    sd = sqrt(variance)
    return Interval(mean-z*sd, mean+z*sd, mean, sd, alpha, 'Gaussian approximation')


def local_entropy_interval(mean: Real, single_block_sd: Real, blocks: int,
                            *, alpha: float = 0.05, scaling: str = 'corrected') -> Interval:
    """Normal interval for the average entropy of independent identical blocks.

    corrected uses single_block_sd/sqrt(blocks). linear uses sd/blocks
    as a comparison rule rather than a valid standard
    error. The caller must provide reference block moments and justify block
    independence (e.g. disjoint blocks of iid pixels). No extra repetition count
    is inferred from a reported average. Entropy units are those of the moments.
    """
    blocks = _integer(blocks, 'blocks', 1)
    sd = _real(single_block_sd, 'single_block_sd')
    if sd < 0:
        raise ValueError('single_block_sd must be nonnegative')
    if scaling not in ('corrected', 'linear'):
        raise ValueError("scaling must be 'corrected' or 'linear'")
    variance = sd*sd/(blocks if scaling == 'corrected' else blocks*blocks)
    interval = gaussian_interval(mean, variance, alpha=alpha)
    return Interval(interval.lower, interval.upper, interval.mean, interval.standard_error,
                    interval.alpha, 'Gaussian approximation' if scaling == 'corrected'
                    else 'Linear scaling comparator; not a valid standard error')


def known_null_rank_pvalue(observed_score: Real, null_scores: Sequence[Real],
                           *, exact: bool = False) -> Number:
    """Upper-tail simulation-rank p-value (1 + #{S_l >= S_0})/(L+1).

    Larger scores must mean more extreme observations. For a two-sided statistic
    pass precomputed absolute deviations from its *specified* reference mean.
    The observation and L simulated scores must be exchangeable under the fully
    specified null, with the same score definition. Ties use >= and are
    conservative. A reused bank preserves marginal, not bank-conditional, level;
    this routine supplies no simultaneous-testing or parameter-fitting guarantee.
    exact preserves rational score ordering and returns Fraction. Float mode
    checks finite real scores before conversion. At least one null score is required.
    """
    exact = _exact_flag(exact)
    convert = _fraction if exact else _real
    observed = convert(observed_score, 'observed_score')
    if isinstance(null_scores, np.ndarray) and null_scores.ndim != 1:
        raise ValueError('null_scores must be one-dimensional')
    scores = [convert(x, 'null score') for x in null_scores]
    if not scores:
        raise ValueError('at least one null score is required')
    value = Fraction(1+sum(x >= observed for x in scores), len(scores)+1)
    return value if exact else float(value)
