"""Exact finite-alphabet distance moments and conditional covariance.

All public functions use fractions.Fraction. No external packages are required.
"""
from collections import Counter
from fractions import Fraction as F


def _alphabet(q):
    if not isinstance(q, int) or isinstance(q, bool) or q < 2:
        raise ValueError('q must be an integer at least two')


def _intensity(v, q):
    _alphabet(q)
    if not isinstance(v, int) or isinstance(v, bool) or not 0 <= v < q:
        raise ValueError('an intensity must be an integer in range(q)')


def pair_moments(q):
    """Return exact mean, variance, shared-endpoint covariance for iid distances."""
    _alphabet(q)
    mu = F(4 * (q**4 - 1), 15 * q**3)
    a = F((q*q-1)*(9*q**6+34*q**4+16*q*q+16), 225*q**6)
    c = F((q*q-4)*(q*q-1)*(3*q**4+20*q*q-8), 450*q**6)
    eg2 = F(7*q**6+q**4-32*q*q+24, 90*q**4)
    return {'mean': mu, 'variance': a, 'shared_covariance': c,
            'conditional_second_moment': eg2}


def conditional_pair_mean(x, q):
    """E|x-D| for the absolute difference D of independent uniform pixels."""
    _intensity(x, q)
    return F(q**3-q+(-3*q*q+2)*x+6*q*x*x-2*x**3, 3*q*q)


def uaci_coordinate_mean(v, q):
    """Mean normalized distance from a fixed intensity v to a uniform pixel."""
    _intensity(v, q)
    tau = F(2*v-(q-1), 2)
    return (4*tau*tau+q*q-1)/(4*q*(q-1))


def uaci_pooling_excess(values, q):
    """Pooled per-image variance minus fixed-coordinate variance.

    Empirical variances in this expression use divisor n, not n-1.
    """
    values = tuple(values)
    if not values:
        raise ValueError('at least one fixed intensity is required')
    for v in values:
        _intensity(v, q)
    n = len(values)
    squared_offsets = [F(2*v-(q-1), 2)**2 for v in values]
    mean_square = sum(squared_offsets, F(0))/n
    empirical_variance = sum((x-mean_square)**2 for x in squared_offsets)/n
    return empirical_variance/(n*q*q*(q-1)**2)


def fixed_difference_counts(v, q):
    """Integer frequencies of |v-U| when U runs once over range(q)."""
    _intensity(v, q)
    counts = Counter(abs(v-u) for u in range(q))
    return tuple(counts.get(d, 0) for d in range(q))


def _distance_sums(counts):
    """All sum_y counts[y]*|x-y| by two prefix sums, in linear time."""
    total = sum(counts)
    total_first = sum(y*c for y, c in enumerate(counts))
    left = left_first = 0
    out = []
    for x, count in enumerate(counts):
        out.append(x*left-left_first + total_first-left_first-x*(total-left))
        left += count
        left_first += x*count
    return out


def shared_fixed_covariance(shared, endpoint_a, endpoint_b, q):
    """Cov(|D0-D1|, |D0-D2|) for three independent uniform comparisons.

    D0=|shared-U0|, D1=|endpoint_a-U1|, D2=|endpoint_b-U2|.
    Distances are in unnormalized intensity units.
    """
    shared_counts = fixed_difference_counts(shared, q)
    ga = _distance_sums(fixed_difference_counts(endpoint_a, q))
    gb = _distance_sums(fixed_difference_counts(endpoint_b, q))
    ma = sum(p*x for p,x in zip(shared_counts, ga))
    mb = sum(p*x for p,x in zip(shared_counts, gb))
    product = sum(p*x*y for p,x,y in zip(shared_counts, ga, gb))
    return F(q*product-ma*mb, q**4)
