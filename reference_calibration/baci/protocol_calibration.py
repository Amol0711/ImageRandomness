"""Reproduce BACI aggregate null-calibration calculations using Python's stdlib.

Scores B are proportions. Gaussian probabilities below are surrogate probabilities,
not finite-image guarantees. Count bounds are exact conditional on the stated
independence and marginal rejection assumptions. No cipher measurements are used.
"""
from __future__ import annotations

import csv
import json
from fractions import Fraction
from math import comb, erf, fsum, prod, sqrt
from pathlib import Path
from statistics import NormalDist


def exact_moments(q=256, M=256, N=256):
    """Independent integer-support evaluation, without a calibration API import."""
    if any(type(x) is not int for x in (q, M, N)) or min(q, M, N) < 2:
        raise ValueError('q, M and N must be integers at least two')
    weights = [q] + [2 * (q - x) for x in range(1, q)]
    assert sum(weights) == q*q
    g_numerators = [sum(w*abs(x-y) for y, w in enumerate(weights)) for x in range(q)]
    mu = Fraction(sum(w*g for w, g in zip(weights, g_numerators)), q**4)
    second = Fraction(sum(wx*wy*(x-y)**2 for x, wx in enumerate(weights)
                          for y, wy in enumerate(weights)), q**4)
    A = second - mu*mu
    C = Fraction(sum(w*g*g for w, g in zip(weights, g_numerators)), q**6) - mu*mu
    K = (M-1)*(N-1)
    S2 = 10*M*N-12*M-12*N+14
    W = 62*M*N-96*M-96*N+148
    return dict(q=q, M=M, N=N, K=K, S2=S2, W=W, mu_pair=mu, A=A, C=C,
                mean=mu/(q-1), variance=(A*S2+2*C*W)/(36*K*K*(q-1)**2),
                pair_variance=A/(6*K*(q-1)**2))


def poisson_binomial(probs):
    """Coefficient vector of product ((1-p_j)+p_j*t), including 0 and 1."""
    out = [1.0]
    for p in probs:
        if not 0 <= p <= 1:
            raise ValueError('probabilities must be in [0, 1]')
        nxt = [0.0]*(len(out)+1)
        for k, weight in enumerate(out):
            nxt[k] += weight*(1-p)
            nxt[k+1] += weight*p
        out = nxt
    return out


def count_tail(probs, threshold):
    """P(sum independent Bernoulli(probs) >= threshold)."""
    return fsum(poisson_binomial(probs)[max(0, threshold):]) if threshold > 0 else 1.0


def aggregate_cutoff(m, alpha, eta):
    """Smallest c in 1,...,m+1 with P(Bin(m,alpha)>=c)<=eta.

    Reject when at least c component tests reject; accept when at least m-c+1
    pass. c=m+1 is allowed and means the aggregate never rejects.
    """
    if type(m) is not int or m < 1 or not 0 <= alpha <= 1 or not 0 <= eta < 1:
        raise ValueError('invalid m, alpha or eta')
    # Decide the cutoff in rational arithmetic, interpreting input decimals
    # exactly. This also handles equality at the requested tail bound.
    a, target = Fraction(str(alpha)), Fraction(str(eta))
    law = [Fraction(comb(m,k))*a**k*(1-a)**(m-k) for k in range(m+1)]
    for c in range(1, m+2):
        bound = sum(law[c:], Fraction(0))
        if bound <= target:
            return c, float(bound)
    raise AssertionError('unreachable')


def strict_majority(passes, m):
    return passes > m/2


def interval_pass(value, mean, halfwidth):
    """Strict source convention, including rejection at either boundary."""
    return abs(value-mean) < halfwidth


def relative_pass(value, mean, epsilon=0.001):
    if mean <= 0:
        raise ValueError('positive null mean required for relative error')
    return abs(value-mean)/mean < epsilon


def gaussian_pass(z):
    return erf(z/sqrt(2))


def joint_rule_probabilities(pass_probs, tolerance_probs, threshold=None):
    """Same observations, two centered thresholds for each independent score.

    A = at least threshold interval passes. B = every relative-error tolerance
    passes. The four elementary probabilities are obtained by nested intervals
    per coordinate; the nesting direction may differ between coordinates.
    Returns joint and marginal probabilities using a direct dynamic program.
    """
    if len(pass_probs) != len(tolerance_probs) or not pass_probs:
        raise ValueError('equal nonempty probability sequences required')
    m = len(pass_probs)
    threshold = m//2+1 if threshold is None else threshold
    if type(threshold) is not int or not 0 <= threshold <= m+1:
        raise ValueError('invalid pass-count threshold')
    # Key (number of interval passes, whether every tolerance has passed).
    state = {(0, True): 1.0}
    for p, t in zip(pass_probs, tolerance_probs):
        if not (0 <= p <= 1 and 0 <= t <= 1):
            raise ValueError('probabilities must be in [0, 1]')
        categories = [(1, True, min(p, t)), (1, False, max(p-t, 0.0)),
                      (0, True, max(t-p, 0.0)), (0, False, 1-max(p, t))]
        nxt = {}
        for (k, all_t), mass in state.items():
            for passed, tol, probability in categories:
                key = (k+passed, all_t and tol)
                nxt[key] = nxt.get(key, 0.0)+mass*probability
        state = nxt
    A_and_B = fsum(mass for (k, all_t), mass in state.items() if k >= threshold and all_t)
    neither = fsum(mass for (k, all_t), mass in state.items() if k < threshold and not all_t)
    return dict(interval_accept=fsum(v for (k,t),v in state.items() if k >= threshold),
                interval_reject=fsum(v for (k,t),v in state.items() if k < threshold),
                tolerance_accept=fsum(v for (k,t),v in state.items() if t),
                tolerance_reject=fsum(v for (k,t),v in state.items() if not t),
                intersection_accept=A_and_B,
                intersection_reject=fsum(v for (k,t),v in state.items() if k < threshold or not t),
                union_accept=fsum(v for (k,t),v in state.items() if k >= threshold or t),
                union_reject=neither)


def _csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def generate(outdir=None):
    outdir = Path(outdir) if outdir else Path(__file__).parent/'data'
    outdir.mkdir(parents=True, exist_ok=True)
    constants = exact_moments()
    mu, V, Vpair = (float(constants[k]) for k in ('mean', 'variance', 'pair_variance'))
    rho = V/Vpair
    encoded = {k: {'rational': str(v), 'decimal': float(v)} if isinstance(v, Fraction) else v
               for k,v in constants.items()}
    encoded['variance_ratio'] = rho
    encoded['probability_model'] = 'Independent centered Gaussian surrogate with exact BACI moments'
    (outdir/'moments.json').write_text(json.dumps(encoded, indent=2)+'\n')
    intervals, majority, composite, cutoffs = [], [], [], []
    for R in (1, 10, 50, 100):
        for alpha in (0.01, 0.05):
            z = NormalDist().inv_cdf(1-alpha/2)
            for rule, variance in [('pair_independence', Vpair), ('full_covariance', V)]:
                h = z*sqrt(variance/R)
                p = gaussian_pass(z*sqrt(variance/V))
                intervals.append(dict(q=256, M=256, N=256, R=R, alpha=alpha, rule=rule,
                                      mean_percent=100*mu, lower_percent=100*(mu-h),
                                      upper_percent=100*(mu+h), halfwidth_percentage_points=100*h,
                                      gaussian_single_pass_probability=p))
                for m in (11, 15, 20, 100):
                    law = poisson_binomial([1-p]*m)
                    # A strict majority of passes means failures <= m-ceil((m+1)/2).
                    first_reject = m-(m//2+1)+1
                    rej = fsum(law[first_reject:])
                    majority.append(dict(R=R, alpha=alpha, rule=rule, m=m,
                                         minimum_passes=m//2+1, gaussian_single_pass_probability=p,
                                         gaussian_majority_pass_probability=fsum(law[:first_reject]),
                                         gaussian_majority_rejection_probability=rej))
                    if R in (50, 100):
                        eps = 0.001
                        b = eps*mu*sqrt(R/V)
                        t = gaussian_pass(b)
                        joint = joint_rule_probabilities([p]*m, [t]*m)
                        composite.append(dict(R=R, alpha=alpha, rule=rule, m=m,
                                              epsilon=eps, standardized_interval_radius=z*sqrt(variance/V),
                                              standardized_tolerance_radius=b,
                                              gaussian_single_tolerance_pass_probability=t,
                                              **{f'gaussian_{k}':v for k,v in joint.items()}))
    for m in (15,20,100):
        for alpha in (0.01,0.05):
            for eta in (0.01,0.05):
                c,bound = aggregate_cutoff(m,alpha,eta)
                cutoffs.append(dict(m=m, per_test_rejection_upper_bound=alpha,
                                    aggregate_rejection_target=eta, reject_if_failures_at_least=c,
                                    accept_if_passes_at_least=m-c+1,
                                    exact_binomial_rejection_bound=bound,
                                    previous_cutoff_rejection_bound=count_tail([alpha]*m,c-1)))
    _csv(outdir/'gaussian_intervals.csv', intervals)
    _csv(outdir/'gaussian_strict_majority.csv', majority)
    _csv(outdir/'gaussian_joint_rules.csv', composite)
    _csv(outdir/'independent_count_cutoffs.csv', cutoffs)
    return dict(interval_rows=len(intervals), majority_rows=len(majority),
                joint_rows=len(composite), cutoff_rows=len(cutoffs), variance_ratio=rho)


if __name__ == '__main__':
    print(json.dumps(generate(), indent=2))
