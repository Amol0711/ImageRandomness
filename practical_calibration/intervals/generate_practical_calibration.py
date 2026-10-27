"""Evaluate Gaussian BACI reference intervals and saved-null shape diagnostics.

Only deterministic computations are performed. Scores are proportions, while
the interval endpoint columns explicitly labelled percent use 100 times the
score. Exact finite-grid moments do not make the Gaussian intervals exact
finite-sample acceptance regions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from collections import Counter
from fractions import Fraction
from itertools import combinations
from pathlib import Path
from statistics import NormalDist
import sys


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def independent_moments(q, m, n):
    """Polynomial finite-alphabet moments and independent degree-count identity."""
    mu_pair = Fraction(4*(q**4-1), 15*q**3)
    a = Fraction(9*q**8+25*q**6-18*q**4-16, 225*q**6)
    c = Fraction((q*q-4)*(q*q-1)*(3*q**4+20*q*q-8), 450*q**6)
    # A coordinate is in one interval at a boundary and two in the interior.
    # Every 2x2 window supplies three incident edges at each of its vertices.
    axis_squared_occurrences = lambda d: 2 + 4*(d-2)
    degree_squared_sum = 9*axis_squared_occurrences(m)*axis_squared_occurrences(n)
    # Horizontal/vertical edges have multiplicities one at the two outer
    # boundaries and two in the interior; both diagonals have multiplicity one.
    gamma2 = ((n-1)*(2+4*(m-2)) + (m-1)*(2+4*(n-2))
              + 2*(m-1)*(n-1))
    k = (m-1)*(n-1)
    variance = ((a-2*c)*gamma2 + c*degree_squared_sum)/(6*k*(q-1))**2
    return dict(mean=mu_pair/(q-1), variance=variance, A=a, C=c,
                gamma2=gamma2, degree_squared_sum=degree_squared_sum)


def small_grid_check():
    """Check count identities against literal overlapping-window enumeration."""
    checked = []
    for m, n in ((2,2), (2,3), (3,2), (3,3), (4,7), (7,4)):
        weights = Counter()
        for i in range(m-1):
            for j in range(n-1):
                vertices = (i*n+j, i*n+j+1, (i+1)*n+j, (i+1)*n+j+1)
                weights.update(combinations(vertices, 2))
        degrees = Counter()
        for (u,v), weight in weights.items():
            degrees[u] += weight
            degrees[v] += weight
        check = independent_moments(256, m, n)
        assert sum(c*c for c in weights.values()) == check['gamma2']
        assert sum(d*d for d in degrees.values()) == check['degree_squared_sum']
        checked.append([m,n])
    return checked


def saved_shape_diagnostics(archive_root, output):
    import numpy as np
    from scipy.stats import kurtosis, skew

    source = archive_root/'core/baseline/data/null_samples.npz'
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    rows = []
    with np.load(source, allow_pickle=False) as arrays:
        for side in (16,64,256):
            key = f'BACI_{side}x{side}_R1_baci_fresh'
            values = arrays[key]
            n = len(values)
            assert n == 5000 and values.ndim == 1
            assert np.all(np.isfinite(values))
            centered = values-values.mean()
            m2, m3, m4 = (float(np.mean(centered**p)) for p in (2,3,4))
            g1 = math.sqrt(n*(n-1))/(n-2)*m3/m2**1.5
            g2 = (n-1)/((n-2)*(n-3))*((n+1)*m4/m2**2-3*(n-1))
            assert math.isclose(g1, float(skew(values, bias=False)), abs_tol=1e-13)
            assert math.isclose(g2, float(kurtosis(values, fisher=True, bias=False)), abs_tol=1e-13)
            se_g1 = math.sqrt(6*n*(n-1)/((n-2)*(n+1)*(n+3)))
            se_g2 = math.sqrt(24*n*(n-1)**2/((n-3)*(n-2)*(n+3)*(n+5)))
            rows.append(dict(M=side,N=side,R=1,q=256,array_key=key,n=n,
                             bias_corrected_skewness=g1,bias_corrected_excess_kurtosis=g2,
                             gaussian_reference_se_skewness=se_g1,
                             gaussian_reference_se_excess_kurtosis=se_g2))
    write_csv(output/'baci_saved_null_shape.csv', rows)
    return dict(source='core/baseline/data/null_samples.npz', sha256=source_hash,
                source_role='independent fresh evaluation observations only; calibration banks excluded',
                estimators='bias-corrected Fisher-Pearson skewness and Fisher excess kurtosis',
                uncertainty='standard errors are exact Gaussian-reference values; not estimated errors under the BACI law',
                interpretation='central-shape diagnostic; does not certify finite-grid tail accuracy',
                scipy_independent_estimator_checks=6, rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True,
                        help='Absent output directory outside the computational archive')
    parser.add_argument('--saved-null-shape', action='store_true')
    args = parser.parse_args()
    root, output = args.archive_root.resolve(), args.output.resolve()
    if output.is_relative_to(root):
        raise ValueError('Choose an output directory outside the computational archive')
    if output.exists():
        raise FileExistsError('Output directory must be absent')
    output.mkdir(parents=True)
    exact = load_module('practical_protocol_calibration',
                        root/'reference_calibration/baci/protocol_calibration.py')
    api = load_module('practical_calibration_core', root/'core/calibration/core.py')
    from scipy.stats import norm

    q = 256
    normal = NormalDist()
    rows, moments, max_endpoint_error = [], [], 0.0
    for side in (256,512,1024):
        source = exact.exact_moments(q=q,M=side,N=side)
        independent = independent_moments(q,side,side)
        for key in ('mean','variance','A','C'):
            assert source[key] == independent[key]
        assert source['S2'] == independent['gamma2']
        assert 2*source['S2']+2*source['W'] == independent['degree_squared_sum']
        moments.append(dict(M=side,N=side,q=q,mean_fraction=str(source['mean']),
                            per_image_variance_fraction=str(source['variance']),
                            full_to_pair_variance_ratio=float(source['variance']/source['pair_variance'])))
        for repeats in (1,100):
            api_moments = api.iid_baci_moments(side,side,q,repeats=repeats,exact=True)
            variance = source['variance']/repeats
            assert api_moments.mean == source['mean']
            assert api_moments.variance == variance
            for alpha in (.05,.01):
                z = normal.inv_cdf(1-alpha/2)
                se = math.sqrt(float(variance))
                mean = float(source['mean'])
                lower, upper = mean-z*se, mean+z*se
                # Separate quantile implementation and rational-moment route.
                z_check = float(norm.ppf(1-alpha/2))
                se_check = math.sqrt(float(independent['variance']/repeats))
                alternate = (float(independent['mean'])-z_check*se_check,
                             float(independent['mean'])+z_check*se_check)
                err = max(abs(lower-alternate[0]),abs(upper-alternate[1]))
                max_endpoint_error = max(max_endpoint_error,err)
                assert err < 1e-14
                assert abs((lower+upper)/2-mean) < 1e-15
                rows.append(dict(M=side,N=side,q=q,window_width=2,R=repeats,
                                 alpha=alpha,mean_proportion=mean,
                                 variance_proportion=float(variance),
                                 standard_error_proportion=se,
                                 lower_proportion=lower,upper_proportion=upper,
                                 mean_percent=100*mean,
                                 standard_error_percentage_points=100*se,
                                 lower_percent=100*lower,upper_percent=100*upper))
    assert len(rows) == 12
    for side in (256,512,1024):
        for alpha in (.05,.01):
            pair = [row for row in rows if row['M']==side and row['alpha']==alpha]
            assert [row['R'] for row in pair] == [1,100]
            assert math.isclose(pair[0]['standard_error_proportion'],
                                10*pair[1]['standard_error_proportion'],rel_tol=1e-15)
            assert pair[0]['variance_proportion'] == 100*pair[1]['variance_proportion'] or math.isclose(
                pair[0]['variance_proportion'],100*pair[1]['variance_proportion'],rel_tol=1e-15)
    write_csv(output/'baci_practical_intervals.csv', rows)
    checks = dict(status='PASS',model='two independent iid uniform images',q=q,
                  window_width=2,interval_method='two-sided Gaussian using exact finite-grid moments',
                  units='proportion unless explicitly percent; standard errors in percentage points where labelled',
                  distinct_designs=6,gaussian_intervals=12,independent_endpoint_checks=24,
                  maximum_independent_endpoint_difference_proportion=max_endpoint_error,
                  exact_per_image_moment_comparisons=18,exact_api_mean_variance_comparisons=12,
                  symmetric_mean_checks=12,independent_repetition_scaling_checks=6,
                  literal_grid_count_checks=small_grid_check(),moments=moments,
                  scope='variance ratios belong to the random/random design; no universal fixed-image multiplier is used')
    if args.saved_null_shape:
        checks['saved_null_shape'] = saved_shape_diagnostics(root,output)
    (output/'practical_calibration_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',gaussian_intervals=len(rows),
                          output=str(output),shape_rows=3 if args.saved_null_shape else 0)))


if __name__ == '__main__':
    main()
