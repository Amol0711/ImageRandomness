#!/usr/bin/env python3
"""Check archive integrity and small exact moment laws without writing files."""
from __future__ import annotations

from fractions import Fraction
from itertools import combinations, product
import json
from pathlib import Path
import platform
import sys

# Do not create bytecode when this entry point is invoked without -B.
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'fixed_image_and_power' / 'fixed_algorithm'))
sys.path.insert(0, str(ROOT / 'core'))

import numpy as np

from calibration import iid_baci_moments
from fixed_image_moments import fixed_baci_moments, fixed_uaci_moments
from verify_all import integrity


def literal_baci(first, second, rows, cols, q):
    """Evaluate the defining window sum, independently of the moment APIs."""
    difference = [abs(a - b) for a, b in zip(first, second)]
    numerator = 0
    for row in range(rows - 1):
        for col in range(cols - 1):
            vertices = (row * cols + col, row * cols + col + 1,
                        (row + 1) * cols + col, (row + 1) * cols + col + 1)
            numerator += sum(abs(difference[i] - difference[j])
                             for i, j in combinations(vertices, 2))
    return Fraction(numerator, 6 * (rows - 1) * (cols - 1) * (q - 1))


def moments(values):
    mean = sum(values, Fraction()) / len(values)
    variance = sum((value * value for value in values), Fraction()) / len(values) - mean * mean
    return mean, variance


def require(condition, description):
    if not condition:
        raise AssertionError(description)


def rational(value):
    return {'numerator': value.numerator, 'denominator': value.denominator}


def run_checks():
    fixed = ((0, 1, 2), (2, 0, 1))
    flat = tuple(value for row in fixed for value in row)
    q, rows, cols = 3, 2, 3
    baci_values, uaci_values = [], []
    for comparison in product(range(q), repeat=len(flat)):
        baci_values.append(literal_baci(flat, comparison, rows, cols, q))
        uaci_values.append(Fraction(sum(abs(a - b) for a, b in zip(flat, comparison)),
                                    len(flat) * (q - 1)))
    baci_expected, uaci_expected = moments(baci_values), moments(uaci_values)
    baci, uaci = fixed_baci_moments(fixed, q=q), fixed_uaci_moments(fixed, q=q)
    for name, actual, expected in (('fixed BACI', baci, baci_expected),
                                   ('fixed UACI', uaci, uaci_expected)):
        require(isinstance(actual.mean, Fraction) and isinstance(actual.variance, Fraction),
                name + ' exact return types')
        require((actual.mean, actual.variance) == expected, name + ' finite-law moments')
    for name, routine, single in (('BACI', fixed_baci_moments, baci),
                                   ('UACI', fixed_uaci_moments, uaci)):
        repeated = routine(fixed, q=q, repeats=7)
        require(repeated.mean == single.mean and repeated.variance == single.variance / 7,
                name + ' whole-image replication scaling')
    require(baci.vertex_projection_variance + baci.edge_residual_variance == baci.variance,
            'BACI variance decomposition')

    binary_images = tuple(product(range(2), repeat=4))
    iid_values = [literal_baci(first, second, 2, 2, 2)
                  for first in binary_images for second in binary_images]
    iid_expected = moments(iid_values)
    iid = iid_baci_moments(2, 2, q=2, exact=True)
    require((iid.mean, iid.variance) == iid_expected, 'random/random BACI finite-law moments')
    return {
        'fixed_reference': {
            'shape': [rows, cols], 'q': q, 'comparison_images_enumerated': len(baci_values),
            'baci_mean': rational(baci.mean), 'baci_variance': rational(baci.variance),
            'uaci_mean': rational(uaci.mean), 'uaci_variance': rational(uaci.variance),
            'replication_scaling_checked_at': 7,
            'baci_variance_decomposition_checked': True,
        },
        'random_random': {
            'shape': [2, 2], 'q': 2, 'image_pairs_enumerated': len(iid_values),
            'baci_mean': rational(iid.mean), 'baci_variance': rational(iid.variance),
        },
    }


def main():
    verified = integrity()
    checks = run_checks()
    integrity()
    report = {
        'status': 'PASS', 'source_snapshot_commit': json.loads((ROOT / 'PUBLIC_EXPORT.json').read_text())['source_snapshot_commit'],
        'files_verified': verified, 'checks': checks,
        'new_random_draws': 0, 'network_requests': 0, 'files_written': 0,
        'scope': 'Manifest verification and small independent exact finite-law checks only',
        'runtime': {'python': platform.python_version(), 'numpy': np.__version__},
    }
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
