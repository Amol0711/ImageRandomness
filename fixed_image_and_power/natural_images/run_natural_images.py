"""Evaluate fixed-image conditional moments and a fixed permutation control.

Run from any directory. Fractions are retained until CSV decimal formatting.
Default output is ./outputs beside this script. No Monte Carlo is involved.
"""
from pathlib import Path
import argparse
import csv
from dataclasses import asdict
from fractions import Fraction
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent/'fixed_algorithm'))
from fixed_image_moments import fixed_baci_moments, fixed_uaci_moments
from prepare_inputs import load_inputs, permutation_indices, sha256


def fraction_text(x):
    return str(x) if isinstance(x, Fraction) else x


def independent_uaci(image, q=256):
    """Direct enumeration of every uniform comparison value, by histogram."""
    histogram = np.bincount(image.ravel(), minlength=q)
    n = image.size
    first = [sum(abs(a-u) for u in range(q)) for a in range(q)]
    second = [sum((a-u)**2 for u in range(q)) for a in range(q)]
    total_first = sum(int(histogram[a])*first[a] for a in range(q))
    total_first_square = sum(int(histogram[a])*first[a]**2 for a in range(q))
    total_variance_numerator = sum(int(histogram[a])*(q*second[a]-first[a]**2) for a in range(q))
    mean = Fraction(total_first, n*q*(q-1))
    variance = Fraction(total_variance_numerator, n*n*q*q*(q-1)**2)
    excess = Fraction(n*total_first_square-total_first**2, n**3*q*q*(q-1)**2)
    return mean, variance, excess


def independent_window_mean_and_pair(image, q=256):
    """Sum all six pair occurrences per window using unreflected laws.

    This independent calculation builds each conditional distribution by
    enumerating the q source pixel values. It uses no graph multiplicity,
    prefix-sum conditional mean, reflected class, or projection routine.
    """
    counts = np.array([np.bincount(np.abs(a-np.arange(q)), minlength=q) for a in range(q)], dtype=np.int64)
    distance = np.abs(np.arange(q)[:, None]-np.arange(q)[None, :])
    mean_numerators = counts @ distance @ counts.T
    second_numerators = counts @ (distance**2) @ counts.T
    variance_numerators = q*q*second_numerators-mean_numerators**2
    views = (image[:-1, :-1], image[:-1, 1:], image[1:, :-1], image[1:, 1:])
    mean_total = 0
    variance_total = 0
    for i in range(4):
        for j in range(i+1, 4):
            mean_total += sum(map(int, mean_numerators[views[i], views[j]].ravel()))
            variance_total += sum(map(int, variance_numerators[views[i], views[j]].ravel()))
    normalization = 6*(image.shape[0]-1)*(image.shape[1]-1)*(q-1)
    return Fraction(mean_total, q*q*normalization), Fraction(variance_total, q**4*normalization**2)


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    inputs = load_inputs()
    permutation = permutation_indices()
    rows, exact_records = [], {}
    checks = {'histogram_pairs': 0, 'uaci_direct_enumerations': 0,
              'baci_direct_window_enumerations': 0, 'projection_decompositions': 0,
              'exact_uaci_permutation_equalities': 0}
    for name, original in inputs.items():
        previous_uaci = None
        for arrangement, image in [('original', original), ('permuted', original.ravel()[permutation].reshape(original.shape))]:
            b = fixed_baci_moments(image, q=256, exact=True)
            u = fixed_uaci_moments(image, q=256, exact=True)
            assert b.variance == b.vertex_projection_variance+b.edge_residual_variance
            assert b.variance > 0 and b.edge_residual_variance > 0
            checks['projection_decompositions'] += 1
            assert (u.mean, u.variance, u.pooled_excess) == independent_uaci(image)
            checks['uaci_direct_enumerations'] += 1
            assert (b.mean, b.pair_occurrence_variance) == independent_window_mean_and_pair(image)
            checks['baci_direct_window_enumerations'] += 1
            if previous_uaci is not None:
                assert u == previous_uaci
                checks['exact_uaci_permutation_equalities'] += 1
                assert np.array_equal(np.bincount(original.ravel(), minlength=256), np.bincount(image.ravel(), minlength=256))
                checks['histogram_pairs'] += 1
            previous_uaci = u
            row = {'image': name, 'arrangement': arrangement, 'rows': 256, 'columns': 256,
                   'alphabet_size': 256, 'baci_mean': b.mean, 'baci_variance': b.variance,
                   'baci_pair_occurrence_variance': b.pair_occurrence_variance,
                   'baci_independent_edge_variance': b.independent_edge_variance,
                   'baci_full_to_pair': b.variance/b.pair_occurrence_variance,
                   'baci_full_to_independent_edge': b.variance/b.independent_edge_variance,
                   'uaci_mean': u.mean, 'uaci_fixed_variance': u.variance,
                   'uaci_pooled_excess': u.pooled_excess, 'uaci_pooled_variance': u.pooled_variance,
                   'uaci_pooled_to_fixed': u.pooled_variance/u.variance}
            rows.append(row)
            exact_records[name+'_'+arrangement] = {
                'input_pixel_sha256': sha256(image.tobytes(order='C')),
                'baci': {k:fraction_text(v) for k,v in asdict(b).items()},
                'uaci': {k:fraction_text(v) for k,v in asdict(u).items()},
                'ratios': {k:str(v) for k,v in row.items() if '_to_' in k}}
            print(name, arrangement, 'BACI', float(b.mean), float(b.variance), float(b.variance/b.pair_occurrence_variance), 'UACI ratio', float(u.pooled_variance/u.variance))
    with (output/'natural_image_moments.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows({k:format(float(v), '.17g') if isinstance(v,Fraction) else v for k,v in r.items()} for r in rows)
    (output/'natural_image_moments_exact.json').write_text(json.dumps(exact_records, indent=2)+'\n')
    (output/'verification.json').write_text(json.dumps({'status': 'PASS', 'checks': checks,
        'normalization': 'BACI and UACI are fractions in [0,1]; variances are in squared fraction units.',
        'method': 'Exact finite sums with integer arithmetic and rational final values; decimal outputs are rounded once.',
        'sampling': 'None. Fixed original crops and a single predetermined permutation of each crop.'}, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'outputs')
    args = parser.parse_args()
    run(args.output_dir)
