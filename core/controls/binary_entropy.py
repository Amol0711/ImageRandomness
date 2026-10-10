#!/usr/bin/env python3
"""Exact numerical control for the mean entropy of 30 binary two-pixel blocks."""
import argparse
import csv
from fractions import Fraction
import json
from math import comb, sqrt
from pathlib import Path
from statistics import NormalDist


def compute():
    blocks = 30
    pmf = [Fraction(comb(blocks, j), 2**blocks) for j in range(blocks + 1)]
    z = NormalDist().inv_cdf(0.975)
    intervals = {
        'linear_scaling': [0.5-z*0.5/blocks, 0.5+z*0.5/blocks],
        'square_root_scaling': [0.5-z*0.5/sqrt(blocks), 0.5+z*0.5/sqrt(blocks)],
    }
    acceptance = {name: [j for j in range(blocks+1) if lo <= j/blocks <= hi]
                  for name, (lo, hi) in intervals.items()}
    rejection = {name: 1-sum((pmf[j] for j in values), Fraction())
                 for name, values in acceptance.items()}
    assert sum(pmf) == 1
    assert acceptance['linear_scaling'] == [15]
    assert acceptance['square_root_scaling'] == list(range(10, 21))
    assert rejection['linear_scaling'] == Fraction(57414019, 67108864)
    assert rejection['square_root_scaling'] == Fraction(22964087, 536870912)
    summary = {'alphabet': 2, 'pixels_per_block': 2, 'blocks_per_trial': blocks,
               'alpha': 0.05, 'intervals': intervals, 'accepted_counts': acceptance,
               'exact_rejection_probabilities': {k: str(v) for k, v in rejection.items()},
               'rejection_probabilities': {k: float(v) for k, v in rejection.items()}}
    return pmf, acceptance, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    pmf, acceptance, summary = compute()
    with (args.output/'pmf.csv').open('w', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(['count', 'mean_entropy', 'probability', 'linear_accept', 'sqrt_accept'])
        for j, probability in enumerate(pmf):
            writer.writerow([j, format(j/30, '.17g'), format(float(probability), '.17g'),
                             int(j in acceptance['linear_scaling']),
                             int(j in acceptance['square_root_scaling'])])
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print('Exact binary-entropy control passed.')


if __name__ == '__main__':
    main()
