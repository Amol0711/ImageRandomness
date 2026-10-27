"""Reproduce deterministic fixed-image moment examples without random draws."""
from dataclasses import asdict
from fractions import Fraction
import argparse
import json
from pathlib import Path

import numpy as np
from fixed_image_moments import fixed_baci_moments, fixed_uaci_moments


def record(value):
    if isinstance(value, Fraction):
        return {'numerator': value.numerator, 'denominator': value.denominator,
                'decimal': float(value)}
    if isinstance(value, dict):
        return {key: record(item) for key, item in value.items()}
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('fixed_pattern_moments.json'))
    args = parser.parse_args()
    row, col = np.indices((256, 256))
    patterns = {
        'constant_0': np.zeros((256, 256), dtype=np.uint8),
        'constant_128': np.full((256, 256), 128, dtype=np.uint8),
        'checker_0_128': (128*((row+col) % 2)).astype(np.uint8),
        'vertical_split_0_128': (128*(col >= 128)).astype(np.uint8),
        'column_ramp_0_255': col.astype(np.uint8),
    }
    results = {'q': 256, 'shape': [256, 256], 'random_draws': 0, 'repeats': 1,
               'patterns': {}}
    for name, image in patterns.items():
        results['patterns'][name] = {
            'baci': record(asdict(fixed_baci_moments(image))),
            'uaci': record(asdict(fixed_uaci_moments(image))),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True)+'\n')
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
