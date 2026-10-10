#!/usr/bin/env python3
"""Verify supplied inputs and run component checks in a fresh output directory."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def integrity():
    count = 0
    for line in (ROOT / 'SHA256SUMS.txt').read_text().splitlines():
        expected, name = line.split('  ', 1)
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ValueError('Missing or invalid manifest entry ' + name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('File integrity mismatch ' + name)
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True,
                        help='Absent output directory outside this archive')
    parser.add_argument('--engine', action='store_true',
                        help='Compile and replay selected original expanded trials')
    parser.add_argument('--regenerate-addenda', action='store_true',
                        help='Regenerate full entropy moments and all newer power arrays')
    args = parser.parse_args()
    destination = args.output.resolve()
    if destination.is_relative_to(ROOT):
        raise ValueError('Choose an output directory outside the supplied archive')
    if destination.exists():
        raise FileExistsError('Output directory must be absent')
    verified = integrity()
    destination.mkdir(parents=True)
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    operations = []

    def run(label, script, *options, cwd=None):
        print('Running ' + label, flush=True)
        with (destination / (label + '.log')).open('w') as log:
            subprocess.run([sys.executable, '-B', str(script), *map(str, options)],
                           cwd=cwd or script.parent, env=environment,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        operations.append(label)

    run('original_verification', ROOT / 'core/verify.py',
        *(['--engine'] if args.engine else []))
    reference = destination / 'reference_calibration'
    shutil.copytree(ROOT / 'reference_calibration', reference)
    if args.regenerate_addenda:
        run('reference_calibration_full', reference / 'reproduce.py')
    else:
        run('aggregate_reference_check', reference / 'baci/check_protocol_calibration.py')
        run('entropy_saved_table_check', reference / 'entropy/check.py')
    windows = destination / 'window_reference'
    shutil.copytree(ROOT / 'window_reference', windows)
    run('window_reference_full', windows / 'reproduce.py')
    run('fixed_image_and_power', ROOT / 'fixed_image_and_power/reproduce.py',
        *([] if args.regenerate_addenda else ['--quick']),
        '--output-dir', destination / 'fixed_image_and_power')
    run('practical_calibration', ROOT / 'practical_calibration/reproduce.py',
        '--archive-root', ROOT, '--output', destination / 'practical_calibration')
    run('supplementary_results', ROOT / 'supplementary_results/check.py',
        '--archive-root', ROOT, '--output', destination / 'supplementary_results')
    integrity()  # Component commands must leave every distributed file intact.
    import numpy
    import scipy
    import mpmath
    report = {
        'status': 'PASS',
        'files_verified': verified,
        'operations': operations,
        'original_baseline_reanalyzed': True,
        'original_baseline_full_regeneration': False,
        'original_expanded_reanalyzed': True,
        'original_expanded_selected_replay': args.engine,
        'original_expanded_full_regeneration': False,
        'reference_entropy_full_moment_regeneration': args.regenerate_addenda,
        'window_reference_full_deterministic_regeneration': True,
        'power_raw_arrays_regenerated': 13 if args.regenerate_addenda else 0,
        'power_archived_array_reanalysis': not args.regenerate_addenda,
        'practical_calibration_full_deterministic_regeneration': True,
        'practical_calibration_numerical_outputs_byte_compared': 7,
        'practical_calibration_new_random_draws': 0,
        'supplementary_results_exports_regenerated_and_compared': True,
        'supplementary_results_new_random_draws': 0,
        'supplementary_figures_rebuilt': False,
        'supplied_files_unchanged_after_verification': True,
        'network_requests': 0,
        'runtime': {'python': platform.python_version(), 'numpy': numpy.__version__,
                    'scipy': scipy.__version__, 'mpmath': mpmath.__version__},
    }
    (destination / 'verification_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
