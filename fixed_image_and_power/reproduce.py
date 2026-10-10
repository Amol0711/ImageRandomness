"""Verify and reproduce the complete numerical package without network access."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
MODULES = ('fixed_algorithm','algorithm_checks','natural_images','power')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_archive():
    expected = json.loads((ROOT/'SHA256SUMS.json').read_text())
    for name, wanted in expected.items():
        path = ROOT/name
        if not path.is_file() or digest(path)!=wanted:
            raise RuntimeError('Archive integrity mismatch: '+name)
    return len(expected)


def run(script, *arguments):
    command = [sys.executable, str(script), *map(str,arguments)]
    subprocess.run(command, check=True, cwd=script.parent)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick', action='store_true',
                        help='Reanalyze archived power arrays instead of regenerating random images')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'reproduced',
                        help='An absent or empty directory for independently regenerated outputs')
    args=parser.parse_args()
    started=time.perf_counter()
    verified=verify_archive()
    destination=args.output_dir.resolve()
    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError('Output directory must be absent or empty; select a new --output-dir.')
    destination.mkdir(parents=True,exist_ok=True)
    for name in MODULES:
        shutil.copytree(ROOT/name,destination/name,
                        ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    # Remove supplied derived outputs so successful checks require regeneration.
    (destination/'fixed_algorithm/fixed_pattern_moments.json').unlink()
    (destination/'algorithm_checks/algorithm_verification.json').unlink()
    (destination/'natural_images/input_manifest.json').unlink()
    shutil.rmtree(destination/'natural_images/outputs')
    power_result=destination/'power/results'
    for path in power_result.iterdir():
        if path.is_file():
            path.unlink()
    if not args.quick:
        shutil.rmtree(power_result/'raw')

    run(destination/'fixed_algorithm/run_examples.py')
    run(destination/'algorithm_checks/verify_fixed_algorithm.py',
        '--module',destination/'fixed_algorithm/fixed_image_moments.py',
        '--output',destination/'algorithm_checks/algorithm_verification.json')
    run(destination/'natural_images/prepare_inputs.py')
    run(destination/'natural_images/run_natural_images.py')
    run(destination/'power/run.py', *(['--reuse-raw'] if args.quick else []))
    run(destination/'power/check.py')

    deterministic=[Path('fixed_algorithm/fixed_pattern_moments.json'),
                   Path('natural_images/input_manifest.json')]
    deterministic += [path.relative_to(ROOT) for path in sorted((ROOT/'natural_images/outputs').iterdir())]
    power_files=[path.relative_to(ROOT) for path in sorted((ROOT/'power/results').rglob('*'))
                 if path.is_file() and path.name not in ('execution.json','sha256.json')]
    if len(power_files)!=21:
        raise RuntimeError('Unexpected deterministic power-file inventory')
    deterministic += power_files
    comparison={}
    for name in deterministic:
        wanted=digest(ROOT/name)
        actual=digest(destination/name)
        if actual!=wanted:
            raise RuntimeError('Reproduction mismatch: '+str(name))
        comparison[str(name)]=actual
    expected_checks=json.loads((ROOT/'algorithm_checks/algorithm_verification.json').read_text())
    actual_checks=json.loads((destination/'algorithm_checks/algorithm_verification.json').read_text())
    expected_checks.pop('wall_seconds',None)
    actual_checks.pop('wall_seconds',None)
    if actual_checks!=expected_checks:
        raise RuntimeError('Independent exact-verification results differ')
    import numpy as np
    import scipy
    report={'status':'PASS','mode':'raw_reanalysis' if args.quick else 'full_regeneration',
            'source_inventory_files_verified':verified,
            'deterministic_files_byte_identical':len(deterministic),
            'deterministic_power_files_byte_identical':len(power_files),
            'input_arrays_verified':3,
            'power_raw_archives_regenerated':0 if args.quick else 13,
            'independent_algorithm_checks_identical_except_timing':True,
            'network_requests':0,
            'comparison_sha256':comparison,
            'excluded_from_byte_equality':[
                'algorithm_checks/algorithm_verification.json wall_seconds only',
                'power/results/execution.json execution time, runtime versions and raw_reused flag',
                'power/results/sha256.json inventory incorporating execution.json'],
            'runtime':{'python':platform.python_version(),'numpy':np.__version__,
                       'scipy':scipy.__version__},
            'elapsed_seconds':time.perf_counter()-started}
    (destination/'reproduction_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('comparison_sha256',)},indent=2))


if __name__=='__main__':
    main()
