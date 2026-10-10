#!/usr/bin/env python3
"""Verify distributed files and reproduce numerical summaries without overwriting data."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent


def integrity():
    manifest = ROOT/'SHA256SUMS.txt'
    if not manifest.is_file():
        raise FileNotFoundError('SHA256SUMS.txt is required')
    count = 0
    for line in manifest.read_text().splitlines():
        digest, name = line.split('  ', 1)
        path = ROOT/name
        if not path.resolve().is_relative_to(ROOT) or not path.is_file():
            raise ValueError(f'Invalid or missing manifest member {name}')
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != digest:
            raise ValueError(f'Checksum mismatch for {name}')
        count += 1
    print(f'Integrity passed for {count} files.', flush=True)


def call(*args):
    subprocess.run([sys.executable, '-B', *args], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', action='store_true',
                        help='Also compile and test the C++ simulation engine')
    args = parser.parse_args()
    integrity()
    call('-m', 'unittest', 'discover', '-s', 'tests')
    call('baseline/run.py', 'verify')
    call('expanded/run.py', 'verify')
    with tempfile.TemporaryDirectory(prefix='entropy-check-') as tmp:
        destination = Path(tmp)/'result'
        call('controls/binary_entropy.py', '--output', str(destination))
        expected = ROOT/'controls/expected'
        for name in ['pmf.csv', 'summary.json']:
            if (destination/name).read_bytes() != (expected/name).read_bytes():
                raise ValueError(f'Binary entropy checkpoint differs for {name}')
    if args.engine:
        call('expanded/run.py', 'check')
    print(json.dumps({'status': 'PASS', 'engine_checks': args.engine,
                      'full_expanded_simulation_rerun': False}))


if __name__ == '__main__':
    main()
