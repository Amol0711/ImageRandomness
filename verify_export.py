"""Verify the public files against the recorded export inventory."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def verify():
    record = json.loads((ROOT / 'PUBLIC_EXPORT.json').read_text())
    counts = {'retained_source_files': 0, 'unchanged_source_files': 0,
              'modified_usage_or_output_files': 0}
    for item in record['files']:
        path = ROOT / item['path']
        if not path.resolve().is_relative_to(ROOT) or not path.is_file():
            raise ValueError('Missing or invalid file ' + item['path'])
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != item['public_sha256']:
            raise ValueError('Export checksum mismatch ' + item['path'])
        counts['retained_source_files'] += 1
        if observed == item['source_sha256']:
            counts['unchanged_source_files'] += 1
        else:
            counts['modified_usage_or_output_files'] += 1
    print(json.dumps({'status': 'PASS', **counts,
                     'scope': 'Distributed files match their public digests. Source digests identify the source snapshot; this check does not reconstruct omitted source files.'}, indent=2))
    return counts


if __name__ == '__main__':
    verify()
