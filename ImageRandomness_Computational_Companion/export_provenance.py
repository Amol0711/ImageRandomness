"""Validate distributed bytes before resolving original experimental digests.

The export map records two different identities. An original digest names a
file used for the recorded experiment. The public digest names its distributed
representation after descriptive metadata normalization. This resolver never
claims the distributed bytes have the original digest.
"""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent

def current_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def numerical_payload(value):
    result = []
    def visit(item, path):
        if isinstance(item, dict):
            for key in sorted(item): visit(item[key], path + [key])
        elif isinstance(item, list):
            for index, child in enumerate(item): visit(child, path + [index])
        elif isinstance(item, (int, float, bool)) or item is None:
            result.append([path, item])
        elif isinstance(item, str):
            try:
                float(item)
            except ValueError:
                return
            result.append([path, item])
    visit(value, [])
    return hashlib.sha256(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()

def recorded_digest(path):
    path = Path(path).resolve()
    actual = current_digest(path)
    try:
        name = path.relative_to(ROOT).as_posix()
    except ValueError:
        return actual
    entries = json.loads((ROOT / "EXPORT_PROVENANCE.json").read_text())["files"]
    entry = entries.get(name)
    if entry is None:
        return actual
    if actual != entry["public_sha256"]:
        raise ValueError("Public-export checksum mismatch: " + name)
    return entry["original_sha256"]

def matches_recorded_digest(path, expected):
    return current_digest(path) == expected or recorded_digest(path) == expected
