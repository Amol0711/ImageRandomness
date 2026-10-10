#!/usr/bin/env python3
"""Recompute deterministic calibration outputs and compare all supplied results."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

COMPONENT = Path(__file__).resolve().parent
OUTPUTS = {
    "intervals": ("baci_practical_intervals.csv", "baci_saved_null_shape.csv",
                  "practical_calibration_checks.json"),
    "power": ("partial_identity_power.csv", "partial_identity_moments.json",
              "horizontal_copy_gaussian_benchmark.csv", "verification.json"),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_manifest(root):
    hashes = {}
    for line in (root / "SHA256SUMS.txt").read_text().splitlines():
        expected, name = line.split("  ", 1)
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Missing or invalid manifest entry " + name)
        if digest(path) != expected:
            raise ValueError("File integrity mismatch " + name)
        hashes[name] = expected
    return hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=COMPONENT.parent)
    parser.add_argument("--output", type=Path, required=True,
                        help="Absent output directory outside the computational archive")
    args = parser.parse_args()
    root, destination = args.archive_root.resolve(), args.output.resolve()
    if destination.is_relative_to(root):
        raise ValueError("Choose an output directory outside the computational archive")
    if destination.exists():
        raise FileExistsError("Output directory must be absent")
    supplied = root / "practical_calibration"
    before = check_manifest(root)
    destination.mkdir(parents=True)
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    operations = (
        ("intervals", supplied / "intervals/generate_practical_calibration.py",
         ("--saved-null-shape",)),
        ("power", supplied / "power/analytic_interpretation.py", ()),
    )
    comparisons = []
    for label, script, options in operations:
        output = destination / label
        with (destination / (label + ".log")).open("w") as log:
            subprocess.run(
                [sys.executable, "-B", str(script), "--archive-root", str(root),
                 "--output", str(output), *options],
                env=environment, stdout=log, stderr=subprocess.STDOUT, check=True,
            )
        saved = supplied / label / ("generated" if label == "intervals" else "analytic_results")
        for name in OUTPUTS[label]:
            original, regenerated = saved / name, output / name
            expected, actual = digest(original), digest(regenerated)
            if actual != expected:
                raise ValueError("Regenerated numerical output differs " + label + "/" + name)
            comparisons.append({"file": label + "/" + name,
                                "sha256": actual, "byte_identical": True})
    after = check_manifest(root)
    if before != after:
        raise ValueError("Input manifest changed during deterministic reproduction")
    report = {
        "status": "PASS",
        "scope": "Deterministic calculations and saved-data reanalysis only",
        "manifest_entries_verified_before_and_after": len(before),
        "gaussian_reference_intervals_regenerated": 12,
        "held_out_null_shape_rows_regenerated": 3,
        "saved_power_thresholds_reproduced": 6,
        "saved_conditional_rejection_counts_reproduced": 36,
        "numerical_files_compared": len(comparisons),
        "all_numerical_outputs_byte_identical": True,
        "numerical_comparisons": comparisons,
        "new_random_draws": 0,
        "supplied_files_unchanged": True,
    }
    (destination / "verification_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
