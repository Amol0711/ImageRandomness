"""Audit saved scaling observations without importing or running the algorithms.

Checks exact recorded outputs, arithmetic summaries, grid identities, censoring
and source provenance. This does not reproduce measurements or certify runtime
stability, maximum intermediate integer size, or process resident memory.
"""
from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from export_provenance import recorded_digest, matches_recorded_digest
import statistics


ROOT = Path(__file__).resolve().parent.parent
METHODS = {"piecewise_exact", "piecewise_cached", "alphabet_exact", "alphabet_float64"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collection_digest(mapping):
    raw = json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def fraction(value):
    if isinstance(value, dict):
        return Fraction(int(value["numerator"]), int(value["denominator"]))
    return Fraction.from_float(float(value))


class Audit:
    def __init__(self):
        self.checks = Counter()
        self.failures = []
        self.counts = Counter()

    def check(self, condition, group, detail):
        self.checks[group] += 1
        if not condition:
            self.failures.append({"check": group, "detail": detail})

    def equal(self, actual, expected, group, detail):
        self.check(actual == expected, group, detail)

    def summary(self, row, repeats, label):
        calls = row.get("calls", [])
        completed = [c for c in calls if c["status"] == "completed"]
        times = [c["elapsed_seconds"] for c in completed]
        self.check(all(isinstance(t, (int, float)) and math.isfinite(t) and t >= 0
                       for t in times), "timing_arithmetic", label + " elapsed times")
        self.equal(row.get("completed_repetitions"), len(completed),
                   "timing_arithmetic", label + " completed count")
        self.equal(row.get("timing_replicates_seconds"), times,
                   "timing_arithmetic", label + " retained times")
        self.equal(row.get("censored_repetitions"),
                   sum(c["status"].startswith("censored_") for c in calls),
                   "timing_arithmetic", label + " censored count")
        self.equal(len({c["repetition"] for c in calls}), len(calls),
                   "timing_arithmetic", label + " unique repetitions")
        full = len(completed) == repeats and len(calls) == repeats
        self.equal(row.get("status") == "completed", full,
                   "timing_arithmetic", label + " completed status")
        for key, operation in (("median", statistics.median), ("min", min), ("max", max)):
            self.equal(row.get(key + "_seconds"), operation(times) if full else None,
                       "timing_arithmetic", label + " " + key)
        self.equal(row.get("partial_completed_median_seconds"),
                   statistics.median(times) if times and not full else None,
                   "timing_arithmetic", label + " partial median")
        if full:
            self.equal(sorted(c["repetition"] for c in calls), list(range(repeats)),
                       "timing_arithmetic", label + " repetition indices")
        return completed

    def output(self, call, method, targets, q, target_error, label):
        errors = {}
        for key, power in (("mean", 2), ("variance", 4)):
            value = fraction(call[key + "_output"])
            difference = value - targets[key]
            relative = float(difference / targets[key]) if targets[key] else float(difference)
            errors[key] = relative
            self.equal(fraction(call[key + "_absolute_error_exact"]), abs(difference),
                       "output_errors", label + " " + key + " exact absolute error")
            self.equal(call[key + "_relative_error"], relative,
                       "output_errors", label + " " + key + " signed error")
            self.equal(call[key + "_error_scale"],
                       "relative" if targets[key] else "absolute_zero_oracle",
                       "output_errors", label + " " + key + " error scale")
            if method != "alphabet_float64":
                self.check(isinstance(call[key + "_output"], dict),
                           "exact_outputs", label + " rational encoding")
                self.equal(value, targets[key], "exact_outputs", label + " " + key)
                self.equal(call.get("exact_" + key + "_equal"), value == targets[key],
                           "exact_outputs", label + " " + key + " equality flag")
                numerator = value * q**power
                self.equal(numerator.denominator, 1, "unreduced_bits", label + " integer numerator")
                bits = abs(numerator.numerator).bit_length()
                self.equal(call["unreduced_" + key + "_numerator_bits"], bits,
                           "unreduced_bits", label + " " + key)
                diagnostic = call.get("diagnostics", {}).get(key + "_numerator_bits")
                if diagnostic is not None:
                    self.equal(diagnostic, bits, "unreduced_bits", label + " diagnostic " + key)
        self.equal(call.get("variance_error_target_met"), abs(errors["variance"]) <= target_error,
                   "output_errors", label + " variance target flag")
        return errors


def grid_counts(spec):
    """Count neighbors from clipped coordinate spans, without graph enumeration."""
    rows, columns, height, width = (spec[k] for k in ("rows", "columns", "height", "width"))
    vertical = Counter(min(rows - 1, r + height - 1) - max(0, r - height + 1) + 1
                       for r in range(rows))
    horizontal = Counter(min(columns - 1, c + width - 1) - max(0, c - width + 1) + 1
                         for c in range(columns))
    degrees = Counter()
    for a, count_a in vertical.items():
        for b, count_b in horizontal.items():
            degrees[a * b - 1] += count_a * count_b
    degree_sum = sum(d * count for d, count in degrees.items())
    assert degree_sum % 2 == 0
    window_area = height * width
    return {
        "vertices": rows * columns,
        "edges": degree_sum // 2,
        "max_degree": max(degrees),
        "sum_degree_squared": sum(d * d * count for d, count in degrees.items()),
        "sum_degree_log2_one_plus_degree": math.fsum(count * d * math.log2(1 + d)
                                                     for d, count in degrees.items()),
        "weight_sum": (rows - height + 1) * (columns - width + 1) * window_area * (window_area - 1) // 2,
    }


def verify(report_path, checkpoint_path, protocol_path):
    audit = Audit()
    input_hashes = {str(p.relative_to(ROOT)): digest(p)
                    for p in (report_path, checkpoint_path, protocol_path)}
    report = json.loads(report_path.read_text())
    checkpoint = json.loads(checkpoint_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    audit.equal(report["status"], "completed", "completion", "study status")
    audit.equal(report["phases"], {"moments": "completed", "vertex": "completed", "memory": "completed"},
                "completion", "all phases finished")
    audit.equal(report["metadata"], checkpoint["metadata"], "preservation", "initial metadata unchanged")
    audit.equal(report["cases"], checkpoint["cases"], "preservation", "all moment records unchanged")
    audit.equal(report["topologies"], checkpoint["topologies"], "preservation", "topology records unchanged")
    audit.equal(report["protocol"], protocol, "provenance", "saved protocol matches file")
    audit.equal(checkpoint["protocol"], protocol, "provenance", "protocol unchanged across interruption")
    protocol_hash = recorded_digest(protocol_path)
    audit.equal(report["metadata"]["protocol_sha256"], protocol_hash, "provenance", "initial protocol hash")
    active_hashes = report["metadata"]["code_sha256"]
    transitions = report.get("source_transitions", [])
    audit.equal(len(transitions), 1, "provenance", "one recorded source transition")
    for transition in transitions:
        audit.equal(transition["from_code_sha256"], active_hashes, "provenance", "source chain")
        audit.equal(transition["from_collection_sha256"], collection_digest(active_hashes),
                    "provenance", "source collection before transition")
        next_hashes = transition["to_code_sha256"]
        changed = sorted(k for k in set(active_hashes) | set(next_hashes)
                         if active_hashes.get(k) != next_hashes.get(k))
        audit.equal(changed, ["code/run_scaling.py"], "provenance", "runner-only source change")
        audit.equal(transition["changed_paths"], changed, "provenance", "declared changed paths")
        audit.equal(transition["to_collection_sha256"], collection_digest(next_hashes),
                    "provenance", "source collection after transition")
        for path_key, hash_key in (("initial_runner_archive", "initial_runner_archive_sha256"),
                                   ("interrupted_report_archive", "interrupted_report_archive_sha256")):
            audit.equal(recorded_digest(ROOT / transition[path_key]), transition[hash_key],
                        "provenance", path_key + " hash")
        audit.equal(transition["initial_runner_archive_sha256"],
                    report["metadata"]["code_sha256"]["code/run_scaling.py"],
                    "provenance", "archived runner matches initial metadata")
        audit.equal(transition["interrupted_report_archive_sha256"], recorded_digest(checkpoint_path),
                    "provenance", "archived report identity")
        audit.equal(transition["protocol_sha256"], protocol_hash, "provenance", "transition protocol hash")
        audit.check(bool(transition.get("reason", "").strip()), "provenance", "transition reason retained")
        active_hashes = next_hashes
    current_hashes = {str(p.relative_to(ROOT)): recorded_digest(p) for p in sorted((ROOT / "code").glob("*.py"))}
    audit.equal(active_hashes, current_hashes, "provenance", "active sources match current files")
    for environment in report.get("resume_environments", []):
        audit.equal(environment["code_sha256"], active_hashes, "provenance", "resume source hashes")
        for key in ("python", "numpy", "platform", "cpu_model", "thread_environment"):
            audit.equal(environment[key], report["metadata"][key], "provenance", "resume environment " + key)

    audit.equal(len(report["cases"]), 18, "moment_counts", "18 distinct moment cases")
    audit.equal(set(report["cases"]), {c["id"] for c in protocol["cases"]}, "moment_counts", "protocol case identifiers")
    target_error = protocol["benchmark"]["relative_variance_error_target"]
    workspace_limit = protocol["benchmark"]["alphabet_workspace_budget_bytes"]
    for case_id, case in report["cases"].items():
        audit.equal(set(case["methods"]), METHODS, "moment_counts", case_id + " methods")
        audit.equal(case["status"], "completed", "moment_counts", case_id + " finished")
        targets = {k: fraction(case["methods"]["piecewise_exact"]["calls"][0][k + "_output"])
                   for k in ("mean", "variance")}
        for key in targets:
            audit.equal(fraction(case["exact_" + key]), targets[key], "exact_outputs", case_id + " saved oracle " + key)
        expected = grid_counts(case["specification"])
        for key, value in expected.items():
            if isinstance(value, float):
                audit.check(math.isclose(case["counts"][key], value, rel_tol=1e-12, abs_tol=1e-12),
                            "grid_identities", case_id + " " + key)
            else:
                audit.equal(case["counts"][key], value, "grid_identities", case_id + " " + key)
        audit.equal(case["counts"]["absolute_weight_sum"], expected["weight_sum"],
                    "grid_identities", case_id + " absolute weight sum")
        for method, row in case["methods"].items():
            label = case_id + "/" + method
            calls = audit.summary(row, 3, label)
            if row["status"] == "omitted_workspace":
                audit.counts["workspace_omissions"] += 1
                audit.equal(row["calls"], [], "workspace", label + " no measured calls")
                audit.check(row["estimated_workspace_bytes"] > workspace_limit,
                            "workspace", label + " estimate exceeds guard")
                audit.equal(row["workspace_estimate"]["total_bytes"], row["estimated_workspace_bytes"],
                            "workspace", label + " stored estimate")
            else:
                audit.equal(row["status"], "completed", "moment_counts", label + " completed status")
                audit.counts["completed_moment_combinations"] += int(row["status"] == "completed")
                if method.startswith("alphabet_"):
                    audit.check(row["estimated_workspace_bytes"] <= workspace_limit,
                                "workspace", label + " eligible estimate")
            errors = [audit.output(call, method, targets, case["specification"]["q"], target_error,
                                   label + "/" + str(call["repetition"])) for call in calls]
            audit.counts["completed_moment_calls"] += len(calls)
            audit.counts["exact_moment_calls" if method != "alphabet_float64" else "floating_moment_calls"] += len(calls)
            for key in ("mean", "variance"):
                audit.equal(row["maximum_absolute_" + key + "_relative_error"],
                            max((abs(e[key]) for e in errors), default=None),
                            "output_errors", label + " aggregate " + key + " error")
                audit.equal(row["exact_" + key + "_equal"], True if calls and method != "alphabet_float64" else None,
                            "exact_outputs", label + " aggregate " + key + " equality")
            audit.equal(row["variance_error_target_met"],
                        all(abs(e["variance"]) <= target_error for e in errors) if errors else None,
                        "output_errors", label + " aggregate variance target")
    for key, expected in (("completed_moment_combinations", 67), ("workspace_omissions", 5),
                          ("completed_moment_calls", 201)):
        audit.equal(audit.counts[key], expected, "moment_counts", key)

    audit.equal(set(report["vertex_cases"]), {str(d) for d in protocol["vertex_ablation"]["degrees"]},
                "vertex", "all declared degrees retained")
    for degree, case in report["vertex_cases"].items():
        outputs = []
        completed_methods = 0
        for method, row in case["methods"].items():
            label = "vertex/" + degree + "/" + method
            calls = audit.summary(row, 5, label)
            completed_methods += int(row["status"] == "completed")
            audit.counts["completed_vertex_combinations"] += int(row["status"] == "completed")
            for call in calls:
                value = int(call["raw_square_sum"])
                outputs.append(value)
                audit.equal(value, int(case["exact_raw_square_sum"]), "vertex", label + " exact output")
                audit.equal(call["exact_output_equal"], True, "vertex", label + " equality flag")
                audit.equal(call["diagnostics"]["raw_square_sum_bits"], abs(value).bit_length(),
                            "vertex", label + " output bit length")
                if method == "pairwise_vertex":
                    audit.equal(call["diagnostics"]["pair_products"], int(degree) * (int(degree) + 1) // 2,
                                "vertex", label + " unordered products")
            audit.counts["completed_vertex_calls"] += len(calls)
        audit.check(len(set(outputs)) <= 1, "vertex", degree + " paired outputs agree")
        audit.counts["paired_vertex_degrees"] += int(completed_methods == 2)
    interrupted = report["vertex_cases"]["8"]["methods"]["sweep_vertex"]
    initial_call = checkpoint["vertex_cases"]["8"]["methods"]["sweep_vertex"]["calls"][0]
    audit.equal(len(interrupted["calls"]), 1, "censoring", "degree-eight sweep not repeated")
    audit.equal(interrupted["status"], "censored_interruption", "censoring", "degree-eight censored status")
    audit.equal(interrupted["calls"][0]["elapsed_seconds"], None, "censoring", "unavailable elapsed time retained")
    for key in ("repetition", "started_utc"):
        audit.equal(interrupted["calls"][0][key], initial_call[key], "censoring", "interrupted call identity " + key)
    audit.equal(interrupted["calls"][0]["status"], "censored_interruption", "censoring", "call censoring retained")
    audit.equal(audit.counts["completed_vertex_combinations"], 11, "vertex", "11 complete five-repetition summaries")
    audit.equal(audit.counts["completed_vertex_calls"], 55, "vertex", "55 completed vertex calls")
    audit.equal(audit.counts["paired_vertex_degrees"], 5, "vertex", "five degrees with both kernels observed")

    expected_memory = {case_id + "/" + method for case_id in protocol["memory"]["case_ids"] for method in METHODS}
    audit.equal(set(report["memory"]), expected_memory, "memory", "all selected memory records present")
    for label, row in report["memory"].items():
        status = row["status"]
        audit.counts["memory_" + status] += 1
        case = report["cases"][row["case_id"]]
        if status == "not_run":
            audit.check(case["methods"][row["method"]]["status"] != "completed", "memory", label + " ineligible timing method")
            audit.check(not row.get("calls") and "call" not in row, "memory", label + " no invented memory call")
            continue
        audit.check(status in {"completed", "censored_timeout", "censored_interruption"}, "memory", label + " terminal status")
        audit.equal(len(row["calls"]), 1, "memory", label + " one fresh memory call")
        call = row["call"]
        audit.equal(call, row["calls"][0], "memory", label + " retained call copies agree")
        audit.equal(call["status"], status, "memory", label + " status agrees")
        audit.equal(call.get("peak_is_censored"), status != "completed", "memory", label + " peak censoring flag")
        peak = call.get("peak_traced_bytes")
        audit.check(isinstance(peak, int) and peak >= 0, "memory", label + " recorded traced allocation peak")
        if status == "completed":
            targets = {k: fraction(case["exact_" + k]) for k in ("mean", "variance")}
            audit.output(call, row["method"], targets, case["specification"]["q"], target_error, "memory/" + label)
        else:
            audit.check("mean_output" not in call and "variance_output" not in call,
                        "memory", label + " no completed output attributed to censored call")

    audit.counts["moment_cases"] = len(report["cases"])
    audit.counts["grid_formula_cases"] = len(report["cases"])
    audit.counts["source_transitions"] = len(transitions)
    for name, value in input_hashes.items():
        audit.equal(digest(ROOT / name), value, "input_stability", name + " unchanged during audit")
    return {
        "status": "passed" if not audit.failures else "failed",
        "scope": "Independent audit of saved records, exact rational errors, arithmetic summaries, grid identities and provenance; no algorithm execution or scientific measurements.",
        "input_sha256": input_hashes,
        "verifier_sha256": digest(Path(__file__)),
        "counts": dict(sorted(audit.counts.items())),
        "checks": dict(sorted(audit.checks.items())),
        "failures": audit.failures,
        "limitations": [
            "Output bit lengths concern unreduced final numerators, not maximum intermediate operands.",
            "Recorded allocation peaks are traced allocations rather than process RSS; censored peaks remain marked.",
            "The degree-eight sweep has no completed observation or paired timing ratio.",
            "This record audit does not rerun the algorithms or establish independent timing reproducibility.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "results" / "scaling.json")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "provenance" / "scaling_interruption.json")
    parser.add_argument("--protocol", type=Path, default=ROOT / "results" / "scaling_protocol.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "scaling_verification.json")
    args = parser.parse_args()
    result = verify(args.report.resolve(), args.checkpoint.resolve(), args.protocol.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"status": result["status"], "counts": result["counts"],
                      "failed_checks": len(result["failures"]), "output": str(args.output)}, indent=2))
    raise SystemExit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
