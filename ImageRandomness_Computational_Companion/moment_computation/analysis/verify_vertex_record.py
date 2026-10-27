"""Audit saved vertex-benchmark records without importing or timing kernels.

Recompute summaries, deterministic inputs and structural counters. An optional
prior companion archive also checks that historical code, results, provenance
and the original scaling-record audit remain byte-identical.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from export_provenance import recorded_digest, matches_recorded_digest
import statistics
import zipfile

ROOT = Path(__file__).resolve().parent.parent
DEGREES = (8, 16, 32, 64, 128, 256)
METHODS = ("sweep_vertex", "pairwise_vertex")
PROTOCOL_HASH = "3eb147e6151525ad3a1a723421a3e607866bf625aafb8e266e8a67cd9321e3bc"
RUNNER_HASH = "f56fded3b1460b276285fc8e59b785180ec9405a69bcb2b5d968dc0f6b948e48"
CSV_FIELDS = ["degree", "q", "method", "repetition", "position", "status",
              "elapsed_seconds", "raw_square_sum", "exact_equality", "pair_products",
              "interval_integrations", "raw_events", "neighbor_coefficient_events",
              "raw_square_sum_bits", "completed_repetitions", "median_seconds",
              "minimum_seconds", "maximum_seconds", "mean_seconds",
              "pairwise_over_sweep_median_ratio"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def structural_counts(degree, q, reference):
    """Count support intervals directly from the frozen reference formulas.

    All leaves here have 0 < m < (q-1)/2, so their three nonzero coefficient
    events are at 1, m+1 and q-m. Only intervals below q-m_center carry mass.
    No moment kernel, profile polynomial or alphabet enumeration is evaluated.
    """
    radii = [min(v, q - 1 - v) for v in reference]
    cuts = [{0, 1, m + 1, q - m, q} for m in radii]
    support_end = q - radii[0]
    sweep_cuts = set(cuts[0])
    for m in radii[1:]:
        sweep_cuts.update((1, m + 1, q - m))
    sweep_intervals = sum(x < support_end for x in sweep_cuts)
    pair_intervals = sum(sum(x < support_end for x in (cuts[0] | cuts[j] | cuts[k]))
                         for j in range(1, degree + 1)
                         for k in range(j, degree + 1))
    return {
        "sweep_vertex": {"pair_products": 0, "interval_integrations": sweep_intervals,
                         "raw_events": 3 * degree + 5,
                         "neighbor_coefficient_events": 3 * degree},
        "pairwise_vertex": {"pair_products": degree * (degree + 1) // 2,
                            "interval_integrations": pair_intervals, "raw_events": 0,
                            "neighbor_coefficient_events": 0},
    }


def audit(results, historical_archive=None):
    checks = 0

    def require(condition, message):
        nonlocal checks
        checks += 1
        if not condition:
            raise AssertionError(message)

    report_path = results / "vertex_benchmark.json"
    csv_path = results / "vertex_benchmark.csv"
    protocol_path = ROOT / "results" / "vertex_protocol.json"
    report = json.loads(report_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    require(recorded_digest(protocol_path) == PROTOCOL_HASH == report["protocol_sha256"],
            "Frozen protocol checksum mismatch")
    require(report["protocol"] == protocol, "Embedded protocol differs")
    require(recorded_digest(ROOT / "analysis" / "run_vertex_benchmark.py") == RUNNER_HASH,
            "Frozen runner checksum mismatch")
    source_hashes = report["environment"]["source_sha256"]
    expected_sources = {str(p.relative_to(ROOT)) for p in (ROOT / "code").glob("*.py")}
    expected_sources.add("analysis/run_vertex_benchmark.py")
    require(set(source_hashes) == expected_sources, "Recorded source collection differs")
    for path, expected in source_hashes.items():
        require(matches_recorded_digest(ROOT / path, expected), "Recorded source changed: " + path)
    for path, expected in protocol["implementation_sha256"].items():
        require(source_hashes[path] == expected, "Kernel differs from selected protocol")
    for path, expected in protocol["historical_records"].items():
        require(matches_recorded_digest(ROOT / path, expected), "Historical record changed: " + path)
    environment = report["environment"]
    require(environment["thread_environment"] == protocol["benchmark"]["thread_settings"],
            "Thread environment differs")
    require(environment["gc_enabled"] is True, "GC was not enabled")
    require(all(environment[key] is False for key in
                ("debugger_trace", "profiler", "memory_tracing")), "Tracing was recorded")
    require(environment["clock"] == "time.perf_counter"
            and environment["clock_info"]["monotonic"] is True, "Clock metadata differs")
    require(report["status"] == "completed", "Batch did not complete")
    require(report["warmup"] == {"status": "passed", "q": 8, "degree": 2,
                                  "raw_square_sum": "7784", "kernel_calls": 2,
                                  "elapsed_time_measured": False}, "Warmup record differs")
    require([case["degree"] for case in report["cases"]] == list(DEGREES), "Case order differs")
    require(len(report["calls"]) == 60 and len(report["summaries"]) == 12,
            "The complete sixty-call record is missing")
    expected_order = [(d, r, position, method) for d in DEGREES for r in range(5)
                      for position, method in enumerate(METHODS if r % 2 == 0 else METHODS[::-1])]
    actual_order = [(row["degree"], row["repetition"], row["position"], row["method"])
                    for row in report["calls"]]
    require(actual_order == expected_order, "Alternating call order differs")
    summary_map = {(row["degree"], row["method"]): row for row in report["summaries"]}
    require(len(summary_map) == 12, "Duplicate summary keys")
    historical = json.loads((ROOT / "results" / "scaling.json").read_text())
    recomputed = []
    for case in report["cases"]:
        degree, q = case["degree"], case["q"]
        reference = [q // 3] + [j * (q - 1) // (2 * (degree + 1)) for j in range(1, degree + 1)]
        require(q == 65536 and case["reference"] == reference, "Star reference formula differs")
        require(case["weights"] == [1] * degree and case["shared_prepared_case"] is True,
                "Star weights or preparation metadata differ")
        leaf_radii = [min(v, q - 1 - v) for v in reference[1:]]
        require(len(set(leaf_radii)) == degree == case["distinct_leaf_folded_radii"],
                "Leaf folded radii are not distinct")
        counts = structural_counts(degree, q, reference)
        calls = [row for row in report["calls"] if row["degree"] == degree]
        values = {row["raw_square_sum"] for row in calls}
        require(len(values) == 1, "Exact outputs disagree between calls")
        value = int(next(iter(values)))
        require(str(value) == next(iter(values)), "Exact integer result is not canonical")
        require(str(value) == historical["vertex_cases"][str(degree)]["exact_raw_square_sum"],
                "New exact output differs from historical exact output")
        medians = {}
        for method in METHODS:
            subset = [row for row in calls if row["method"] == method]
            require(len(subset) == 5, "Incomplete method repetitions")
            for row in subset:
                require(row["q"] == q and row["status"] == "completed"
                        and row["exact_equality"] is True,
                        "Incomplete or unequal call")
                require(type(row["elapsed_seconds"]) is float
                        and math.isfinite(row["elapsed_seconds"]) and row["elapsed_seconds"] > 0,
                        "Invalid elapsed time")
                expected_diagnostics = dict(counts[method], degree=degree,
                                            raw_square_sum_bits=abs(value).bit_length(),
                                            kernel=("sorted_event_sweep" if method == "sweep_vertex"
                                                    else "incident_pair_expansion"))
                require(row["diagnostics"] == expected_diagnostics, "Structural counters differ")
            times = [row["elapsed_seconds"] for row in subset]
            expected = {"degree": degree, "q": q, "method": method,
                        "completed_repetitions": 5, "all_five_complete": True,
                        "median_seconds": statistics.median(times),
                        "minimum_seconds": min(times), "maximum_seconds": max(times),
                        "mean_seconds": statistics.mean(times)}
            medians[method] = expected["median_seconds"]
            for field, value_expected in expected.items():
                require(summary_map[(degree, method)][field] == value_expected,
                        "Summary field differs: " + field)
        ratio = medians["pairwise_vertex"] / medians["sweep_vertex"]
        for method in METHODS:
            require(summary_map[(degree, method)]["pairwise_over_sweep_median_ratio"] == ratio,
                    "Median ratio differs")
        recomputed.append({"degree": degree, "q": q, "raw_square_sum": str(value),
                           "sweep_median_seconds": medians["sweep_vertex"],
                           "pairwise_median_seconds": medians["pairwise_vertex"],
                           "pairwise_over_sweep_median_ratio": ratio,
                           "structural_counts": counts})
    with csv_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        csv_rows = list(reader)
    require(fields == CSV_FIELDS, "CSV schema differs")
    require(len(csv_rows) == 60, "CSV does not contain sixty calls")
    for csv_row, call in zip(csv_rows, report["calls"]):
        expected = dict(call)
        expected.update(call["diagnostics"])
        expected.update(summary_map[(call["degree"], call["method"])])
        for field in fields:
            require(csv_row[field] == str(expected[field]), "CSV/JSON mismatch: " + field)
    archive_audit = None
    if historical_archive is not None:
        preserved = []
        prefix = "ImageRandomness_Computational_Companion/moment_computation/"
        with zipfile.ZipFile(historical_archive) as archive:
            for name in archive.namelist():
                if not name.startswith(prefix) or name.endswith("/"):
                    continue
                relative = name[len(prefix):]
                if not (relative.startswith(("code/", "results/", "provenance/"))
                        or relative == "analysis/verify_scaling_record.py"):
                    continue
                expected = hashlib.sha256(archive.read(name)).hexdigest()
                require((ROOT / relative).exists() and matches_recorded_digest(ROOT / relative, expected),
                        "Archived historical file changed: " + relative)
                preserved.append(relative)
        require(len(preserved) > 0, "No historical files found in the archive")
        archive_audit = {"archive_sha256": digest(historical_archive),
                         "original_digest_matches": preserved}
    return {"status": "passed", "checks": checks,
            "scope": "Saved-record consistency and deterministic structural audit. No kernels or timings were rerun.",
            "protocol_sha256": PROTOCOL_HASH, "runner_sha256": RUNNER_HASH,
            "auditor_sha256": digest(__file__), "record_sha256": digest(report_path),
            "csv_sha256": digest(csv_path), "completed_calls": 60,
            "complete_method_summaries": 12, "historical_archive_audit": archive_audit,
            "recomputed_results": recomputed,
            "limitations": [
                "Saved records and source inspection establish internal consistency, not independent observation of wall-clock execution or system load.",
                "This comparison isolates prepared vertex kernels; profile construction and complete moment evaluation are outside its timing scope.",
                "Five alternating repetitions do not fully balance kernel order. Timing ratios describe this recorded machine and workload.",
                "Structural counters count mathematical operations and intervals, not CPU instructions."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--historical-archive", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "vertex_verification.json")
    args = parser.parse_args()
    result = audit(args.results_dir, args.historical_archive)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"status": result["status"], "checks": result["checks"],
                      "degree_256": result["recomputed_results"][-1]}, indent=2))


if __name__ == "__main__":
    main()
