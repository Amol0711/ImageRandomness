"""Prespecified comparison of identical raw first/second moment outputs.

Run with the selected protocol SHA-256 supplied explicitly::

    python code/run_comparison.py --authorized-protocol-sha256 HASH

Budgets omit infeasible alphabet-enumerating rows before execution. An omitted
row is never represented as a timeout or an observed speedup. Timing and peak
traced allocations are measured in separate calls. All Fraction materialization
is inside the moment-evaluation timer. The topology is reusable and is timed
separately. This harness is intentionally sequential.
"""
from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time
import tracemalloc

# Set before importing NumPy or the alphabet baseline.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROTOCOL = ROOT / "results" / "protocol.json"
METHODS = ("piecewise_exact", "piecewise_cached", "alphabet_exact", "alphabet_float64")


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_protocol(path=DEFAULT_PROTOCOL, authorized_sha256=None):
    path = Path(path)
    digest = sha256_file(path)
    if authorized_sha256 is not None and digest != authorized_sha256:
        raise ValueError(f"Protocol SHA-256 mismatch: expected {authorized_sha256}, found {digest}")
    return json.loads(path.read_text()), digest


def reference_for(case):
    """The frozen deterministic reference; no data-adaptive case selection."""
    q, rows, columns = case["q"], case["rows"], case["columns"]
    values = []
    for index in range(rows * columns):
        x = (1664525 * (index + 271828) + 1013904223) & 0xffffffff
        x ^= x >> 13
        kind = case["reference"]
        if kind == "palette":
            v = (x % 32) * (q - 1) // 31
        elif kind == "diverse":
            v = x % q
        elif kind == "checker":
            row, column = divmod(index, columns)
            v = 0 if (row + column) % 2 == 0 else q // 2
        elif kind == "split":
            v = 0 if index % columns < columns // 2 else q // 2
        else:
            raise ValueError(f"Unknown reference construction {kind!r}")
        values.append(v)
    return tuple(values)


def problem_counts(reference, q, design):
    classes = tuple(min(v, q - 1 - v) for v in reference)
    pairs = {(min(classes[i], classes[j]), max(classes[i], classes[j]))
             for i, j, _ in design.edges}
    degrees = tuple(map(len, design.adjacency))
    return {
        "vertices": design.n,
        "edges": len(design.edges),
        "folded_classes": len(set(classes)),
        "occurring_class_pairs": len(pairs),
        "max_degree": max(degrees, default=0),
        "sum_degree_squared": sum(d * d for d in degrees),
        "weight_sum": design.weight_sum,
        "alphabet_work_proxy": q * (2 * len(design.edges) + len(pairs) + len(set(classes))),
    }


def exact_relative_error(value, oracle):
    """Avoid rounding the exact oracle before assessing binary64 error."""
    value_fraction = value if isinstance(value, Fraction) else Fraction.from_float(float(value))
    error = value_fraction - oracle
    return float(error / oracle) if oracle else float(error)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Fraction):
        return {"numerator": str(value.numerator), "denominator": str(value.denominator)}
    if hasattr(value, "item"):
        return value.item()
    return value


def save_json(path, payload):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(_json_safe(payload), indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def environment_metadata(protocol_sha256):
    import numpy as np
    cpu_model = "unknown"
    info = Path("/proc/cpuinfo")
    if info.exists():
        for line in info.read_text().splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    paths = [Path(__file__), Path(__file__).with_name("moment_engine.py"),
             Path(__file__).with_name("alphabet_baseline.py"),
             Path(__file__).with_name("monte_carlo.py")]
    return {
        "protocol_sha256": protocol_sha256,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": cpu_model,
        "logical_cpu_count": os.cpu_count(),
        "clock": "time.perf_counter",
        "gc_policy": "gc.collect before each measured call; normal garbage collection remains enabled",
        "thread_environment": {name: os.environ.get(name) for name in
            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
        "code_sha256": {str(path.relative_to(ROOT)): sha256_file(path)
                        for path in paths if path.exists()},
    }


def method_functions(workspace_budget):
    from moment_engine import moments_prepared
    from alphabet_baseline import enumerate_exact, enumerate_float
    return {
        "piecewise_exact": lambda ref, q, design: moments_prepared(ref, q, design, cache=False),
        "piecewise_cached": lambda ref, q, design: moments_prepared(ref, q, design, cache=True),
        "alphabet_exact": lambda ref, q, design: enumerate_exact(ref, q, design,
            max_workspace_bytes=workspace_budget),
        "alphabet_float64": lambda ref, q, design: enumerate_float(ref, q, design,
            max_workspace_bytes=workspace_budget),
    }


def materialize(function, reference, q, design):
    result = function(reference, q, design)
    mean, variance = result.mean, result.variance
    return result, mean, variance


def eligibility(method, counts, reference, q, design, config):
    """Apply declared work budget first, then the baseline's workspace estimator."""
    if method == "piecewise_exact" and counts["edges"] > config["deterministic_edge_budget"]:
        return "budget_edges", None
    if method.startswith("alphabet_"):
        limit = config["alphabet_exact_work_budget"] if method == "alphabet_exact" else config["alphabet_float64_work_budget"]
        if counts["alphabet_work_proxy"] > limit:
            return "budget_work", None
        from alphabet_baseline import estimate_workspace
        estimate = estimate_workspace(reference, q, design,
            mode="exact" if method == "alphabet_exact" else "float64")
        if estimate["total_bytes"] > config["alphabet_workspace_budget_bytes"]:
            return "budget_workspace", estimate
        return None, estimate
    return None, None


def _write_csv(path, rows):
    fields = ["case_id", "rows", "columns", "height", "width", "q", "reference",
              "vertices", "edges", "folded_classes", "occurring_class_pairs", "max_degree",
              "sum_degree_squared", "weight_sum", "alphabet_work_proxy", "method", "status",
              "skip_reason", "median_seconds", "min_seconds", "max_seconds",
              "peak_traced_bytes", "mean_relative_error", "variance_relative_error",
              "exact_mean_equal", "exact_variance_equal", "variance_target_met",
              "estimated_workspace_bytes"]
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run(protocol, digest, output_dir):
    from moment_engine import prepare_graph, rectangular_edges
    output_dir.mkdir(parents=True, exist_ok=True)
    config = protocol["benchmark"]
    functions = method_functions(config["alphabet_workspace_budget_bytes"])
    report = {"metadata": environment_metadata(digest), "protocol": protocol,
              "topologies": [], "cases": [], "rows": [], "status": "running"}
    report_path = output_dir / "comparison.json"
    csv_path = output_dir / "comparison.csv"

    warm_design = prepare_graph(4, [(0, 1, 1), (1, 2, 2), (2, 3, 1)])
    for method in config["method_order"]:
        materialize(functions[method], (0, 2, 5, 7), 8, warm_design)

    topology_cache = {}
    memory_jobs = []
    for case_index, case in enumerate(protocol["cases"]):
        geometry = tuple(case[name] for name in ("rows", "columns", "height", "width"))
        if geometry not in topology_cache:
            enum_times, validate_times = [], []
            for _ in range(config["repeats"]):
                gc.collect()
                start = time.perf_counter()
                edges = rectangular_edges(*geometry)
                enum_times.append(time.perf_counter() - start)
                gc.collect()
                start = time.perf_counter()
                design = prepare_graph(case["rows"] * case["columns"], edges)
                validate_times.append(time.perf_counter() - start)
            topology_cache[geometry] = design
            report["topologies"].append({
                "geometry": geometry, "vertices": design.n, "edges": len(design.edges),
                "edge_enumeration_seconds": enum_times,
                "validation_adjacency_seconds": validate_times,
                "edge_enumeration_median_seconds": statistics.median(enum_times),
                "validation_adjacency_median_seconds": statistics.median(validate_times),
            })
        design = topology_cache[geometry]
        reference = reference_for(case)
        counts = problem_counts(reference, case["q"], design)
        case_record = dict(case, **counts)
        report["cases"].append(case_record)
        oracle_mean = oracle_variance = None
        for method in config["method_order"]:
            row = dict(case_record, case_id=case["id"], method=method)
            skip_reason, estimate = eligibility(method, counts, reference, case["q"], design, config)
            row["estimated_workspace_bytes"] = estimate["total_bytes"] if estimate else None
            if skip_reason:
                row.update(status="omitted", skip_reason=skip_reason)
                report["rows"].append(row)
                continue
            times, diagnostics = [], []
            for _ in range(config["repeats"]):
                gc.collect()
                start = time.perf_counter()
                result, mean, variance = materialize(functions[method], reference, case["q"], design)
                elapsed = time.perf_counter() - start
                times.append(elapsed)
                diagnostics.append(result.diagnostics)
                if oracle_mean is None:
                    if not isinstance(mean, Fraction) or not isinstance(variance, Fraction):
                        raise TypeError("The first completed method must provide the exact oracle")
                    oracle_mean, oracle_variance = mean, variance
                if method != "alphabet_float64" and (mean != oracle_mean or variance != oracle_variance):
                    raise AssertionError(f"Exact mismatch for {case['id']} / {method}")
            row.update(status="completed", skip_reason=None,
                       median_seconds=statistics.median(times), min_seconds=min(times),
                       max_seconds=max(times), timing_replicates_seconds=times,
                       mean_relative_error=exact_relative_error(mean, oracle_mean),
                       variance_relative_error=exact_relative_error(variance, oracle_variance),
                       exact_mean_equal=(mean == oracle_mean) if method != "alphabet_float64" else None,
                       exact_variance_equal=(variance == oracle_variance) if method != "alphabet_float64" else None,
                       diagnostics_replicates=diagnostics,
                       mean_output=mean, variance_output=variance,
                       peak_traced_bytes=None)
            row["variance_target_met"] = abs(row["variance_relative_error"]) <= config["relative_variance_error_target"]
            report["rows"].append(row)
            if case["id"] in config["memory_case_ids"]:
                memory_jobs.append((row, method, reference, case["q"], design))
        case_record["exact_mean"] = oracle_mean
        case_record["exact_variance"] = oracle_variance
        save_json(report_path, report)
        _write_csv(csv_path, report["rows"])
        print(f"timing {case_index + 1}/{len(protocol['cases'])}: {case['id']} complete", flush=True)

    # Allocation tracing is deliberately outside all wall-clock comparisons.
    for row, method, reference, q, design in memory_jobs:
        gc.collect()
        tracemalloc.start()
        try:
            result, mean, variance = materialize(functions[method], reference, q, design)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        row["peak_traced_bytes"] = peak
        row["memory_measurement"] = "separate call; reference/topology excluded; output included"
        if method != "alphabet_float64":
            if mean != row["mean_output"] or variance != row["variance_output"]:
                raise AssertionError("Memory-run output mismatch")
        save_json(report_path, report)
        _write_csv(csv_path, report["rows"])
    report["status"] = "completed"
    report["metadata"]["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    report["summary"] = {
        "cases": len(protocol["cases"]),
        "completed_method_rows": sum(r["status"] == "completed" for r in report["rows"]),
        "omitted_method_rows": sum(r["status"] == "omitted" for r in report["rows"]),
        "all_exact_comparisons_equal": all(r.get("exact_mean_equal", True) is not False and r.get("exact_variance_equal", True) is not False for r in report["rows"]),
        "maximum_absolute_float_variance_relative_error": max(abs(r["variance_relative_error"]) for r in report["rows"] if r["method"] == "alphabet_float64" and r["status"] == "completed"),
    }
    save_json(report_path, report)
    _write_csv(csv_path, report["rows"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--authorized-protocol-sha256", required=True)
    args = parser.parse_args()
    protocol, digest = load_protocol(args.protocol, args.authorized_protocol_sha256)
    run(protocol, digest, args.output_dir)


if __name__ == "__main__":
    main()
