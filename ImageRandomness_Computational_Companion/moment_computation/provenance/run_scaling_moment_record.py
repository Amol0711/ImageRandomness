"""Reproducible scaling study of exact fixed-reference moment computation.

Inspect the protocol and run untimed correctness checks with --verify-only.
Measurements require the SHA-256 of the selected protocol. The default phase
order is moment timings, vertex-kernel timings, then separate allocation tracing.
Completed calls are checkpointed and are never silently replaced on resume.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
from fractions import Fraction
import gc
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import signal
import statistics
import sys
import time
import tracemalloc

# Set these before importing either NumPy or an implementation that uses it.
THREAD_NAMES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")
for _name in THREAD_NAMES:
    os.environ[_name] = "1"
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROTOCOL = ROOT / "results" / "scaling_protocol.json"
METHODS = ("piecewise_exact", "piecewise_cached", "alphabet_exact", "alphabet_float64")
VERTEX_METHODS = ("sweep_vertex", "pairwise_vertex")
TERMINAL = {"completed", "censored_timeout", "censored_interruption", "omitted_workspace",
            "omitted_edges", "blocked_topology", "error"}


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_json(value):
    if isinstance(value, Fraction):
        return {"numerator": str(value.numerator), "denominator": str(value.denominator)}
    if isinstance(value, dict):
        return {str(key): safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(item) for item in value]
    if hasattr(value, "item"):
        return safe_json(value.item())
    return value


def as_fraction(value):
    return Fraction(int(value["numerator"]), int(value["denominator"]))


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(safe_json(value), indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def atomic_csv(path, fields, rows):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def source_hashes():
    # Hash the complete Python source collection, including the imported reference.
    return {str(path.relative_to(ROOT)): sha256_file(path)
            for path in sorted((ROOT / "code").glob("*.py"))}


def environment_metadata(digest):
    import numpy as np
    cpu = "unknown"
    if Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    config = io.StringIO()
    with contextlib.redirect_stdout(config):
        np.show_config()
    metadata = {
        "protocol_sha256": digest, "code_sha256": source_hashes(),
        "started_utc": utc_now(), "python": sys.version, "numpy": np.__version__,
        "platform": platform.platform(), "machine": platform.machine(),
        "cpu_model": cpu, "logical_cpu_count": os.cpu_count(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "clock": "time.perf_counter", "clock_info": vars(time.get_clock_info("perf_counter")),
        "thread_environment": {name: os.environ[name] for name in THREAD_NAMES},
        "numpy_build_configuration": config.getvalue(),
        "timeout_mechanism": "SIGALRM with ITIMER_REAL in a single Linux process",
    }
    try:
        from threadpoolctl import threadpool_info
        metadata["threadpools"] = threadpool_info()
        if any(pool.get("num_threads") != 1 for pool in metadata["threadpools"]):
            raise RuntimeError("An active numerical thread pool is not limited to one thread")
    except ImportError:
        metadata["threadpools"] = "threadpoolctl unavailable; environment settings recorded"
    return metadata


def predicted_edges(case):
    rows, columns, height, width = (case[name] for name in ("rows", "columns", "height", "width"))
    return (sum(rows * (columns - dc) for dc in range(1, width))
            + sum((rows - dr) * (columns - abs(dc))
                  for dr in range(1, height) for dc in range(1 - width, width)))


def validate_protocol(protocol):
    config = protocol["benchmark"]
    if tuple(config["method_order"]) != METHODS or config["repeats"] != 3:
        raise ValueError("The four-method cyclic design requires the declared three repetitions")
    if config["alphabet_workspace_budget_bytes"] != 512 * 1024**2:
        raise ValueError("Unexpected alphabet workspace guard")
    if config["call_timeout_seconds"] != 120 or config["prepared_graph_edge_cap"] != 300000:
        raise ValueError("Unexpected time or graph budget")
    cases = protocol["cases"]
    if len(cases) != 18 or len({case["id"] for case in cases}) != 18:
        raise ValueError("Exactly 18 uniquely identified cases are required")
    keys = set()
    for case in cases:
        for key in ("rows", "columns", "height", "width", "q"):
            if isinstance(case[key], bool) or not isinstance(case[key], int) or case[key] < 2:
                raise ValueError(f"Invalid integer {key} in {case['id']}")
        if case["height"] > case["rows"] or case["width"] > case["columns"] or case["q"] > 2**32:
            raise ValueError(f"Invalid geometry or alphabet in {case['id']}")
        if case["reference"] not in ("palette", "diverse"):
            raise ValueError("Unknown reference construction")
        key = tuple(case[name] for name in ("rows", "columns", "height", "width", "q", "reference"))
        if key in keys:
            raise ValueError("Repeated physical case; shared cases must be referenced, not rerun")
        keys.add(key)
    expected = {(32, 32, w, w, 256, "palette") for w in (2, 3, 4, 6, 8, 12, 16)}
    expected |= {(16, 16, 2, 2, 2**b, "palette") for b in (8, 12, 16, 20, 24, 32)}
    expected |= {(s, s, 2, 2, 256, "palette") for s in (64, 128, 256)}
    expected |= {(32, 32, 2, 2, q, "diverse") for q in (256, 4096)}
    if keys != expected:
        raise ValueError("Cases differ from the declared scaling design")
    vertex = protocol["vertex_ablation"]
    if vertex["degrees"] != [8, 16, 32, 64, 128, 256] or vertex["q"] != 65536 or vertex["repeats"] != 5:
        raise ValueError("Unexpected vertex ablation design")
    if tuple(vertex["method_order"]) != VERTEX_METHODS:
        raise ValueError("Unexpected vertex kernels")
    memory_cases = {case["id"] for case in cases if case["rows"] == 16 and case["q"] in (256, 65536, 2**32)}
    if set(protocol["memory"]["case_ids"]) != memory_cases:
        raise ValueError("Unexpected memory cases")


def method_functions(budget):
    from moment_engine import moments_prepared
    from alphabet_baseline import enumerate_exact, enumerate_float
    return {
        "piecewise_exact": lambda r, q, g: moments_prepared(r, q, g, cache=False),
        "piecewise_cached": lambda r, q, g: moments_prepared(r, q, g, cache=True),
        "alphabet_exact": lambda r, q, g: enumerate_exact(r, q, g, max_workspace_bytes=budget),
        "alphabet_float64": lambda r, q, g: enumerate_float(r, q, g, max_workspace_bytes=budget),
    }


def materialize(function, reference, q, design):
    result = function(reference, q, design)
    return result, result.mean, result.variance


def warmup(functions):
    from moment_engine import prepare_graph
    graph = prepare_graph(4, [(0, 1, 1), (1, 2, 2), (2, 3, 1)])
    oracle = None
    for method in METHODS:
        _, mean, variance = materialize(functions[method], (0, 2, 5, 7), 8, graph)
        if oracle is None:
            oracle = (mean, variance)
        elif method != "alphabet_float64" and (mean, variance) != oracle:
            raise ArithmeticError("Tiny warmup exact outputs disagree")
        elif method == "alphabet_float64":
            if any(abs(Fraction.from_float(float(v)) - target) > Fraction(1, 10**10) * max(1, abs(target))
                   for v, target in zip((mean, variance), oracle)):
                raise ArithmeticError("Tiny warmup floating outputs disagree")


class CallTimeout(TimeoutError):
    pass


def measured_call(function, timeout, trace=False):
    """GC is outside the timer. Tracing is used only for separate memory calls."""
    if tracemalloc.is_tracing():
        raise RuntimeError("Allocation tracing must be inactive before any benchmark call")
    gc.collect()
    old_handler = signal.getsignal(signal.SIGALRM)
    if signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
        raise RuntimeError("An existing real-time alarm would conflict with the benchmark")

    def expired(_signum, _frame):
        raise CallTimeout(f"Call exceeded the prespecified {timeout} second limit")

    signal.signal(signal.SIGALRM, expired)
    if trace:
        tracemalloc.start()
    value, failure = None, None
    record = {"status": "completed", "started_utc": utc_now()}
    signal.setitimer(signal.ITIMER_REAL, timeout)
    start = time.perf_counter()
    try:
        value = function()
    except CallTimeout:
        record["status"] = "censored_timeout"
    except BaseException as error:
        record["status"] = "censored_interruption" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "error"
        record["error_type"] = type(error).__name__
        record["error_message"] = str(error)
        failure = error
    finally:
        record["elapsed_seconds"] = time.perf_counter() - start
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)
        if trace:
            _, peak = tracemalloc.get_traced_memory()
            record["peak_traced_bytes"] = peak
            record["peak_is_censored"] = record["status"] != "completed"
            tracemalloc.stop()
    return record, value, failure


def timing_summary(record, repeats, counts=None):
    calls = record.get("calls", [])
    completed = [call["elapsed_seconds"] for call in calls if call["status"] == "completed"]
    failures = [call for call in calls if call["status"] not in ("completed", "running")]
    record["completed_repetitions"] = len(completed)
    record["timing_replicates_seconds"] = completed
    record["censored_repetitions"] = sum(call["status"].startswith("censored_") for call in calls)
    full = len(completed) == repeats and not failures
    if failures:
        record["status"] = failures[0]["status"]
    elif full:
        record["status"] = "completed"
    elif record.get("status") not in TERMINAL:
        record["status"] = "pending"
    for label, operation in (("median", statistics.median), ("min", min), ("max", max)):
        record[f"{label}_seconds"] = operation(completed) if full else None
    record["partial_completed_median_seconds"] = statistics.median(completed) if completed and not full else None
    if counts:
        denominator = counts["vertices"] + counts["edges"] + counts["sum_degree_log2_one_plus_degree"]
        record["median_seconds_per_edge"] = record["median_seconds"] / counts["edges"] if full and counts["edges"] else None
        record["median_seconds_per_n_plus_e_plus_sum_d_log2"] = record["median_seconds"] / denominator if full else None
        for call in calls:
            if call["status"] == "completed":
                call["seconds_per_edge"] = call["elapsed_seconds"] / counts["edges"] if counts["edges"] else None
                call["seconds_per_n_plus_e_plus_sum_d_log2"] = call["elapsed_seconds"] / denominator


def problem_counts(reference, q, graph):
    classes = tuple(min(value, q - 1 - value) for value in reference)
    degrees = tuple(len(neighbors) for neighbors in graph.adjacency)
    pairs = {(min(classes[i], classes[j]), max(classes[i], classes[j])) for i, j, _ in graph.edges}
    return {
        "vertices": graph.n, "edges": len(graph.edges), "max_degree": max(degrees, default=0),
        "sum_degree_squared": sum(d * d for d in degrees),
        "sum_degree_log2_one_plus_degree": math.fsum(d * math.log2(1 + d) for d in degrees),
        "folded_classes": len(set(classes)), "occurring_class_pairs": len(pairs),
        "weight_sum": graph.weight_sum, "absolute_weight_sum": graph.absolute_weight_sum,
        "minimum_edge_weight": min((c for _, _, c in graph.edges), default=0),
        "maximum_edge_weight": max((c for _, _, c in graph.edges), default=0),
    }


def record_moment_output(call, value):
    result, mean, variance = value
    call.update(mean_output=safe_json(mean), variance_output=safe_json(variance),
                diagnostics=safe_json(result.diagnostics))
    if isinstance(mean, Fraction):
        call["unreduced_mean_numerator_bits"] = abs(result.mean_numerator).bit_length()
        call["unreduced_variance_numerator_bits"] = abs(result.variance_numerator).bit_length()


def compare_outputs(case_record, variance_error_target=1e-10):
    """Recheck every saved repetition, including records restored from disk."""
    oracle_calls = case_record.get("methods", {}).get("piecewise_exact", {}).get("calls", [])
    oracle = next((call for call in oracle_calls if call["status"] == "completed"), None)
    if oracle is None:
        case_record["oracle_status"] = "unavailable"
        return
    targets = [as_fraction(oracle[key + "_output"]) for key in ("mean", "variance")]
    case_record["oracle_status"] = "available"
    case_record["exact_mean"], case_record["exact_variance"] = (safe_json(value) for value in targets)
    for method, row in case_record["methods"].items():
        checked = []
        for call in row.get("calls", []):
            if call["status"] != "completed":
                continue
            for key, target in zip(("mean", "variance"), targets):
                value = call[key + "_output"]
                exact = as_fraction(value) if isinstance(value, dict) else Fraction.from_float(float(value))
                difference = exact - target
                call[key + "_absolute_error_exact"] = safe_json(abs(difference))
                call[key + "_relative_error"] = float(difference / target) if target else float(difference)
                call[key + "_error_scale"] = "relative" if target else "absolute_zero_oracle"
                if method != "alphabet_float64":
                    call["exact_" + key + "_equal"] = exact == target
                    if exact != target:
                        raise ArithmeticError(f"Exact {key} mismatch in {case_record['id']} / {method}")
            call["variance_error_target_met"] = abs(call["variance_relative_error"]) <= variance_error_target
            checked.append(call)
        row["exact_mean_equal"] = all(call["exact_mean_equal"] for call in checked) if checked and method != "alphabet_float64" else None
        row["exact_variance_equal"] = all(call["exact_variance_equal"] for call in checked) if checked and method != "alphabet_float64" else None
        for key in ("mean", "variance"):
            row["maximum_absolute_" + key + "_relative_error"] = max((abs(call[key + "_relative_error"]) for call in checked), default=None)
        row["variance_error_target_met"] = all(call["variance_error_target_met"] for call in checked) if checked else None


class Study:
    def __init__(self, protocol, digest, output_dir, restart=False):
        self.protocol, self.digest = protocol, digest
        self.config = protocol["benchmark"]
        self.output_dir = output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        self.path = output_dir / "scaling.json"
        self.graphs = {}
        self.functions = method_functions(self.config["alphabet_workspace_budget_bytes"])
        metadata = environment_metadata(digest)
        if self.path.exists() and not restart:
            self.report = json.loads(self.path.read_text())
            prior = self.report["metadata"]
            if prior["protocol_sha256"] != digest or prior["code_sha256"] != metadata["code_sha256"]:
                raise ValueError("Resume rejected because protocol or source hashes differ; use a new output directory")
            for key in ("python", "numpy", "platform", "cpu_model", "thread_environment"):
                if prior[key] != metadata[key]:
                    raise ValueError(f"Resume environment differs in {key}; use a new output directory")
            self.report.setdefault("resume_environments", []).append(metadata)
            self.recover_pending(self.report)
            for row in self.report["memory"].values():
                if "call" not in row and row.get("calls"):
                    row["call"] = row["calls"][-1]
                    row["status"] = row["call"]["status"]
        else:
            if restart and self.path.exists():
                backup = output_dir / ("scaling_previous_" + time.strftime("%Y%m%dT%H%M%S", time.gmtime()) + ".json")
                if backup.exists():
                    raise FileExistsError(backup)
                self.path.replace(backup)
            self.report = {"metadata": metadata, "protocol": protocol, "status": "running",
                           "phases": {}, "topologies": {}, "cases": {}, "vertex_cases": {}, "memory": {}}

    @staticmethod
    def recover_pending(value):
        if isinstance(value, dict):
            if value.get("status") == "running" and "repetition" in value:
                value.update(status="censored_interruption", elapsed_seconds=None,
                             error_message="Process ended before the call checkpoint; elapsed time unavailable and call not repeated")
            for item in value.values():
                Study.recover_pending(item)
        elif isinstance(value, list):
            for item in value:
                Study.recover_pending(item)

    def save(self):
        rows, vertex_rows, memory_rows = [], [], []
        for case in self.report["cases"].values():
            for method, row in case.get("methods", {}).items():
                timing_summary(row, self.config["repeats"], case.get("counts"))
                rows.append(dict(case_id=case["id"], **case["specification"], **case.get("counts", {}), method=method,
                                 **{key: value for key, value in row.items() if key != "calls"}))
        for case in self.report["vertex_cases"].values():
            for method, row in case["methods"].items():
                timing_summary(row, self.protocol["vertex_ablation"]["repeats"])
                vertex_rows.append(dict(degree=case["degree"], q=case["q"], method=method,
                                        **{key: value for key, value in row.items() if key != "calls"}))
        for row in self.report["memory"].values():
            memory_rows.append({key: value for key, value in row.items() if key != "call"} | row.get("call", {}))
        self.report["updated_utc"] = utc_now()
        comparisons = [call for case in self.report["cases"].values()
                       for row in case.get("methods", {}).values()
                       for call in row.get("calls", []) if call["status"] == "completed"]
        self.report["verification_summary"] = {
            "completed_moment_calls": len(comparisons),
            "checked_moment_calls": sum("variance_error_target_met" in call for call in comparisons),
            "calls_exceeding_variance_error_target": sum(call.get("variance_error_target_met") is False for call in comparisons),
            "exact_mismatches": sum(call.get("exact_mean_equal") is False or call.get("exact_variance_equal") is False for call in comparisons),
            "oracle_unavailable_cases": [case["id"] for case in self.report["cases"].values() if case.get("oracle_status") == "unavailable"],
        }
        atomic_json(self.path, self.report)
        fields = ["case_id", "rows", "columns", "height", "width", "q", "reference", "vertices", "edges", "max_degree",
                  "sum_degree_squared", "sum_degree_log2_one_plus_degree", "folded_classes", "occurring_class_pairs",
                  "weight_sum", "absolute_weight_sum", "method", "status", "completed_repetitions", "censored_repetitions",
                  "median_seconds", "min_seconds", "max_seconds", "partial_completed_median_seconds",
                  "median_seconds_per_edge", "median_seconds_per_n_plus_e_plus_sum_d_log2", "estimated_workspace_bytes",
                  "exact_mean_equal", "exact_variance_equal", "maximum_absolute_mean_relative_error",
                  "maximum_absolute_variance_relative_error", "variance_error_target_met"]
        atomic_csv(self.output_dir / "scaling.csv", fields, rows)
        atomic_csv(self.output_dir / "vertex_scaling.csv", ["degree", "q", "method", "status", "completed_repetitions",
                   "censored_repetitions", "median_seconds", "min_seconds", "max_seconds", "partial_completed_median_seconds",
                   "exact_output_equal", "pair_products", "interval_integrations", "raw_events", "raw_square_sum_bits"], vertex_rows)
        atomic_csv(self.output_dir / "scaling_memory.csv", ["case_id", "method", "status", "estimated_workspace_bytes",
                   "peak_traced_bytes", "peak_is_censored", "elapsed_seconds", "exact_mean_equal", "exact_variance_equal"], memory_rows)

    def invoke(self, row, repetition, function, trace=False):
        call = {"repetition": repetition, "status": "running", "started_utc": utc_now()}
        row.setdefault("calls", []).append(call)
        self.save()
        observation, value, failure = measured_call(function, self.config["call_timeout_seconds"], trace)
        call.update(observation)
        if failure is not None:
            self.save()
            raise failure
        return call, value

    def graph(self, case, measure):
        from moment_engine import prepare_graph, rectangular_edges
        geometry = tuple(case[key] for key in ("rows", "columns", "height", "width"))
        key = "x".join(map(str, geometry))
        if predicted_edges(case) > self.config["prepared_graph_edge_cap"]:
            return None
        if key in self.graphs:
            return self.graphs[key]
        topology = self.report["topologies"].setdefault(key, {"geometry": geometry, "enumeration": {"calls": []},
                                                              "validation": {"calls": []}})
        graph = None
        if measure:
            for repetition in range(self.config["repeats"]):
                edges = None
                for stage in ("enumeration", "validation"):
                    row = topology[stage]
                    timing_summary(row, self.config["repeats"])
                    if row["status"] in TERMINAL and row["status"] != "completed":
                        return None
                    if any(call["repetition"] == repetition for call in row["calls"]):
                        continue
                    if stage == "validation" and edges is None:
                        edges = rectangular_edges(*geometry)
                    function = (lambda: rectangular_edges(*geometry)) if stage == "enumeration" else (lambda: prepare_graph(case["rows"] * case["columns"], edges))
                    call, value = self.invoke(row, repetition, function)
                    if call["status"] != "completed":
                        self.save()
                        return None
                    if stage == "enumeration":
                        edges = value
                    else:
                        graph = value
                    value = None
                    self.save()
                # No previous graph survives into the next fresh validation call.
                if repetition + 1 < self.config["repeats"]:
                    graph = None
        if graph is None:
            graph = prepare_graph(case["rows"] * case["columns"], rectangular_edges(*geometry))
        if len(graph.edges) > self.config["prepared_graph_edge_cap"] or len(graph.edges) != predicted_edges(case):
            raise ArithmeticError("Prepared graph disagrees with the prespecified edge guard")
        topology.update(vertices=graph.n, edges=len(graph.edges))
        for stage in ("enumeration", "validation"):
            timing_summary(topology[stage], self.config["repeats"])
        self.graphs[key] = graph
        return graph

    def eligibility(self, method, reference, q, graph):
        if method.startswith("alphabet_"):
            from alphabet_baseline import estimate_workspace
            estimate = estimate_workspace(reference, q, graph, mode="exact" if method == "alphabet_exact" else "float64")
            return estimate, estimate["total_bytes"] > self.config["alphabet_workspace_budget_bytes"]
        return None, False

    def moments(self):
        from run_comparison import reference_for
        for index, case in enumerate(self.protocol["cases"]):
            record = self.report["cases"].setdefault(case["id"], {"id": case["id"], "specification": case, "methods": {}})
            if record.get("status") == "completed":
                compare_outputs(record)
                continue
            graph = self.graph(case, measure=True)
            if graph is None:
                status = "omitted_edges" if predicted_edges(case) > self.config["prepared_graph_edge_cap"] else "blocked_topology"
                record["methods"] = {method: {"status": status, "calls": []} for method in METHODS}
                record["status"] = "completed"
                self.save()
                continue
            reference = reference_for(case)
            record["counts"] = problem_counts(reference, case["q"], graph)
            for method in METHODS:
                if method not in record["methods"]:
                    estimate, omit = self.eligibility(method, reference, case["q"], graph)
                    record["methods"][method] = {"status": "omitted_workspace" if omit else "pending", "calls": [],
                                                "workspace_estimate": estimate,
                                                "estimated_workspace_bytes": estimate["total_bytes"] if estimate else None}
            for repetition in range(self.config["repeats"]):
                order = METHODS[repetition % len(METHODS):] + METHODS[:repetition % len(METHODS)]
                for method in order:
                    row = record["methods"][method]
                    timing_summary(row, self.config["repeats"], record["counts"])
                    if row["status"] in TERMINAL or any(call["repetition"] == repetition for call in row["calls"]):
                        continue
                    call, value = self.invoke(row, repetition, lambda: materialize(self.functions[method], reference, case["q"], graph))
                    if value is not None:
                        record_moment_output(call, value)
                    try:
                        compare_outputs(record)
                    finally:
                        self.save()
            record["status"] = "completed"
            self.save()
            print(f"moments {index + 1}/{len(self.protocol['cases'])} {case['id']}", flush=True)
        self.report["phases"]["moments"] = "completed"
        self.save()

    def vertex(self):
        from vertex_ablation import make_star_case, sweep_vertex, pairwise_vertex
        config = self.protocol["vertex_ablation"]
        functions = {"sweep_vertex": sweep_vertex, "pairwise_vertex": pairwise_vertex}
        tiny = make_star_case(2, 8)
        for function in functions.values():
            function(tiny)
        for degree in config["degrees"]:
            record = self.report["vertex_cases"].setdefault(str(degree), {"degree": degree, "q": config["q"],
                "methods": {method: {"status": "pending", "calls": []} for method in VERTEX_METHODS}})
            if record.get("status") == "completed":
                continue
            prepared = make_star_case(degree, config["q"])
            for repetition in range(config["repeats"]):
                order = VERTEX_METHODS[repetition % 2:] + VERTEX_METHODS[:repetition % 2]
                for method in order:
                    row = record["methods"][method]
                    timing_summary(row, config["repeats"])
                    if row["status"] in TERMINAL or any(call["repetition"] == repetition for call in row["calls"]):
                        continue
                    call, value = self.invoke(row, repetition, lambda: functions[method](prepared))
                    if value is not None:
                        call.update(raw_square_sum=str(value.raw_square_sum), diagnostics=value.diagnostics)
                        row.update(value.diagnostics)
                        oracle = record.setdefault("exact_raw_square_sum", str(value.raw_square_sum))
                        call["exact_output_equal"] = str(value.raw_square_sum) == oracle
                        row["exact_output_equal"] = all(c.get("exact_output_equal", True) for c in row["calls"])
                        self.save()
                        if not call["exact_output_equal"]:
                            raise ArithmeticError(f"Vertex kernel mismatch at degree {degree}")
                    self.save()
            record["status"] = "completed"
            self.save()
            print(f"vertex degree {degree}", flush=True)
        self.report["phases"]["vertex"] = "completed"
        self.save()

    def memory(self):
        from run_comparison import reference_for
        if self.report["phases"].get("moments") != "completed":
            raise ValueError("The memory phase requires completed moment timing records and exact oracles")
        for case in self.protocol["cases"]:
            if case["id"] not in self.protocol["memory"]["case_ids"]:
                continue
            graph = self.graph(case, measure=False)
            reference = reference_for(case)
            case_record = self.report["cases"][case["id"]]
            for method in METHODS:
                key = case["id"] + "/" + method
                if key in self.report["memory"]:
                    continue
                timing_row = case_record["methods"][method]
                row = {"case_id": case["id"], "method": method,
                       "estimated_workspace_bytes": timing_row.get("estimated_workspace_bytes")}
                self.report["memory"][key] = row
                if timing_row["status"] != "completed":
                    row.update(status="not_run", reason="Moment timing method did not complete all repetitions",
                               timing_status=timing_row["status"])
                    self.save()
                    continue
                call, value = self.invoke(row, 0, lambda: materialize(self.functions[method], reference, case["q"], graph), trace=True)
                row["call"] = call
                row["status"] = call["status"]
                if value is not None:
                    record_moment_output(call, value)
                    temporary = {"id": case["id"], "methods": {"piecewise_exact": case_record["methods"]["piecewise_exact"]}}
                    if method == "piecewise_exact":
                        temporary["methods"][method] = {"calls": [case_record["methods"][method]["calls"][0], call]}
                    else:
                        temporary["methods"][method] = {"calls": [call]}
                    try:
                        compare_outputs(temporary)
                    finally:
                        self.save()
                self.save()
                print(f"memory {case['id']} {method} {row['status']}", flush=True)
        self.report["phases"]["memory"] = "completed"
        self.save()

    def run(self, phase):
        phases = ("moments", "vertex", "memory") if phase == "all" else (phase,)
        self.report.setdefault("invocations", []).append({"started_utc": utc_now(), "requested_phase": phase})
        if "moments" in phases or "memory" in phases:
            warmup(self.functions)
        self.save()
        for item in phases:
            if self.report["phases"].get(item) != "completed":
                getattr(self, item)()
        self.report["status"] = "completed" if all(self.report["phases"].get(item) == "completed" for item in ("moments", "vertex", "memory")) else "partial"
        self.report["invocations"][-1]["completed_utc"] = utc_now()
        self.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--authorized-protocol-sha256")
    parser.add_argument("--phase", choices=("moments", "vertex", "memory", "all"), default="all")
    parser.add_argument("--verify-only", action="store_true", help="Untimed protocol and small-input correctness checks only")
    parser.add_argument("--restart", action="store_true", help="Explicitly start a new run, preserving the previous JSON report")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    digest = sha256_file(args.protocol)
    validate_protocol(protocol)
    print(f"Protocol SHA-256 {digest}", flush=True)
    if args.authorized_protocol_sha256 and args.authorized_protocol_sha256 != digest:
        parser.error("The authorized protocol SHA-256 does not match this protocol")
    if args.verify_only:
        from vertex_ablation import verify
        warmup(method_functions(protocol["benchmark"]["alphabet_workspace_budget_bytes"]))
        print(json.dumps({"status": "passed", "measurements_run": False, "cases": len(protocol["cases"]),
                          "largest_predicted_graph_edges": max(map(predicted_edges, protocol["cases"])),
                          "vertex_verification": verify()}, indent=2))
        return
    if not args.authorized_protocol_sha256:
        print("No measurements started. Supply --authorized-protocol-sha256 with the matching protocol digest.")
        return
    if sys.platform != "linux" or not hasattr(signal, "setitimer"):
        parser.error("Measurements require Linux SIGALRM/ITIMER_REAL timeout support")
    Study(protocol, digest, args.output_dir, args.restart).run(args.phase)


if __name__ == "__main__":
    main()
