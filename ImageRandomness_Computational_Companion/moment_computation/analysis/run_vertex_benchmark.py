"""Fresh, checkpointed vertex-only timings of the unchanged exact kernels.

Protocol inspection and --verify-only are untimed. Actual measurements require
the selected protocol checksum and a fresh destination. No historical scaling
record, implementation, or provenance file is modified by this runner.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from export_provenance import matches_recorded_digest
import platform
import signal
import statistics
import sys
import tempfile
import time

THREAD_NAMES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")
for _name in THREAD_NAMES:
    os.environ[_name] = "1"
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROTOCOL = ROOT / "results" / "vertex_protocol.json"
DEGREES = (8, 16, 32, 64, 128, 256)
METHODS = ("sweep_vertex", "pairwise_vertex")
FIELDS = ("degree", "q", "method", "repetition", "position", "status",
          "elapsed_seconds", "raw_square_sum", "exact_equality", "pair_products",
          "interval_integrations", "raw_events", "neighbor_coefficient_events",
          "raw_square_sum_bits", "completed_repetitions", "median_seconds",
          "minimum_seconds", "maximum_seconds", "mean_seconds",
          "pairwise_over_sweep_median_ratio")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def atomic_text(path, content, *, exclusive=False):
    """Persist a complete file; initial publication never replaces a file."""
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp",
                                              dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive:
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def summaries(report):
    rows = []
    for degree in DEGREES:
        by_method = {}
        for method in METHODS:
            records = [row for row in report["calls"]
                       if row["degree"] == degree and row["method"] == method]
            completed = [row for row in records if row["status"] == "completed"]
            full = (len(records) == len(completed) == 5
                    and {row["repetition"] for row in completed} == set(range(5))
                    and all(row["exact_equality"] for row in completed))
            times = [row["elapsed_seconds"] for row in completed]
            result = {"degree": degree, "q": 65536, "method": method,
                      "completed_repetitions": len(completed), "all_five_complete": full,
                      "median_seconds": statistics.median(times) if full else None,
                      "minimum_seconds": min(times) if full else None,
                      "maximum_seconds": max(times) if full else None,
                      "mean_seconds": statistics.mean(times) if full else None}
            by_method[method] = result
            rows.append(result)
        sweep, pairwise = (by_method[method] for method in METHODS)
        ratio = (pairwise["median_seconds"] / sweep["median_seconds"]
                 if sweep["all_five_complete"] and pairwise["all_five_complete"] else None)
        for result in by_method.values():
            result["pairwise_over_sweep_median_ratio"] = ratio
    return rows


def checkpoint(report, output_dir, *, exclusive=False):
    report["updated_utc"] = utc_now()
    report["summaries"] = summaries(report)
    content = json.dumps(report, indent=2, allow_nan=False) + "\n"
    atomic_text(output_dir / "vertex_benchmark.json", content, exclusive=exclusive)
    summary_map = {(row["degree"], row["method"]): row for row in report["summaries"]}
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
    writer.writeheader()
    for call in report["calls"]:
        row = dict(call)
        row.update(call.get("diagnostics", {}))
        row.update(summary_map[(call["degree"], call["method"])])
        writer.writerow(row)
    atomic_text(output_dir / "vertex_benchmark.csv", stream.getvalue(), exclusive=exclusive)


def validate_protocol(protocol):
    if (protocol["cases"]["q"] != 65536
            or protocol["cases"]["degrees"] != list(DEGREES)
            or protocol["benchmark"]["repeats"] != 5
            or protocol["benchmark"]["methods"] != list(METHODS)
            or protocol["benchmark"]["call_timeout_seconds"] != 120
            or protocol["benchmark"]["thread_settings"] != {
                name: "1" for name in THREAD_NAMES}):
        raise ValueError("Protocol numerical settings do not match this runner")
    for relative, expected in protocol["implementation_sha256"].items():
        if not matches_recorded_digest(ROOT / relative, expected):
            raise RuntimeError("Unchanged kernel source hash mismatch: " + relative)
    for relative, expected in protocol["historical_records"].items():
        if not matches_recorded_digest(ROOT / relative, expected):
            raise RuntimeError("Historical record hash mismatch: " + relative)


def environment_metadata():
    cpu = "unknown"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    try:
        numpy_version = importlib.metadata.version("numpy")
    except importlib.metadata.PackageNotFoundError:
        numpy_version = None
    sources = sorted((ROOT / "code").glob("*.py")) + [Path(__file__).resolve()]
    return {
        "python": sys.version, "python_implementation": platform.python_implementation(),
        "numpy_installed_version": numpy_version,
        "numpy_imported_for_benchmark": False,
        "platform": platform.platform(), "machine": platform.machine(),
        "cpu_model": cpu, "logical_cpu_count": os.cpu_count(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "clock": "time.perf_counter", "clock_info": vars(time.get_clock_info("perf_counter")),
        "thread_environment": {name: os.environ[name] for name in THREAD_NAMES},
        "gc_enabled": gc.isenabled(), "gc_thresholds": list(gc.get_threshold()),
        "debugger_trace": False, "profiler": False, "memory_tracing": False,
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in sources},
    }


def load_kernels():
    sys.path.insert(0, str(ROOT / "code"))
    from vertex_ablation import make_star_case, pairwise_vertex, sweep_vertex
    return make_star_case, {"sweep_vertex": sweep_vertex, "pairwise_vertex": pairwise_vertex}


def tiny_check(make_star_case, kernels):
    """One tiny warmup only, with an independent defining-sum check."""
    case = make_star_case(2, 8)
    outputs = [kernels[method](case).raw_square_sum for method in METHODS]
    direct = sum(sum(weight * sum(abs(abs(case.reference[0] - u0)
                                    - abs(case.reference[j] - u))
                                for u in range(case.q))
                     for j, weight in enumerate(case.weights, start=1)) ** 2
                 for u0 in range(case.q))
    if outputs != [direct, direct]:
        raise ArithmeticError("Tiny defining-sum check failed")
    return {"status": "passed", "q": 8, "degree": 2,
            "raw_square_sum": str(direct), "kernel_calls": 2,
            "elapsed_time_measured": False}


def timeout_handler(signum, frame):
    raise TimeoutError("Vertex kernel exceeded the nominal 120-second deadline")


def benchmark(protocol, digest, output_dir):
    if sys.gettrace() is not None or sys.getprofile() is not None:
        raise RuntimeError("Disable debugger tracing and profiling before timing")
    tracing_module = sys.modules.get("tracemalloc")
    if tracing_module is not None and tracing_module.is_tracing():
        raise RuntimeError("Disable allocation tracing before timing")
    if not hasattr(signal, "setitimer"):
        raise RuntimeError("This frozen protocol requires SIGALRM/ITIMER_REAL")
    if signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
        raise RuntimeError("An existing real-time alarm would interfere with this run")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("vertex_benchmark.json", "vertex_benchmark.csv"):
        if (output_dir / name).exists():
            raise FileExistsError("Use an explicitly fresh --output-dir; refusing to replace " + name)
    gc.enable()
    make_star_case, kernels = load_kernels()
    warmup = tiny_check(make_star_case, kernels)
    report = {"schema_version": "1.0", "study": protocol["study"],
              "status": "running", "started_utc": utc_now(), "protocol": protocol,
              "protocol_sha256": digest, "environment": environment_metadata(),
              "warmup": warmup, "cases": [], "calls": [], "summaries": []}
    checkpoint(report, output_dir, exclusive=True)
    previous_handler = signal.signal(signal.SIGALRM, timeout_handler)
    oracle = {}
    counter_oracle = {}
    try:
        for degree in DEGREES:
            case = make_star_case(degree, 65536)
            leaf_radii = [min(v, case.q - 1 - v) for v in case.reference[1:]]
            if len(set(leaf_radii)) != degree:
                raise ArithmeticError("Leaf folded radii are not distinct")
            report["cases"].append({"degree": degree, "q": case.q,
                                    "reference": list(case.reference),
                                    "weights": list(case.weights),
                                    "distinct_leaf_folded_radii": len(set(leaf_radii)),
                                    "shared_prepared_case": True})
            for repetition in range(5):
                order = METHODS if repetition % 2 == 0 else METHODS[::-1]
                for position, method in enumerate(order):
                    row = {"degree": degree, "q": case.q, "method": method,
                           "repetition": repetition, "position": position,
                           "status": "in_flight", "started_utc": utc_now(),
                           "elapsed_seconds": None, "raw_square_sum": None,
                           "exact_equality": None}
                    report["calls"].append(row)
                    checkpoint(report, output_dir)
                    gc.collect()
                    if not gc.isenabled():
                        raise RuntimeError("Ordinary automatic GC unexpectedly disabled")
                    signal.setitimer(signal.ITIMER_REAL, 120)
                    started = time.perf_counter()
                    try:
                        result = kernels[method](case)
                    except BaseException as error:
                        row["elapsed_seconds"] = time.perf_counter() - started
                        row["status"] = ("censored_timeout" if isinstance(error, TimeoutError)
                                         else "censored_interruption" if isinstance(error, KeyboardInterrupt)
                                         else "error")
                        row["error"] = type(error).__name__ + ": " + str(error)
                        raise
                    else:
                        row["elapsed_seconds"] = time.perf_counter() - started
                    finally:
                        signal.setitimer(signal.ITIMER_REAL, 0)
                    row["raw_square_sum"] = str(result.raw_square_sum)
                    row["diagnostics"] = dict(result.diagnostics)
                    oracle.setdefault(degree, result.raw_square_sum)
                    row["exact_equality"] = result.raw_square_sum == oracle[degree]
                    counter_oracle.setdefault((degree, method), result.diagnostics)
                    if not row["exact_equality"]:
                        raise ArithmeticError("Exact vertex outputs disagree")
                    if result.diagnostics != counter_oracle[(degree, method)]:
                        raise ArithmeticError("Kernel counters changed between repetitions")
                    if (method == "pairwise_vertex"
                            and result.diagnostics["pair_products"] != degree * (degree + 1) // 2):
                        raise ArithmeticError("Pair-product counter mismatch")
                    if method == "sweep_vertex" and result.diagnostics["raw_events"] > 3 * degree + 5:
                        raise ArithmeticError("Sweep-event bound violated")
                    row["status"] = "completed"
                    row["finished_utc"] = utc_now()
                    checkpoint(report, output_dir)
        report["status"] = "completed"
        report["finished_utc"] = utc_now()
        checkpoint(report, output_dir)
    except BaseException as error:
        if report["calls"] and report["calls"][-1]["status"] == "in_flight":
            report["calls"][-1]["status"] = "error"
            report["calls"][-1]["error"] = type(error).__name__ + ": " + str(error)
        report["status"] = "aborted"
        report["error"] = type(error).__name__ + ": " + str(error)
        checkpoint(report, output_dir)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--authorized-protocol-sha256")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    digest = sha256(args.protocol)
    validate_protocol(protocol)
    print("Protocol SHA-256 " + digest, flush=True)
    if args.verify_only:
        make_star_case, kernels = load_kernels()
        print(json.dumps(tiny_check(make_star_case, kernels), indent=2))
        return
    if args.authorized_protocol_sha256 is None:
        print("No measurements started. Supply --authorized-protocol-sha256 with the matching protocol digest.")
        return
    if args.authorized_protocol_sha256 != digest:
        parser.error("Authorization checksum does not match the frozen protocol")
    report = benchmark(protocol, digest, args.output_dir.resolve())
    print(json.dumps({"status": report["status"], "calls": len(report["calls"]),
                      "summaries": report["summaries"]}, indent=2))


if __name__ == "__main__":
    main()
