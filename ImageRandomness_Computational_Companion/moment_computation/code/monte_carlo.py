"""Prespecified replicated Monte Carlo accuracy study for raw graph moments.

The exact graph calculation supplies an oracle. Simulation estimates its raw
mean and variance using independent uniform random intensities. It does not
validate encryption, rejection rates, or a nonuniform null model. A percentile
bootstrap interval based on ten replications is descriptive uncertainty, not a
precision guarantee or a certification of a 1% population RMSE target.
"""
from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import gc
import hashlib
from pathlib import Path
import statistics
import time

# run_comparison fixes thread variables before NumPy is imported.
from run_comparison import (ROOT, DEFAULT_PROTOCOL, environment_metadata,
                            exact_relative_error, load_protocol, reference_for,
                            save_json)
import numpy as np
from moment_engine import moments_prepared, prepare_image_graph


def prepare_simulation(reference, q, design, edge_chunk_size):
    """Create fixed NumPy indexing outside the simulation timing scope."""
    if q > np.iinfo(np.int32).max:
        raise ValueError("The simulation kernel requires q within signed int32")
    if (q - 1) * design.absolute_weight_sum > np.iinfo(np.int64).max:
        raise ValueError("A possible raw score exceeds signed int64")
    ref = np.asarray(reference, dtype=np.int32)
    if len(ref) != design.n:
        raise ValueError("Reference length does not match graph")
    chunks = []
    for offset in range(0, len(design.edges), edge_chunk_size):
        chunk = design.edges[offset:offset + edge_chunk_size]
        chunks.append((np.asarray([i for i, _, _ in chunk], dtype=np.intp),
                       np.asarray([j for _, j, _ in chunk], dtype=np.intp),
                       np.asarray([c for _, _, c in chunk], dtype=np.int64)))
    return ref, tuple(chunks)


def scores_from_uniforms(uniforms, reference_array, chunks):
    """Raw integer S for a sample batch; also exposed for deterministic QA."""
    distances = np.abs(uniforms - reference_array[None, :])
    scores = np.zeros(len(uniforms), dtype=np.int64)
    for first, second, weights in chunks:
        contrasts = np.abs(distances[:, first] - distances[:, second])
        scores += np.sum(contrasts * weights[None, :], axis=1, dtype=np.int64)
    return scores


def simulate_moments(reference_array, q, chunks, sample_size, seed_sequence, batch_size):
    """One fresh independent replication, including RNG and estimation."""
    if sample_size < 2:
        raise ValueError("At least two samples are needed for an unbiased sample variance")
    rng = np.random.default_rng(seed_sequence)
    scores = np.empty(sample_size, dtype=np.int64)
    for offset in range(0, sample_size, batch_size):
        count = min(batch_size, sample_size - offset)
        uniforms = rng.integers(0, q, size=(count, len(reference_array)), dtype=np.int32)
        scores[offset:offset + count] = scores_from_uniforms(uniforms, reference_array, chunks)
    return float(np.mean(scores)), float(np.var(scores, ddof=1)), scores


def rmse(errors):
    values = np.asarray(errors, dtype=np.float64)
    return float(np.sqrt(np.mean(values * values)))


def summarize_replicates(rows, case_index, sample_size, config):
    mean_errors = np.asarray([r["mean_relative_error"] for r in rows])
    variance_errors = np.asarray([r["variance_relative_error"] for r in rows])
    rng = np.random.default_rng(np.random.SeedSequence(
        [config["bootstrap_seed"], case_index, sample_size]))
    indices = rng.integers(0, len(rows), size=(config["bootstrap_resamples"], len(rows)))
    bootstrap_rmse = np.sqrt(np.mean(variance_errors[indices] ** 2, axis=1))
    lower, upper = np.quantile(bootstrap_rmse, [0.025, 0.975])
    target = config["variance_relative_rmse_target"]
    value = rmse(variance_errors)
    return {
        "case_id": rows[0]["case_id"], "sample_size": sample_size,
        "independent_replicates": len(rows),
        "mean_relative_rmse": rmse(mean_errors),
        "mean_relative_error_min": float(np.min(mean_errors)),
        "mean_relative_error_max": float(np.max(mean_errors)),
        "mean_relative_error_sd": float(np.std(mean_errors, ddof=1)),
        "variance_relative_rmse": value,
        "variance_relative_error_min": float(np.min(variance_errors)),
        "variance_relative_error_max": float(np.max(variance_errors)),
        "variance_relative_error_sd": float(np.std(variance_errors, ddof=1)),
        "variance_rmse_descriptive_bootstrap_lower": float(lower),
        "variance_rmse_descriptive_bootstrap_upper": float(upper),
        "observed_rmse_below_target": value <= target,
        "descriptive_interval_below_target": float(upper) <= target,
        "variance_relative_rmse_target": target,
        "median_seconds": statistics.median(r["seconds"] for r in rows),
        "min_seconds": min(r["seconds"] for r in rows),
        "max_seconds": max(r["seconds"] for r in rows),
        "total_seconds": sum(r["seconds"] for r in rows),
        "interpretation": "Ten-replication descriptive assessment; neither threshold flag certifies population precision.",
    }


def write_csv(path, rows):
    if not rows:
        return
    fields = list(rows[0])
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(protocol, digest, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    config = protocol["monte_carlo"]
    cases_by_id = {case["id"]: case for case in protocol["cases"]}
    report = {
        "metadata": environment_metadata(digest), "protocol": config,
        "status": "running", "oracles": [], "replicates": [], "summaries": [],
        "interpretation": "Exact raw moments are the oracle. Simulation accuracy is empirical for the frozen cases and budgets. No finite-sample precision guarantee is asserted.",
    }
    output_json = output_dir / "monte_carlo.json"
    for case_index, case_id in enumerate(config["case_ids"]):
        case = cases_by_id[case_id]
        design = prepare_image_graph(case["rows"], case["columns"], case["height"], case["width"])
        reference = reference_for(case)
        oracle = moments_prepared(reference, case["q"], design, cache=True)
        exact_mean, exact_variance = oracle.mean, oracle.variance
        report["oracles"].append({"case_id": case_id, "case": case,
            "mean": exact_mean, "variance": exact_variance,
            "vertices": design.n, "edges": len(design.edges), "weight_sum": design.weight_sum})
        reference_array, chunks = prepare_simulation(reference, case["q"], design, config["edge_chunk_size"])
        for sample_size in config["sample_sizes"]:
            batch_rows = []
            for replicate in range(config["independent_replicates"]):
                entropy = [config["master_seed"], case_index, sample_size, replicate]
                seed = np.random.SeedSequence(entropy)
                gc.collect()
                start = time.perf_counter()
                mean, variance, scores = simulate_moments(reference_array, case["q"], chunks,
                    sample_size, seed, config["batch_size"])
                elapsed = time.perf_counter() - start
                row = {"case_id": case_id, "sample_size": sample_size,
                       "replicate_index": replicate, "seed_entropy": entropy,
                       "mean_estimate": mean, "variance_estimate": variance,
                       "mean_relative_error": exact_relative_error(mean, exact_mean),
                       "variance_relative_error": exact_relative_error(variance, exact_variance),
                       "seconds": elapsed,
                       "raw_scores_sha256": hashlib.sha256(scores.astype("<i8", copy=False).tobytes()).hexdigest()}
                batch_rows.append(row)
                report["replicates"].append(row)
            summary = summarize_replicates(batch_rows, case_index, sample_size, config)
            report["summaries"].append(summary)
            save_json(output_json, report)
            write_csv(output_dir / "monte_carlo_replicates.csv", report["replicates"])
            write_csv(output_dir / "monte_carlo_summary.csv", report["summaries"])
            print(f"Monte Carlo {case_id}, R={sample_size}: relative variance RMSE {summary['variance_relative_rmse']:.6g}, descriptive interval [{summary['variance_rmse_descriptive_bootstrap_lower']:.6g}, {summary['variance_rmse_descriptive_bootstrap_upper']:.6g}]", flush=True)
    report["status"] = "completed"
    report["metadata"]["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save_json(output_json, report)
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
