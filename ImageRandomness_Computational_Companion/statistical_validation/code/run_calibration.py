"""Frozen-protocol null calibration of exact-moment Gaussian rules.

The data-generating model is explicit for every case. Fixed-reference folded
laws and two-random-image triangular laws are separate null models. All
replications within a case are independent. Wilson intervals describe the
rejection probability of that fixed Gaussian rule, not the uncertainty of a
reused rank-calibration pool. Nominal coverage is not an equivalence test.

Run with the selected protocol's SHA256. The complete raw integer score
vectors are saved for independent reanalysis. No reference parameters are
fitted to the simulated validation sample.
"""
from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
from statistics import NormalDist
import sys
import time

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
from moment_engine import prepare_image_graph
from statistical_core import (generate_null, null_moments, prepare_score,
                              reference_for, score_fields, wilson_interval)

ROOT = Path(__file__).resolve().parent.parent
NORMAL = NormalDist()


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json_safe(value):
    if isinstance(value, Fraction):
        return {"numerator": str(value.numerator), "denominator": str(value.denominator)}
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_json_safe(payload), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save_scores(path, arrays):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def save_csv(path, rows):
    if not rows:
        return
    fields = list(rows[0])
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def environment_metadata(protocol_hash):
    cpu_model = "unknown"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    paths = [Path(__file__), Path(__file__).with_name("statistical_core.py"),
             Path(__file__).with_name("moment_engine.py"),
             Path(__file__).with_name("alphabet_baseline.py")]
    return {
        "protocol_sha256": protocol_hash,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version, "numpy": np.__version__,
        "platform": platform.platform(), "cpu_model": cpu_model,
        "thread_environment": {k: os.environ[k] for k in
            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
        "code_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in paths},
    }


def parse_pmf(value):
    """Protocol PMFs use exact rational strings or integer entries.

    A one-dimensional vector is an iid law. Nested vectors give one law per
    pixel. Rational pairs are not used, avoiding ambiguity with nested laws.
    """
    if value is None:
        return None
    if not isinstance(value, list) or not value:
        raise ValueError("pmf must be a nonempty vector or list of vectors")
    if isinstance(value[0], list):
        return [[Fraction(x) for x in row] for row in value]
    return [Fraction(x) for x in value]


def case_inputs(case):
    M, N, q = (int(case[k]) for k in ("rows", "columns", "q"))
    reference = None
    if case["model"] == "fixed":
        reference_spec = case.get("reference", "palette")
        reference = (reference_for(reference_spec, M, N, q)
                     if isinstance(reference_spec, str) else tuple(reference_spec))
    pmf = parse_pmf(case.get("pmf"))
    return M, N, q, reference, pmf


def _proportion(count, sample_size, confidence):
    low, high = wilson_interval(int(count), int(sample_size), confidence=confidence)
    return {"count": int(count), "rate": float(count/sample_size),
            "wilson_lower": float(low), "wilson_upper": float(high)}


def empirical_normal_cdf_distance(z):
    """Descriptive two-sided empirical CDF discrepancy; no KS decision."""
    values = np.sort(z)
    n = len(values)
    cdf = np.fromiter((NORMAL.cdf(float(x)) for x in values), dtype=np.float64, count=n)
    upper = np.arange(1, n+1, dtype=np.float64)/n
    lower = np.arange(n, dtype=np.float64)/n
    return float(max(np.max(upper-cdf), np.max(cdf-lower)))


def summarize_scores(scores, mean, variance, naive_variance, *, alphas,
                     quantile_probabilities, confidence=0.95):
    """Summarize independent raw S values against an externally fixed oracle."""
    scores = np.asarray(scores)
    if scores.ndim != 1 or len(scores) < 2 or not np.issubdtype(scores.dtype, np.integer):
        raise ValueError("at least two integer raw scores are required")
    if variance <= 0 or naive_variance <= 0:
        raise ValueError("Gaussian calibration requires positive exact and diagnostic variances")
    n = len(scores)
    mu, var, naive = float(mean), float(variance), float(naive_variance)
    values = scores.astype(np.float64)
    z = (values-mu)/math.sqrt(var)
    z_naive = (values-mu)/math.sqrt(naive)
    observed_mean = float(np.mean(values))
    observed_variance = float(np.var(values, ddof=1))
    center = values-observed_mean
    second_central = float(np.mean(center**2))
    skewness = float(np.mean(center**3)/second_central**1.5) if second_central else None
    excess_kurtosis = float(np.mean(center**4)/second_central**2-3) if second_central else None
    probabilities = list(map(float, quantile_probabilities))
    quantiles = np.quantile(z, probabilities, method="linear")
    quantile_rows = [{"probability": p, "empirical_standardized_quantile": float(value),
                      "normal_quantile": NORMAL.inv_cdf(p),
                      "difference": float(value)-NORMAL.inv_cdf(p)}
                     for p, value in zip(probabilities, quantiles)]
    rejection_rows = []
    for alpha in alphas:
        alpha = float(alpha)
        if not 0 < alpha < 1:
            raise ValueError("alphas must lie strictly between zero and one")
        critical = NORMAL.inv_cdf(1-alpha/2)
        rejected = {}
        for method, standardized in (("exact_moment_gaussian", z),
                                      ("pair_occurrence_independent_gaussian", z_naive)):
            lower = standardized < -critical
            upper = standardized > critical
            total = lower | upper
            rejected[method] = total
            row = {"method": method, "alpha": alpha, "critical_z": critical,
                   "sample_size": n,
                   "lower_tail": _proportion(np.count_nonzero(lower), n, confidence),
                   "upper_tail": _proportion(np.count_nonzero(upper), n, confidence),
                   "two_sided": _proportion(np.count_nonzero(total), n, confidence)}
            row["observed_size_error"] = row["two_sided"]["rate"]-alpha
            rejection_rows.append(row)
        diagnostic = rejected["pair_occurrence_independent_gaussian"]
        exact = rejected["exact_moment_gaussian"]
        for row in rejection_rows[-2:]:
            row["paired_naive_only_rejections"] = int(np.count_nonzero(diagnostic & ~exact))
            row["paired_exact_only_rejections"] = int(np.count_nonzero(exact & ~diagnostic))
    return {
        "sample_size": n, "sample_mean": observed_mean,
        "sample_variance_ddof1": observed_variance,
        "mean_error": observed_mean-mu,
        "mean_error_in_monte_carlo_standard_errors": (observed_mean-mu)/math.sqrt(var/n),
        "variance_ratio_to_exact": observed_variance/var,
        "variance_relative_error": observed_variance/var-1,
        "variance_inflation_over_pair_occurrence": var/naive,
        "standardized_sample_mean": float(np.mean(z)),
        "standardized_sample_variance_ddof1": float(np.var(z, ddof=1)),
        "sample_skewness_uncorrected": skewness,
        "sample_excess_kurtosis_uncorrected": excess_kurtosis,
        "empirical_normal_cdf_distance": empirical_normal_cdf_distance(z),
        "quantiles": quantile_rows, "rejections": rejection_rows,
        "confidence": confidence,
        "interval_interpretation": "Pointwise Wilson intervals for the fixed rule's true rejection probability under the specified null. No simultaneous or equivalence claim.",
    }


def flatten_rejections(case, summary):
    rows = []
    for entry in summary["rejections"]:
        row = {"case_id": case["id"], "model": case["model"],
               "rows": case["rows"], "columns": case["columns"],
               "height": case["height"], "width": case["width"], "q": case["q"],
               "reference": case.get("reference", ""),
               "method": entry["method"], "alpha": entry["alpha"],
               "sample_size": summary["sample_size"], "critical_z": entry["critical_z"]}
        for category in ("lower_tail", "upper_tail", "two_sided"):
            for key, value in entry[category].items():
                row[f"{category}_{key}"] = value
        for key in ("observed_size_error", "paired_naive_only_rejections", "paired_exact_only_rejections"):
            row[key] = entry[key]
        rows.append(row)
    return rows


def run(protocol, protocol_hash, output_dir):
    config = protocol["calibration"]
    cases = config["cases"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"metadata": environment_metadata(protocol_hash), "status": "running",
              "protocol": config, "cases": [],
              "scope": "Independent replicated null calibration only. Exact moments and Gaussian tail accuracy are distinct claims."}
    score_arrays = {}
    rows = []
    for case_index, case in enumerate(cases):
        case_id = case["id"]
        if not re.fullmatch(r"[A-Za-z0-9_]+", case_id) or case_id in score_arrays:
            raise ValueError("case identifiers must be distinct alphanumeric/underscore names")
        M, N, q, reference, pmf = case_inputs(case)
        design = prepare_image_graph(M, N, case["height"], case["width"])
        if (q-1)*design.absolute_weight_sum > 2**53:
            raise ValueError("protocol raw scores exceed the exact-integer range used for float64 standardization")
        started = time.perf_counter()
        moments = null_moments(case["model"], design, q, reference=reference, pmf=pmf)
        moment_seconds = time.perf_counter()-started
        index = prepare_score(design, edge_chunk_size=config.get("edge_chunk_size", 8192))
        sample_size = int(case.get("sample_size", config["sample_size"]))
        batch_size = int(config.get("batch_size", 32))
        if sample_size < 2 or batch_size < 1:
            raise ValueError("sample_size>=2 and batch_size>=1 are required")
        entropy = [int(config["master_seed"]), case_index]
        rng = np.random.default_rng(np.random.SeedSequence(entropy))
        scores = np.empty(sample_size, dtype=np.int64)
        started = time.perf_counter()
        for offset in range(0, sample_size, batch_size):
            count = min(batch_size, sample_size-offset)
            fields = generate_null(case["model"], rng, count, M, N, q,
                                   reference=reference, pmf=pmf)
            scores[offset:offset+count] = score_fields(fields, index)
        simulation_seconds = time.perf_counter()-started
        summary = summarize_scores(scores, moments.mean, moments.variance, moments.naive_variance,
            alphas=config.get("alphas", [0.05, 0.01, 0.001]),
            quantile_probabilities=config.get("quantile_probabilities", [0.001,0.005,0.025,0.5,0.975,0.995,0.999]),
            confidence=config.get("confidence", 0.95))
        record = {"case": case, "case_index": case_index, "seed_entropy": entropy,
                  "vertices": design.n, "edges": len(design.edges), "weight_sum": design.weight_sum,
                  "exact_mean": moments.mean, "exact_variance": moments.variance,
                  "naive_pair_occurrence_variance": moments.naive_variance,
                  "moment_seconds": moment_seconds, "simulation_seconds": simulation_seconds,
                  "raw_scores_sha256": hashlib.sha256(scores.astype("<i8", copy=False).tobytes()).hexdigest(),
                  "raw_scores_member": case_id, "summary": summary}
        report["cases"].append(record)
        score_arrays[case_id] = scores
        rows.extend(flatten_rejections(case, summary))
        save_scores(output_dir / "calibration_scores.npz", score_arrays)
        save_csv(output_dir / "calibration.csv", rows)
        save_json(output_dir / "calibration.json", report)
        rates = ", ".join(f"alpha={r['alpha']:g}: {r['two_sided']['rate']:.5f}"
                          for r in summary["rejections"] if r["method"] == "exact_moment_gaussian")
        print(f"Calibration {case_index+1}/{len(cases)} {case_id}: {rates}", flush=True)
    report["status"] = "completed"
    report["metadata"]["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    report["summary"] = {"cases": len(report["cases"]), "method_alpha_rows": len(rows),
                         "total_independent_images": sum(r["summary"]["sample_size"] for r in report["cases"])}
    save_json(output_dir / "calibration.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "results" / "protocol.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--authorized-protocol-sha256", required=True)
    args = parser.parse_args()
    digest = sha256_file(args.protocol)
    if digest != args.authorized_protocol_sha256:
        raise ValueError("Protocol SHA256 does not match the authorized design")
    run(json.loads(args.protocol.read_text()), digest, args.output_dir)


if __name__ == "__main__":
    main()
