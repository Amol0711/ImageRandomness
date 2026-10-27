"""Frozen-protocol paired power study with independently calibrated rank rules.

All statistics use identical simulated difference fields within a model. Null
calibration is an independent shared bank. The exact nominal rank level is
marginal over bank and observation; fixed-bank evaluation targets its random
conditional rejection probability. Paired Clopper-Pearson differences provide
at least 95% pointwise conditional coverage, without a simultaneous assertion.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np
import scipy
from scipy.stats import beta

from moment_engine import prepare_image_graph
from statistical_core import (
    exact_discrepancy, fit_rank_threshold, generate_null, null_moments,
    prepare_score, rank_size_interval, reject_rank_threshold, score_fields,
    wilson_interval,
)
from alternative_analysis import copying_pairs, exact_mean_shift

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROTOCOL = ROOT / "results" / "protocol.json"


@dataclass(frozen=True)
class Statistic:
    name: str
    mean: Fraction
    variance: Fraction
    design: object = None
    prepared: object = None
    window: tuple | None = None


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_safe(value):
    if isinstance(value, Fraction):
        return {"numerator": str(value.numerator), "denominator": str(value.denominator)}
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_statistics(geometry, edge_chunk_size):
    M, N, q = (geometry[key] for key in ("rows", "columns", "q"))
    n = M * N
    p = Fraction(q - 1, q)
    mean_d = Fraction(q*q - 1, 3*q)
    var_d = Fraction((q*q - 1)*(q*q + 2), 18*q*q)
    result = [Statistic("NPCR_count", n*p, n*p*(1-p)),
              Statistic("UACI_sum", n*mean_d, n*var_d)]
    for a, b in geometry["windows"]:
        design = prepare_image_graph(M, N, a, b)
        moments = null_moments("two_random", design, q)
        result.append(Statistic(f"S_{a}x{b}", moments.mean, moments.variance,
                                design, prepare_score(design, edge_chunk_size), (a,b)))
    return result


def apply_alternative(fields, rng, geometry, alternative):
    """Mutate independent triangular fields by the explicitly defined model."""
    if alternative is None:
        return fields
    kind = alternative["kind"]
    if kind == "copying":
        theta = float(alternative["theta"])
        if not 0 <= theta <= 1:
            raise ValueError("copying probability must lie in [0,1]")
        pairs = copying_pairs(geometry["rows"], geometry["columns"], alternative["lag"])
        if pairs:
            donors = np.asarray([i for i,j in pairs], dtype=np.intp)
            recipients = np.asarray([j for i,j in pairs], dtype=np.intp)
            active = rng.random((len(fields), len(pairs))) < theta
            fields[:, recipients] = np.where(active, fields[:, donors], fields[:, recipients])
    elif kind == "partial_identity":
        epsilon = float(alternative["epsilon"])
        if not 0 <= epsilon <= 1:
            raise ValueError("identity probability must lie in [0,1]")
        fields[rng.random(fields.shape) < epsilon] = 0
    else:
        raise ValueError(f"unknown alternative kind {kind!r}")
    return fields


def statistics_from_fields(fields, statistics):
    values = np.empty((len(fields), len(statistics)), dtype=np.int64)
    values[:, 0] = np.count_nonzero(fields, axis=1)
    values[:, 1] = np.sum(fields, axis=1, dtype=np.int64)
    for j, statistic in enumerate(statistics[2:], 2):
        values[:, j] = score_fields(fields, statistic.prepared)
    return values


def generate_statistics(geometry, statistics, count, master_seed,
                        geometry_index, phase, batch_size, alternative=None):
    """Bounded-memory image batches; save only the paired integer statistics."""
    entropy = [master_seed, geometry_index, phase, 0]
    rng = np.random.default_rng(np.random.SeedSequence(entropy))
    result = np.empty((count, len(statistics)), dtype=np.int64)
    for start in range(0, count, batch_size):
        size = min(batch_size, count - start)
        fields = generate_null("two_random", rng, size,
                               geometry["rows"], geometry["columns"], geometry["q"])
        apply_alternative(fields, rng, geometry, alternative)
        result[start:start+size] = statistics_from_fields(fields, statistics)
    return result, entropy


def ties_for(count, stat_count, master_seed, geometry_index, phase):
    entropy = [master_seed, geometry_index, phase, 1]
    rng = np.random.default_rng(np.random.SeedSequence(entropy))
    return rng.random((count, stat_count)), entropy


def clopper_pearson(successes, trials, confidence=.95):
    """Equal-tailed exact binomial interval, including zero/all successes."""
    if not 0 <= successes <= trials or trials <= 0:
        raise ValueError("invalid binomial counts")
    error = 1-confidence
    lower = 0.0 if successes == 0 else float(beta.ppf(error/2, successes, trials-successes+1))
    upper = 1.0 if successes == trials else float(beta.ppf(1-error/2, successes+1, trials-successes))
    return lower, upper


def paired_difference(first, second):
    """Bonferroni difference of two 97.5% exact discordance intervals."""
    first, second = np.asarray(first, dtype=bool), np.asarray(second, dtype=bool)
    if first.shape != second.shape or first.ndim != 1 or not len(first):
        raise ValueError("paired decisions must be nonempty equal-length vectors")
    n = len(first)
    n10 = int(np.count_nonzero(first & ~second))
    n01 = int(np.count_nonzero(~first & second))
    lower10, upper10 = clopper_pearson(n10, n, .975)
    lower01, upper01 = clopper_pearson(n01, n, .975)
    return {"sample_size": n, "first_only": n10, "second_only": n01,
            "difference": (n10-n01)/n,
            "difference_lower": lower10-upper01,
            "difference_upper": upper10-lower01,
            "interval": "two 97.5% Clopper-Pearson discordance intervals; at least 95% pointwise conditional coverage"}


def comparison_pairs(statistics):
    names = [statistic.name for statistic in statistics]
    baseline = names[2]
    comparisons = []
    for name in names[2:]:
        comparisons.extend([(name, "NPCR_count"), (name, "UACI_sum")])
        if name != baseline:
            comparisons.append((name, baseline))
    return comparisons


def expected_mean(statistic, geometry, alternative):
    if alternative is None:
        return statistic.mean
    n, q = geometry["rows"]*geometry["columns"], geometry["q"]
    if alternative["kind"] == "copying":
        if statistic.design is None:
            return statistic.mean  # every one-point marginal is unchanged
        shift = exact_mean_shift(statistic.design, q, alternative["lag"],
                                 Fraction(str(alternative["theta"])),
                                 geometry["rows"], geometry["columns"])
        return statistic.mean - shift
    epsilon = Fraction(str(alternative["epsilon"]))
    if statistic.design is None:
        return (1-epsilon)*statistic.mean
    mean_d = Fraction(q*q-1, 3*q)
    edge_mean = statistic.mean / statistic.design.weight_sum
    modified_edge_mean = (1-epsilon)**2*edge_mean + 2*epsilon*(1-epsilon)*mean_d
    return statistic.design.weight_sum*modified_edge_mean


def moments_diagnostic(values, statistic, geometry, alternative):
    expected = expected_mean(statistic, geometry, alternative)
    observed = float(np.mean(values))
    sample_variance = float(np.var(values, ddof=1))
    se = math.sqrt(sample_variance/len(values))
    error = float(Fraction.from_float(observed) - expected)
    return {"statistic": statistic.name, "expected_mean": expected,
            "observed_mean": observed, "sample_variance": sample_variance,
            "mean_error": error, "mean_standard_error_estimated": se,
            "mean_error_in_estimated_se": error/se if se else None,
            "expected_standardized_mean_decrease": float(statistic.mean-expected)/math.sqrt(float(statistic.variance)),
            "observed_standardized_mean_decrease": (float(statistic.mean)-observed)/math.sqrt(float(statistic.variance))}


def evaluate(raw, ties, statistics, thresholds, alphas):
    decisions = {}
    for j, statistic in enumerate(statistics):
        discrepancies = exact_discrepancy(raw[:,j], statistic.mean)
        for alpha in alphas:
            decisions[statistic.name, alpha] = reject_rank_threshold(
                discrepancies, thresholds[statistic.name, alpha], ties=ties[:,j])
    return decisions


def metadata(protocol_digest):
    source_names = ["run_power.py", "statistical_core.py", "alternative_analysis.py",
                    "moment_engine.py", "alphabet_baseline.py"]
    return {"protocol_sha256": protocol_digest,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
            "platform": platform.platform(),
            "thread_environment": {key: os.environ[key] for key in
                ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
            "source_sha256": {name: sha256_file(Path(__file__).with_name(name)) for name in source_names}}


def run(protocol, digest, output_dir):
    config = protocol["power"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"metadata": metadata(digest), "protocol": config, "status": "running",
              "geometries": [], "thresholds": [], "groups": [], "rates": [], "paired_differences": []}
    alphas = config["alphas"]
    for geometry_index, geometry in enumerate(config["geometries"]):
        statistics = build_statistics(geometry, config["edge_chunk_size"])
        names = [statistic.name for statistic in statistics]
        if names != config["statistics"]:
            raise ValueError("computed statistic order differs from the frozen protocol")
        report["geometries"].append({"geometry": geometry,
            "statistics": [{"name": s.name, "null_mean": s.mean, "null_variance": s.variance,
                            "window": s.window,
                            "edges": len(s.design.edges) if s.design else None,
                            "weight_sum": s.design.weight_sum if s.design else None} for s in statistics]})
        start = time.perf_counter()
        bank, field_seed = generate_statistics(geometry, statistics, config["calibration_size"],
            config["master_seed"], geometry_index, 0, config["batch_size"])
        elapsed = time.perf_counter()-start
        bank_ties, tie_seed = ties_for(len(bank), len(statistics), config["master_seed"], geometry_index, 0)
        bank_file = output_dir/f"power_{geometry['id']}_calibration.npz"
        np.savez_compressed(bank_file, statistics=bank, statistic_names=np.asarray(names))
        thresholds = {}
        for j, statistic in enumerate(statistics):
            discrepancies = exact_discrepancy(bank[:,j], statistic.mean)
            for alpha in alphas:
                threshold = fit_rank_threshold(discrepancies, alpha, ties=bank_ties[:,j])
                thresholds[statistic.name, alpha] = threshold
                beta_interval = rank_size_interval(threshold)
                report["thresholds"].append({"geometry_id": geometry["id"],
                    "statistic": statistic.name, "alpha": alpha, **asdict(threshold),
                    "conditional_size_beta_interval": beta_interval})
        report["groups"].append({"geometry_id": geometry["id"], "model_id": "calibration",
            "sample_size": len(bank), "field_seed_entropy": field_seed, "tie_seed_entropy": tie_seed,
            "raw_file": bank_file.name, "raw_file_sha256": sha256_file(bank_file),
            "simulation_seconds": elapsed,
            "raw_statistics_sha256": hashlib.sha256(bank.astype('<i8',copy=False).tobytes()).hexdigest()})
        save_json(output_dir/"power.json", report)
        print(f"power {geometry['id']}: independent calibration bank complete ({len(bank)} fields)", flush=True)

        groups = [("null", None, config["null_sample_size"], 1)]
        groups += [(alternative["id"], alternative, config["alternative_sample_size"], 2+index)
                   for index, alternative in enumerate(config["alternatives"])]
        for model_id, alternative, count, phase in groups:
            start = time.perf_counter()
            raw, field_seed = generate_statistics(geometry, statistics, count, config["master_seed"],
                geometry_index, phase, config["batch_size"], alternative)
            elapsed = time.perf_counter()-start
            ties, tie_seed = ties_for(count, len(statistics), config["master_seed"], geometry_index, phase)
            decisions = evaluate(raw, ties, statistics, thresholds, alphas)
            raw_file = output_dir/f"power_{geometry['id']}_{model_id}.npz"
            np.savez_compressed(raw_file, statistics=raw, statistic_names=np.asarray(names))
            report["groups"].append({"geometry_id": geometry["id"], "model_id": model_id,
                "alternative": alternative, "sample_size": count,
                "field_seed_entropy": field_seed, "tie_seed_entropy": tie_seed,
                "raw_file": raw_file.name, "raw_file_sha256": sha256_file(raw_file),
                "raw_statistics_sha256": hashlib.sha256(raw.astype('<i8',copy=False).tobytes()).hexdigest(),
                "simulation_seconds": elapsed,
                "moments": [moments_diagnostic(raw[:,j], statistic, geometry, alternative)
                            for j,statistic in enumerate(statistics)]})
            for statistic in statistics:
                for alpha in alphas:
                    rejected = int(np.count_nonzero(decisions[statistic.name, alpha]))
                    lower, upper = wilson_interval(rejected, count)
                    report["rates"].append({"geometry_id": geometry["id"], "model_id": model_id,
                        "statistic": statistic.name, "alpha": alpha, "sample_size": count,
                        "rejections": rejected, "rate": rejected/count,
                        "wilson_lower": lower, "wilson_upper": upper,
                        "interval_target": "fixed-bank conditional rejection probability"})
            for first, second in comparison_pairs(statistics):
                for alpha in alphas:
                    row = {"geometry_id": geometry["id"], "model_id": model_id,
                           "first_statistic": first, "second_statistic": second, "alpha": alpha}
                    row.update(paired_difference(decisions[first,alpha], decisions[second,alpha]))
                    report["paired_differences"].append(row)
            save_json(output_dir/"power.json", report)
            write_csv(output_dir/"power_rates.csv", report["rates"])
            write_csv(output_dir/"power_paired.csv", report["paired_differences"])
            print(f"power {geometry['id']}: {model_id} complete ({count} paired fields)", flush=True)
    report["status"] = "completed"
    report["metadata"]["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save_json(output_dir/"power.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-dir", type=Path, default=ROOT/"results")
    parser.add_argument("--authorized-protocol-sha256", required=True)
    args = parser.parse_args()
    digest = sha256_file(args.protocol)
    if digest != args.authorized_protocol_sha256:
        raise ValueError("protocol checksum differs from the authorized frozen value")
    run(json.loads(args.protocol.read_text()), digest, args.output_dir)


if __name__ == "__main__":
    main()
