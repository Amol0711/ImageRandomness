"""Reproduce the moment-based predictions accompanying the null study.

Run ``python code/calibration_predictions.py`` from statistical_validation, or
invoke this file by its absolute path. The script recalculates the exact mean,
variance and pair-occurrence variance for twelve specified null configurations. It checks the recorded rational moments, reconstructs both Gaussian
rules from the saved integer scores, and checks their counts and Wilson
intervals before writing calibration_predictions.json and .csv in results.

The variance ratio is rational and exact. The predicted rejection probability
uses a normal tail and is a distributional approximation, not an exact level.
No observations, random seeds, thresholds or original result files are changed.
"""
from __future__ import annotations

import csv
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np

from moment_engine import prepare_image_graph
from statistical_core import null_moments, reference_for, wilson_interval


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
ALPHA = 0.01
CONFIDENCE = 0.95
CASE_IDS = (
    "random_q2_4", "random_q256_64", "random_q65536_64", "random_window4",
    "random_rectangle", "random_full_window", "fixed_checker_16", "fixed_split_16",
    "fixed_diverse_q65536_64", "fixed_palette_q65536_window4",
    "known_nonuniform_64", "known_sparse_16",
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fraction_record(value):
    return {"numerator": str(value.numerator), "denominator": str(value.denominator)}


def recorded_fraction(value):
    return Fraction(int(value["numerator"]), int(value["denominator"]))


def require_equal(first, second, description):
    if first != second:
        raise ArithmeticError(f"{description} differs from the recorded calculation")


def recompute_case(record, scores):
    case = record["case"]
    case_id = case["id"]
    rows, columns, q = (int(case[key]) for key in ("rows", "columns", "q"))
    design = prepare_image_graph(rows, columns, case["height"], case["width"])
    reference = None
    if case["model"] == "fixed":
        spec = case["reference"]
        reference = reference_for(spec, rows, columns, q) if isinstance(spec, str) else tuple(spec)
    pmf = case.get("pmf")
    if pmf is not None:
        pmf = ([[Fraction(x) for x in row] for row in pmf]
               if isinstance(pmf[0], list) else [Fraction(x) for x in pmf])
    moments = null_moments(case["model"], design, q, reference=reference, pmf=pmf)
    for value, key in ((moments.mean, "exact_mean"),
                       (moments.variance, "exact_variance"),
                       (moments.naive_variance, "naive_pair_occurrence_variance")):
        require_equal(value, recorded_fraction(record[key]), f"{case_id} {key}")
    if moments.variance <= 0 or moments.naive_variance <= 0:
        raise ArithmeticError(f"{case_id} requires positive variances")
    require_equal(hashlib.sha256(scores.tobytes()).hexdigest(), record["raw_scores_sha256"],
                  f"{case_id} raw-score hash")
    require_equal(len(scores), record["summary"]["sample_size"], f"{case_id} sample size")
    critical = NormalDist().inv_cdf(1 - ALPHA / 2)
    values = scores.astype(np.float64)
    observed = {}
    for method, variance in (("exact_moment_gaussian", moments.variance),
                             ("pair_occurrence_independent_gaussian", moments.naive_variance)):
        normalized = (values - float(moments.mean)) / math.sqrt(float(variance))
        lower = int(np.count_nonzero(normalized < -critical))
        upper = int(np.count_nonzero(normalized > critical))
        count = lower + upper
        archived = next(row for row in record["summary"]["rejections"]
                        if row["method"] == method and row["alpha"] == ALPHA)
        for actual, part in ((lower, "lower_tail"), (upper, "upper_tail"), (count, "two_sided")):
            require_equal(actual, archived[part]["count"], f"{case_id} {method} {part}")
        interval = wilson_interval(count, len(scores), confidence=CONFIDENCE)
        for actual, key in zip(interval, ("wilson_lower", "wilson_upper")):
            if not math.isclose(actual, archived["two_sided"][key], rel_tol=0, abs_tol=1e-14):
                raise ArithmeticError(f"{case_id} {method} Wilson interval differs")
        observed[method] = {"lower_count": lower, "upper_count": upper, "count": count,
                            "rate": count / len(scores), "wilson_interval": list(interval)}
    ratio = moments.variance / moments.naive_variance
    prediction = math.erfc(critical / math.sqrt(2 * float(ratio)))
    return {
        "case": case, "sample_size": len(scores), "alpha": ALPHA,
        "exact_mean": fraction_record(moments.mean),
        "exact_variance": fraction_record(moments.variance),
        "pair_occurrence_variance": fraction_record(moments.naive_variance),
        "variance_ratio": fraction_record(ratio), "variance_ratio_float": float(ratio),
        "gaussian_pair_occurrence_prediction": prediction,
        "observations": observed,
        "verification": "exact moments, saved-score hash, both tail counts and Wilson intervals agree",
    }


def main():
    original = json.loads((RESULTS / "calibration.json").read_text())
    records = {item["case"]["id"]: item for item in original["cases"]}
    with np.load(RESULTS / "calibration_scores.npz", allow_pickle=False) as arrays:
        rows = [recompute_case(records[name], arrays[records[name]["raw_scores_member"]])
                for name in CASE_IDS]
    sources = (Path(__file__), Path(__file__).with_name("statistical_core.py"),
               Path(__file__).with_name("moment_engine.py"),
               Path(__file__).with_name("alphabet_baseline.py"),
               RESULTS / "calibration.json", RESULTS / "calibration_scores.npz")
    result = {
        "status": "verified", "alpha": ALPHA, "confidence": CONFIDENCE,
        "prediction": "2 Phi(-z_(1-alpha/2)/sqrt(V_exact/V_pair))",
        "scope": "Normal-tail approximation using exact moments; not an exact rejection probability. "
                 "The fixed-square uniform-image normal theorem does not assert these predictions "
                 "for rectangular or prescribed nonuniform-law cases.",
        "provenance_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in sources},
        "cases": rows,
    }
    (RESULTS / "calibration_predictions.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n")
    flat = []
    for row in rows:
        case = row["case"]
        exact = row["observations"]["exact_moment_gaussian"]
        pair = row["observations"]["pair_occurrence_independent_gaussian"]
        flat.append({
            "case_id": case["id"], "model": case["model"], "rows": case["rows"],
            "columns": case["columns"], "height": case["height"], "width": case["width"],
            "q": case["q"], "reference": case.get("reference", ""),
            "alpha": ALPHA, "sample_size": row["sample_size"],
            "variance_ratio_numerator": row["variance_ratio"]["numerator"],
            "variance_ratio_denominator": row["variance_ratio"]["denominator"],
            "variance_ratio": row["variance_ratio_float"],
            "exact_variance_count": exact["count"], "exact_variance_rate": exact["rate"],
            "exact_variance_wilson_lower": exact["wilson_interval"][0],
            "exact_variance_wilson_upper": exact["wilson_interval"][1],
            "pair_occurrence_gaussian_prediction": row["gaussian_pair_occurrence_prediction"],
            "pair_occurrence_count": pair["count"], "pair_occurrence_rate": pair["rate"],
        })
    with (RESULTS / "calibration_predictions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    print(f"Verified exact moments and saved-score decisions for {len(rows)} cases.")
    print("Wrote results/calibration_predictions.json and results/calibration_predictions.csv.")


if __name__ == "__main__":
    main()
