"""Check the copied-field generator and NPCR probabilities for the saved bank.

The NPCR count is n minus the number of zero difference pixels. For p=1/q,
one designated donor-recipient pair contributes 0, 1 or 2 zeros with PGF

  (1-theta) * ((1-p) + p*z)**2 + theta * ((1-p) + p*z*z).

Different pairs and untouched pixels are independent under the declared model.
Integer polynomial coefficients give the rejection probability at the actual
saved rank threshold, including its binary64 tie cutoff interpreted exactly as
a rational number. Only coefficients inside the acceptance region are needed.

This diagnostic preserves every original observation. It exhaustively checks
small fields and activation masks, independently replays the NPCR and UACI
sequences from their recorded seeds, and compares the existing rejection counts
with the exact-law bank-conditional probabilities. It generates no new trial
sample and does not replace a result because of a Monte Carlo discrepancy.
"""
from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
from itertools import product
import json
from math import prod, sqrt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from export_provenance import recorded_digest, matches_recorded_digest

import numpy as np
from scipy.stats import binomtest

from alternative_analysis import copying_pairs
from run_power import apply_alternative


ROOT = Path(__file__).resolve().parent.parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def decimal_string(value, precision=40):
    with localcontext() as context:
        context.prec = precision
        return str(Decimal(value.numerator) / Decimal(value.denominator))


def independent_pairs(rows, columns, lag):
    """Column classification, independent of the production block-loop code."""
    complete = columns - columns % (2 * lag)
    return tuple((r * columns + c - lag, r * columns + c)
                 for r in range(rows) for c in range(complete)
                 if c % (2 * lag) >= lag)


def multiply_truncated(coefficients, factor, maximum_degree):
    result = [0] * min(maximum_degree + 1,
                       len(coefficients) + len(factor) - 1)
    for j, a in enumerate(coefficients):
        for k, b in enumerate(factor):
            if j + k < len(result):
                result[j + k] += a * b
    return result


def zero_count_coefficients(n, pairs, q, theta, maximum_degree):
    """Exact truncated PGF, returned as integer coefficients/common denominator.

    Truncation does not approximate the retained coefficients, since all
    factors have nonnegative powers. Rejection is one minus acceptance.
    """
    theta = Fraction(theta)
    if not (0 <= 2 * pairs <= n and 0 <= theta <= 1 and q >= 2):
        raise ValueError("invalid copying model")
    t, d = theta.numerator, theta.denominator
    pair = ((d-t)*(q-1)**2 + t*q*(q-1),
            2*(d-t)*(q-1), (d-t) + t*q)
    pair_denominator = d*q*q
    assert sum(pair) == pair_denominator
    coefficients = [1]
    for _ in range(pairs):
        coefficients = multiply_truncated(coefficients, pair, maximum_degree)
    for _ in range(n - 2*pairs):
        coefficients = multiply_truncated(coefficients, (q-1, 1), maximum_degree)
    denominator = pair_denominator**pairs * q**(n - 2*pairs)
    return coefficients, denominator


def exact_rejection_probability(n, pairs, q, theta, threshold):
    center = Fraction(n, q)  # null mean of the zero count
    if center.denominator != 1:
        raise ValueError("this saved-bank diagnostic expects an integer center")
    radius = int(threshold["value"])
    lower, upper = int(center)-radius, int(center)+radius
    coefficients, denominator = zero_count_coefficients(
        n, pairs, q, theta, min(n, upper))
    interior = sum(coefficients[max(0, lower+1):min(n, upper-1)+1])
    boundary = sum(coefficients[j] for j in {lower, upper}
                   if 0 <= j <= n)
    cutoff = Fraction.from_float(float(threshold["tie"]))
    return 1 - Fraction(interior, denominator) - cutoff*Fraction(boundary, denominator)


class FixedMask:
    def __init__(self, mask):
        self.mask = np.asarray(mask, dtype=bool)

    def random(self, shape):
        if tuple(shape) != (1, len(self.mask)):
            raise AssertionError("unexpected activation-mask shape")
        return np.where(self.mask[None, :], 0.0, 0.999)


def exact_small_checks():
    geometry_count = 0
    for rows, columns, lag in product(range(1, 5), range(1, 10), range(1, 6)):
        pairs = independent_pairs(rows, columns, lag)
        assert pairs == copying_pairs(rows, columns, lag)
        endpoints = [i for pair in pairs for i in pair]
        assert len(endpoints) == len(set(endpoints))
        assert len(pairs) == rows*lag*(columns//(2*lag))
        assert all(i//columns == j//columns and j-i == lag for i, j in pairs)
        geometry_count += 1
    configurations = 0
    laws_checked = []
    for rows, columns, q, lag in [(1, 4, 2, 1), (1, 5, 2, 2),
                                  (2, 3, 2, 1), (1, 4, 3, 1)]:
        pairs = independent_pairs(rows, columns, lag)
        n, theta = rows*columns, Fraction(1, 4)
        # Derive the difference multiplicities from all underlying (U,V).
        counts = Counter(abs(u-v) for u in range(q) for v in range(q))
        assert counts[0] == q and sum(counts.values()) == q*q
        histogram = [Fraction() for _ in range(n+1)]
        geometry = {"rows": rows, "columns": columns, "q": q}
        alternative = {"kind": "copying", "lag": lag, "theta": float(theta)}
        for field in product(range(q), repeat=n):
            mass = Fraction(prod(counts[x] for x in field), q**(2*n))
            for mask in product((False, True), repeat=len(pairs)):
                active = sum(mask)
                weight = mass*theta**active*(1-theta)**(len(pairs)-active)
                expected = list(field)
                for enabled, (donor, recipient) in zip(mask, pairs):
                    if enabled:
                        expected[recipient] = field[donor]
                observed = apply_alternative(
                    np.asarray([field], dtype=np.int32), FixedMask(mask),
                    geometry, alternative)[0]
                assert np.array_equal(observed, expected)
                histogram[sum(x == 0 for x in expected)] += weight
                configurations += 1
        coefficients, denominator = zero_count_coefficients(n, len(pairs), q, theta, n)
        assert histogram == [Fraction(x, denominator) for x in coefficients]
        assert sum(histogram) == 1
        laws_checked.append({"rows": rows, "columns": columns, "q": q,
                             "lag": lag, "theta": "1/4", "exact_law_match": True})
        # Boundary activation probabilities preserve the intended fields.
        marker = np.arange(n, dtype=np.int32)[None, :]
        for probability, mask in [(0.0, [False]*len(pairs)),
                                  (1.0, [True]*len(pairs))]:
            expected = marker.copy()
            if probability:
                for donor, recipient in pairs:
                    expected[0, recipient] = marker[0, donor]
            got = apply_alternative(marker.copy(), FixedMask(mask), geometry,
                                    {**alternative, "theta": probability})
            assert np.array_equal(got, expected)
    return {"geometry_checks": geometry_count,
            "exhaustive_field_mask_configurations": configurations,
            "small_exact_laws": laws_checked,
            "zero_and_unit_activation_checks": 8}


def independently_replay(group, geometry, batch_size):
    """Reconstruct NPCR/UACI without the production generation/scoring helpers."""
    rng = np.random.default_rng(np.random.SeedSequence(group["field_seed_entropy"]))
    rows, columns, q = (geometry[k] for k in ("rows", "columns", "q"))
    n, count = rows*columns, group["sample_size"]
    alternative = group.get("alternative")
    result = np.empty((count, 2), dtype=np.int64)
    for start in range(0, count, batch_size):
        size = min(batch_size, count-start)
        first = rng.integers(q, size=(size, n), dtype=np.int32)
        second = rng.integers(q, size=(size, n), dtype=np.int32)
        field = np.abs(first-second)
        if alternative:
            if alternative["kind"] == "copying":
                pairs = independent_pairs(rows, columns, alternative["lag"])
                active = rng.random((size, len(pairs))) < alternative["theta"]
                for k, (donor, recipient) in enumerate(pairs):
                    enabled = active[:, k]
                    field[enabled, recipient] = field[enabled, donor]
            elif alternative["kind"] == "partial_identity":
                field[rng.random(field.shape) < alternative["epsilon"]] = 0
            else:
                raise ValueError("unexpected alternative")
        result[start:start+size, 0] = n-np.sum(field == 0, axis=1)
        result[start:start+size, 1] = np.sum(field, axis=1, dtype=np.int64)
    return result


def run(results_dir):
    results_dir = Path(results_dir)
    power_path, protocol_path = results_dir/"power.json", results_dir/"protocol.json"
    saved = json.loads(power_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    config = protocol["power"]
    assert saved["protocol"] == config
    assert len(config["geometries"]) == 1
    geometry = config["geometries"][0]
    rows, columns, q = (geometry[k] for k in ("rows", "columns", "q"))
    n = rows*columns
    mean_count = Fraction(n*(q-1), q)
    assert mean_count.denominator == 1
    source_paths = [Path(__file__).with_name(name)
                    for name in saved["metadata"]["source_sha256"]]
    input_paths = [power_path, protocol_path] + source_paths
    input_paths += [results_dir/g["raw_file"] for g in saved["groups"]]
    before = {str(p.relative_to(ROOT)): digest(p) for p in input_paths}
    for name, expected in saved["metadata"]["source_sha256"].items():
        assert matches_recorded_digest(Path(__file__).with_name(name), expected)
    assert matches_recorded_digest(protocol_path, saved["metadata"]["protocol_sha256"])
    entropies = [tuple(g[k]) for g in saved["groups"]
                 for k in ("field_seed_entropy", "tie_seed_entropy")]
    assert len(entropies) == len(set(entropies))
    report = {"status": "running", "numpy_version": np.__version__,
              "original_numpy_version": saved["metadata"]["numpy"],
              "method": "Exact rational zero-count PGF evaluated at the saved NPCR rank boundary and tie cutoff",
              "scope": "Existing observations are retained; no additional trial sample is generated",
              "source_and_protocol_hashes_match": True,
              "distinct_field_and_tie_seed_entropies": len(entropies),
              "small_checks": exact_small_checks(), "groups": [],
              "bank_conditional_probabilities": [], "input_sha256": before}
    thresholds = [t for t in saved["thresholds"] if t["statistic"] == "NPCR_count"]
    for group in saved["groups"]:
        path = results_dir/group["raw_file"]
        assert digest(path) == group["raw_file_sha256"]
        with np.load(path) as data:
            raw = data["statistics"]
            assert list(data["statistic_names"]) == config["statistics"]
        assert hashlib.sha256(raw.astype("<i8", copy=False).tobytes()).hexdigest() == group["raw_statistics_sha256"]
        replay = independently_replay(group, geometry, config["batch_size"])
        assert np.array_equal(replay, raw[:, :2]), group["model_id"]
        ties = np.random.default_rng(np.random.SeedSequence(group["tie_seed_entropy"])).random(raw.shape)[:, 0]
        discrepancies = np.abs(raw[:, 0]-int(mean_count))
        if group["model_id"] == "calibration":
            order = np.lexsort((ties, discrepancies))
            for threshold in thresholds:
                index = order[len(raw)-threshold["k"]]
                assert int(discrepancies[index]) == threshold["value"]
                assert float(ties[index]) == threshold["tie"]
        else:
            for threshold in thresholds:
                rejected = (discrepancies > threshold["value"]) | (
                    (discrepancies == threshold["value"]) & (ties > threshold["tie"]))
                count = int(np.count_nonzero(rejected))
                original = next(r for r in saved["rates"]
                                if r["model_id"] == group["model_id"]
                                and r["statistic"] == "NPCR_count"
                                and r["alpha"] == threshold["alpha"])
                assert count == original["rejections"]
                alternative = group.get("alternative")
                if alternative is None or alternative["kind"] == "copying":
                    lag = alternative["lag"] if alternative else 1
                    theta = Fraction(str(alternative["theta"])) if alternative else Fraction(0)
                    pair_count = len(independent_pairs(rows, columns, lag))
                    probability = exact_rejection_probability(n, pair_count, q, theta, threshold)
                    p, trials = float(probability), len(raw)
                    se = sqrt(p*(1-p)/trials)
                    exact_variance = Fraction(q-1, q*q)*(n+2*pair_count*theta)
                    report["bank_conditional_probabilities"].append({
                        "model_id": group["model_id"], "alpha": threshold["alpha"],
                        "paired_blocks": pair_count, "untouched_pixels": n-2*pair_count,
                        "theta": str(theta), "threshold": threshold["value"],
                        "tie_cutoff": threshold["tie"],
                        "probability_decimal": decimal_string(probability),
                        "probability": p, "observed_rejections": count,
                        "trials": trials, "observed_rate": count/trials,
                        "difference_in_model_standard_errors": (count/trials-p)/se,
                        "pointwise_binomial_two_sided_p": float(binomtest(count, trials, p).pvalue),
                        "exact_count_mean": str(mean_count),
                        "exact_count_variance": str(exact_variance),
                        "observed_count_mean": float(np.mean(raw[:, 0])),
                        "observed_count_variance": float(np.var(raw[:, 0], ddof=1))})
        report["groups"].append({"model_id": group["model_id"],
                                 "observations": len(raw),
                                 "npcr_and_uaci_seed_replay": "exact match",
                                 "npcr_rank_replay": "exact match"})
        print("Verified", group["model_id"], flush=True)
    assert before == {str(p.relative_to(ROOT)): digest(p) for p in input_paths}
    report.update(status="passed", original_inputs_unchanged=True,
                  conclusion="All exact small-law, geometry, seed-replay and saved-rank checks pass. The recorded rejection discrepancies remain sampling deviations under the verified model and decision rule; no generator or transcription error was detected.",
                  inference_note="Binomial diagnostics are pointwise conditional on the saved bank. They are not a simultaneous family test and were selected after inspecting the original results.")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT/"results")
    parser.add_argument("--output", type=Path, default=ROOT/"results/npcr_copying_diagnostic.json")
    args = parser.parse_args()
    report = run(args.results_dir)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print("All checks passed; original observations retained.")


if __name__ == "__main__":
    main()
