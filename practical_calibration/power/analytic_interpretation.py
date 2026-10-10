"""Analytic interpretation of a saved paired image-statistic power experiment.

No random draws are made. Exact rational moments are evaluated under partial
identity, saved integer-score calibration thresholds are checked, and exact
binomial NPCR probabilities are compared with continuous Gaussian BACI/UACI
approximations and the existing conditional Monte Carlo estimates.
"""
from __future__ import annotations

import argparse
import csv
from fractions import Fraction as F
import hashlib
from itertools import combinations, product
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import binom, norm

NAMES = ("BACI", "UACI", "NPCR")


def law_moments(q, strength):
    """Exact rational one-pixel and pair-distance moments for partial identity."""
    lam = F(str(strength))
    if not (q >= 2 and 0 <= lam <= 1):
        raise ValueError("Require q >= 2 and 0 <= strength <= 1")
    p0 = [F(1, q)] + [F(2*(q-x), q*q) for x in range(1, q)]
    p = [(1-lam)*px for px in p0]
    p[0] += lam
    first = sum((x*px for x, px in enumerate(p)), F())
    second = sum((x*x*px for x, px in enumerate(p)), F())
    # Independent evaluation of the closed mixture moments.
    assert first == (1-lam)*F(q*q-1, 3*q)
    assert second == (1-lam)*F(q*q-1, 6)
    cumulative = partial_first = F()
    g = []
    for x, px in enumerate(p):
        cumulative += px
        partial_first += x*px
        g.append(x*(2*cumulative-1)+first-2*partial_first)
    mu = sum((px*gx for px, gx in zip(p, g)), F())
    null_mu = F(4*(q**4-1), 15*q**3)
    assert mu == (1-lam)**2*null_mu + 2*lam*(1-lam)*F(q*q-1, 3*q)
    A = 2*(second-first*first)-mu*mu
    C = sum((px*(gx-mu)**2 for px, gx in zip(p, g)), F())
    assert A >= 0 and C >= 0 and sum(p) == 1
    return p, {"difference_mean": first, "difference_second_moment": second,
               "pair_mean": mu, "pair_variance": A, "shared_endpoint_covariance": C}


def statistic_moments(q, rows, columns, strength):
    p, c = law_moments(q, strength)
    n, K = rows*columns, (rows-1)*(columns-1)
    gamma2 = 10*n-12*rows-12*columns+14
    W = 62*n-96*rows-96*columns+148
    raw = {
        "BACI": (6*K*c["pair_mean"],
                 gamma2*c["pair_variance"]+2*W*c["shared_endpoint_covariance"]),
        "UACI": (n*c["difference_mean"],
                 n*(c["difference_second_moment"]-c["difference_mean"]**2)),
        "NPCR": (n*(1-p[0]), n*p[0]*(1-p[0])),
    }
    scale = {"BACI": 6*K*(q-1), "UACI": n*(q-1), "NPCR": n}
    normalized = {name: (mean/scale[name], var/scale[name]**2)
                  for name, (mean, var) in raw.items()}
    return c, raw, normalized


def score_parameters(cfg):
    q, rows, columns = cfg["q"], cfg["rows"], cfg["columns"]
    n, K = rows*columns, (rows-1)*(columns-1)
    return [(15*q**3, 4*(q**4-1)*6*K),
            (3*q, n*(q*q-1)), (q, n*(q-1))]


def gaussian_tail(mean, variance, center, halfwidth):
    if variance <= 0:
        raise ValueError("Gaussian comparison requires positive variance")
    sd = math.sqrt(variance)
    return float(norm.cdf((center-halfwidth-mean)/sd) +
                 norm.sf((center+halfwidth-mean)/sd))


def exact_npcr(q, pixels, strength, threshold):
    success = (1-F(str(strength)))*F(q-1, q)
    x = np.arange(pixels+1, dtype=np.int64)
    score = np.abs(q*x-pixels*(q-1))
    pmf = binom.pmf(x, pixels, float(success))
    strict = float(pmf[score > threshold["score"]].sum())
    boundary = float(pmf[score == threshold["score"]].sum())
    power = strict+(1-threshold["tie_uniform"])*boundary
    # Independent cumulative-tail evaluation on contiguous rejected runs.
    selected = np.flatnonzero(score > threshold["score"])
    runs = np.split(selected, np.flatnonzero(np.diff(selected) > 1)+1)
    cumulative = 0.0
    for run in runs:
        if not len(run):
            continue
        lo, hi = int(run[0]), int(run[-1])
        cumulative += (binom.cdf(hi, pixels, float(success)) if lo == 0 else
                       binom.sf(lo-1, pixels, float(success)) if hi == pixels else
                       binom.cdf(hi, pixels, float(success))-
                       binom.cdf(lo-1, pixels, float(success)))
    independent = float(cumulative)+(1-threshold["tie_uniform"])*boundary
    # Compare unmodified sums before correcting floating-point roundoff.
    assert abs(strict-float(cumulative)) < 5e-13
    assert abs(power-independent) < 5e-13
    for probability in (power, strict, boundary):
        if not math.isfinite(probability) or not -5e-13 <= probability <= 1+5e-13:
            raise ArithmeticError("Binomial probability outside roundoff tolerance")
    return tuple(min(1., max(0., probability)) for probability in (power, strict, boundary))


def literal_statistics(values, rows, columns):
    baci = 0
    for row in range(rows-1):
        for column in range(columns-1):
            vertices = (row*columns+column, row*columns+column+1,
                        (row+1)*columns+column, (row+1)*columns+column+1)
            baci += sum(abs(values[a]-values[b]) for a, b in combinations(vertices, 2))
    return baci, sum(values), sum(x != 0 for x in values)


def check_small_exact():
    """Enumerate finite overlapping grids without simulation to verify moments."""
    verified = []
    for q, rows, columns in ((2, 2, 2), (2, 2, 3), (2, 3, 3), (3, 2, 2), (3, 2, 3)):
        for strength in (0, F(1, 5), F(3, 5)):
            p, _ = law_moments(q, strength)
            _, raw, _ = statistic_moments(q, rows, columns, strength)
            first = [F() for _ in NAMES]
            second = [F() for _ in NAMES]
            for values in product(range(q), repeat=rows*columns):
                mass = math.prod(p[x] for x in values)
                statistics = literal_statistics(values, rows, columns)
                for j, value in enumerate(statistics):
                    first[j] += mass*value
                    second[j] += mass*value*value
            for j, name in enumerate(NAMES):
                assert (first[j], second[j]-first[j]**2) == raw[name]
            verified.append({"alphabet_size": q, "rows": rows, "columns": columns,
                             "strength": str(strength), "states_enumerated": q**(rows*columns),
                             "statistics_verified": list(NAMES)})
    return verified


def check_null_and_copy_counts():
    for q in (2, 3, 4, 16, 256):
        _, constants = law_moments(q, 0)
        assert constants["pair_mean"] == F(4*(q**4-1), 15*q**3)
        assert constants["pair_variance"] == F((q*q-1)*(9*q**6+34*q**4+16*q*q+16), 225*q**6)
        assert constants["shared_endpoint_covariance"] == F((q*q-4)*(q*q-1)*(3*q**4+20*q*q-8), 450*q**6)
    for rows, columns in ((2, 2), (2, 4), (3, 4), (7, 6), (64, 64)):
        copied_edges = 0
        occurrences = 0
        for row in range(rows-1):
            for column in range(columns-1):
                vertices = ((row, column), (row, column+1), (row+1, column), (row+1, column+1))
                for (i, j), (k, ell) in combinations(vertices, 2):
                    occurrences += 1
                    copied_edges += (i == k and j//2 == ell//2)
        assert copied_edges == columns*(rows-1)
        assert F(copied_edges, occurrences) == F(columns, 6*(columns-1))


def write_csv(path, records):
    with path.open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--power-dir", type=Path)
    inputs.add_argument("--archive-root", type=Path)
    parser.add_argument("--output", "--output-dir", dest="output", type=Path, required=True)
    args = parser.parse_args()
    src = (args.power_dir if args.power_dir is not None else
           args.archive_root/"fixed_image_and_power/power").resolve()
    dest = args.output.resolve()
    protected = args.archive_root.resolve() if args.archive_root is not None else src
    if dest.is_relative_to(protected):
        raise ValueError("Choose an output directory outside the supplied input tree")
    if dest.exists():
        raise FileExistsError("Output directory must be absent")
    dest.mkdir(parents=True)
    cfg = json.loads((src/"design.json").read_text())
    thresholds = json.loads((src/"results/thresholds.json").read_text())
    parameters = score_parameters(cfg)
    q, rows, columns = cfg["q"], cfg["rows"], cfg["columns"]
    pixels = rows*columns
    assert columns % 2 == 0
    with np.load(src/"results/raw/calibration.npz", allow_pickle=False) as bank:
        raw, ties = bank["statistics"], bank["tie_uniforms"]
        assert raw.shape == ties.shape == (cfg["calibration_size"], 3)
        for alpha in cfg["alpha"]:
            k = round(alpha*(len(raw)+1))
            assert k == alpha*(len(raw)+1)
            for j, (factor, offset) in enumerate(parameters):
                score = np.abs(factor*raw[:, j]-offset)
                order = np.lexsort((ties[:, j], score))
                index = order[len(raw)-k]
                assert thresholds[str(alpha)][j] == {
                    "statistic": NAMES[j], "score": int(score[index]),
                    "tie_uniform": float(ties[index, j]), "rank_from_largest": k}
    with (src/"results/power_summary.csv").open() as f:
        summaries = list(csv.DictReader(f))
    lookup = {(r["family"], float(r["strength"]), float(r["alpha"]), r["statistic"]): r
              for r in summaries}
    # Verify the saved conditional estimates against every reused identity sample.
    strengths = [0.] + cfg["identity_probabilities"]
    for index, strength in enumerate(strengths):
        name = "null_validation" if index == 0 else f"identity_{index:02d}"
        with np.load(src/f"results/raw/{name}.npz", allow_pickle=False) as data:
            raw, ties = data["statistics"], data["tie_uniforms"]
            for alpha in cfg["alpha"]:
                for j, (factor, offset) in enumerate(parameters):
                    threshold = thresholds[str(alpha)][j]
                    score = np.abs(factor*raw[:, j]-offset)
                    rejected = (score > threshold["score"]) | ((score == threshold["score"]) &
                                (ties[:, j] > threshold["tie_uniform"]))
                    observed = lookup[("null" if strength == 0 else "identity", strength, alpha, NAMES[j])]
                    assert int(rejected.sum()) == int(observed["rejections"])
                    assert len(raw) == int(observed["replications"])

    checks = check_small_exact()
    check_null_and_copy_counts()
    _, null_raw, null_normalized = statistic_moments(q, rows, columns, 0)
    power_rows, moment_rows = [], []
    for strength in strengths:
        constants, raw_moments, normalized = statistic_moments(q, rows, columns, strength)
        moment_rows.append({"strength": strength,
                           "exact_constants": {k: str(v) for k, v in constants.items()},
                           "normalized_moments": {name: {"mean_exact": str(values[0]),
                                  "variance_exact": str(values[1]), "mean": float(values[0]),
                                  "variance": float(values[1])} for name, values in normalized.items()}})
        for alpha in cfg["alpha"]:
            for j, name in enumerate(NAMES):
                threshold = thresholds[str(alpha)][j]
                factor, offset = parameters[j]
                observed = lookup[("null" if strength == 0 else "identity", strength, alpha, name)]
                center, width = offset/factor, threshold["score"]/factor
                assert F(offset, factor) == null_raw[name][0]
                if name == "NPCR":
                    prediction, strict, boundary = exact_npcr(q, pixels, strength, threshold)
                    method = "exact binomial with randomized boundary"
                else:
                    prediction = gaussian_tail(float(raw_moments[name][0]), float(raw_moments[name][1]),
                                               center, width)
                    # A continuous Gaussian surrogate has zero boundary mass.
                    strict, boundary = prediction, 0.
                    method = "moment Gaussian at saved numerical threshold"
                power_rows.append({"family": "identity", "strength": strength, "alpha": alpha,
                    "statistic": name, "method": method, "prediction": prediction,
                    "strict_tail_probability": strict, "boundary_probability": boundary,
                    "boundary_rejection_probability": 1-threshold["tie_uniform"],
                    "integer_score_threshold": threshold["score"], "raw_halfwidth": width,
                    "observed_probability": float(observed["probability"]),
                    "observed_ci_low": float(observed["ci_low"]), "observed_ci_high": float(observed["ci_high"]),
                    "observed_replications": int(observed["replications"])})
    horizontal = []
    for theta in [0.] + cfg["horizontal_copy_probabilities"]:
        baci_shift = float(null_normalized["BACI"][0])*columns*theta/(6*(columns-1))
        standardized = baci_shift/math.sqrt(float(null_normalized["BACI"][1]))
        for alpha in cfg["alpha"]:
            probability = float(2*norm.sf(norm.isf(alpha/2)/math.sqrt(1+theta)))
            horizontal.append({"theta": theta, "alpha": alpha,
                "uaci_npcr_ideal_threshold_gaussian_power": probability,
                "baci_mean_decrease": baci_shift, "baci_standardized_mean_decrease": standardized,
                "mean_shift_scaling_coefficient": standardized/(theta*math.sqrt(pixels)) if theta else
                    float(null_normalized["BACI"][0])*columns/(6*(columns-1))/
                    math.sqrt(float(null_normalized["BACI"][1])*pixels),
                "benchmark": "ideal null Gaussian threshold; not saved-bank rank power"})
    write_csv(dest/"partial_identity_power.csv", power_rows)
    write_csv(dest/"horizontal_copy_gaussian_benchmark.csv", horizontal)
    (dest/"partial_identity_moments.json").write_text(json.dumps(moment_rows, indent=2)+"\n")
    source_paths = ["design.json", "results/thresholds.json", "results/power_summary.csv",
                    "results/raw/calibration.npz", "results/raw/null_validation.npz"]
    source_paths += [f"results/raw/identity_{i:02d}.npz" for i in range(1, len(strengths))]
    verification = {
        "scope": "Deterministic analytic evaluation and reuse of saved samples only; no new random draws",
        "source_hashes": {name: hashlib.sha256((src/name).read_bytes()).hexdigest() for name in source_paths},
        "calibration_thresholds_reproduced": 3*len(cfg["alpha"]),
        "conditional_estimate_counts_reproduced": len(strengths)*3*len(cfg["alpha"]),
        "exact_small_grid_enumerations": checks,
        "null_pair_constants": "Exact finite sums match closed formulas for q=2,3,4,16,256",
        "horizontal_copy_count": "Literal window-pair enumeration matches the finite-boundary coefficient on five grids, including 64 by 64",
        "exact_npcr_crosscheck": "Summed binomial masses agree with independent cumulative-tail evaluation within 5e-13",
        "gaussian_boundary_convention": "Saved numerical threshold used; continuous surrogate has zero point mass, so the saved tie probability does not affect that approximation",
        "horizontal_copy_scope": "Ideal Gaussian variance-only benchmark for UACI/NPCR and exact BACI mean shift; no BACI alternative variance or full power curve is claimed",
        "gaussian_predictions_outside_pointwise_wilson_intervals": [
            {key: r[key] for key in ("strength", "alpha", "statistic", "prediction", "observed_probability", "observed_ci_low", "observed_ci_high")}
            for r in power_rows if r["statistic"] != "NPCR" and not r["observed_ci_low"] <= r["prediction"] <= r["observed_ci_high"]],
        "units": "Fractions, not percentages",
    }
    (dest/"verification.json").write_text(json.dumps(verification, indent=2)+"\n")
    selected = [r for r in power_rows if r["strength"] == .005 and r["alpha"] == .05]
    print(json.dumps({"selected_power": selected, "checks_passed": True}, indent=2))


if __name__ == "__main__":
    main()
