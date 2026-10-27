"""Exact finite-support diagnostics for the published fixed Gaussian rule.

From the companion root, run
    python -B statistical_validation/analysis/exact_calibration.py --verify-only
    python -B statistical_validation/analysis/exact_calibration.py

Requires Python 3.10+ and a C++17 compiler (CXX, or c++). No Python packages are
required. The C++ executable is temporary and is never placed in the package.
Existing simulation files and protocols are read only. Outputs are
statistical_validation/results/exact_calibration.json and exact_calibration.csv.

All probabilities, moments, omitted mass and decisions use rational/integer
arithmetic. Display intervals are rounded outward. Normal-quantile brackets
are certified by rational enclosures of the Gaussian integral, using its
alternating Taylor series and a Machin-formula enclosure of pi. The sparse
enumeration retains original Bernoulli probabilities without renormalization.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from fractions import Fraction as F
import hashlib
import io
from itertools import combinations, product
import json
from math import comb, isqrt
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ALPHAS = (F(1, 20), F(1, 100), F(1, 1000))
# These candidates are verified below against rigorous CDF enclosures.
QUANTILE_CANDIDATES = {
    F(1, 20): ("1.959963984540", "1.959963984541"),
    F(1, 100): ("2.575829303548", "2.575829303549"),
    F(1, 1000): ("3.290526731491", "3.290526731492"),
}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_safe(value):
    if isinstance(value, F):
        return {"numerator": str(value.numerator), "denominator": str(value.denominator)}
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    return value


def outward(value, digits, upper=False):
    """Exact fixed-point decimal floor or ceiling, including negative values."""
    scaled = value * 10**digits
    integer = (-((-scaled.numerator)//scaled.denominator) if upper
               else scaled.numerator//scaled.denominator)
    sign = "-" if integer < 0 else ""
    integer = abs(integer)
    return f"{sign}{integer//10**digits}.{integer%10**digits:0{digits}d}"


def interval(low, high, digits=15):
    require(low <= high, "reversed interval")
    return {"lower": low, "upper": high,
            "decimal_lower": outward(low, digits),
            "decimal_upper": outward(high, digits, True),
            "percent_lower": outward(100*low, 9),
            "percent_upper": outward(100*high, 9, True)}


def alternating_atan(x, terms=100):
    partial = sum(((-1)**k*x**(2*k+1)/F(2*k+1) for k in range(terms)), F(0))
    next_sum = partial + (-1)**terms*x**(2*terms+1)/F(2*terms+1)
    return min(partial, next_sum), max(partial, next_sum)


def pi_bounds():
    a, b = alternating_atan(F(1, 5))
    c, d = alternating_atan(F(1, 239))
    # Machin's identity pi = 16 atan(1/5) - 4 atan(1/239).
    return 16*a-4*d, 16*b-4*c


def sqrt_bounds(x, digits=80):
    scale = 10**digits
    a = isqrt(x.numerator*scale*scale//x.denominator)
    low, high = F(a, scale), F(a+1, scale)
    require(low*low <= x <= high*high, "square-root enclosure failed")
    return low, high


def normal_cdf_bounds(x):
    """Enclose Phi(x) for rational 0<=x<=4 using rational arithmetic only."""
    require(0 <= x <= 4, "CDF proof is limited to [0,4]")
    terms = 120
    term, partial = x, F(0)
    for k in range(terms):
        partial += (-1)**k * term
        term *= x*x*F(2*k+1, 2*(k+1)*(2*k+3))
    adjacent = partial + (-1)**terms*term
    # The unsigned term ratio is at most 8/(k+1), hence decreases below 1
    # throughout the omitted tail. Adjacent partial sums bracket the integral.
    require(F(8, terms+1) < 1, "alternating-tail condition failed")
    integral_low, integral_high = min(partial, adjacent), max(partial, adjacent)
    pi_low, pi_high = pi_bounds()
    root_low, _ = sqrt_bounds(2*pi_low)
    _, root_high = sqrt_bounds(2*pi_high)
    return F(1, 2)+integral_low/root_high, F(1, 2)+integral_high/root_low


def certify_quantiles():
    results = {}
    for alpha, bounds in QUANTILE_CANDIDATES.items():
        low, high = map(F, bounds)
        target = 1-alpha/2
        cdf_low, cdf_high = normal_cdf_bounds(low), normal_cdf_bounds(high)
        require(cdf_low[1] < target < cdf_high[0], "quantile bracket not certified")
        results[alpha] = {"lower": low, "upper": high, "target_cdf": target,
                          "cdf_at_lower": interval(*cdf_low, digits=30),
                          "cdf_at_upper": interval(*cdf_high, digits=30)}
    return results


def graph(rows, columns):
    """Independent Python construction from geometric membership of pairs."""
    weights = {}
    n = rows*columns
    for i, j in combinations(range(n), 2):
        ri, ci = divmod(i, columns)
        rj, cj = divmod(j, columns)
        count_r = max(0, min(ri, rj, rows-2)-max(ri-1, rj-1, 0)+1)
        count_c = max(0, min(ci, cj, columns-2)-max(ci-1, cj-1, 0)+1)
        if count_r*count_c:
            weights[i, j] = count_r*count_c
    degrees = [0]*n
    for (i, j), weight in weights.items():
        degrees[i] += weight
        degrees[j] += weight
    require(sum(weights.values()) == 6*(rows-1)*(columns-1), "window pair total")
    for i, degree in enumerate(degrees):
        r, c = divmod(i, columns)
        expected = 3 * (1 if r in (0, rows-1) else 2) * (1 if c in (0, columns-1) else 2)
        require(degree == expected, "boundary/interior weighted degree")
    return weights, degrees


def literal_window_histogram(rows, columns):
    """Test oracle sums discordant pairs inside each window, with no graph."""
    hist = Counter()
    for values in product((0, 1), repeat=rows*columns):
        score = 0
        for r in range(rows-1):
            for c in range(columns-1):
                pixels = [values[r*columns+c], values[r*columns+c+1],
                          values[(r+1)*columns+c], values[(r+1)*columns+c+1]]
                score += sum(abs(a-b) for a, b in combinations(pixels, 2))
        hist[sum(values), score] += 1
    return hist


def run_kernel(executable, rows, columns, mode, maximum_k):
    result = subprocess.run([str(executable), str(rows), str(columns), mode,
                             str(maximum_k)], check=True, capture_output=True, text=True)
    hist = {(int(row["occupied"]), int(row["cut"])): int(row["count"])
            for row in csv.DictReader(io.StringIO(result.stdout))}
    for k in range(maximum_k+1):
        require(sum(count for (occupied, cut), count in hist.items() if occupied == k)
                == comb(rows*columns, k), f"cardinality total {k}")
    return hist


def exact_moments(rows, columns, p, amplitude):
    weights, degrees = graph(rows, columns)
    gamma1 = sum(weights.values())
    gamma2 = sum(w*w for w in weights.values())
    degree2 = sum(d*d for d in degrees)
    a = 2*p*(1-p)
    covariance = p*(1-p)*(1-2*p)**2
    # Independently compute the shared-edge product from eight assignments.
    product_expectation = sum((p**sum(v)*(1-p)**(3-sum(v))*
                              abs(v[0]-v[1])*abs(v[0]-v[2])
                              for v in product((0, 1), repeat=3)), F(0))
    require(product_expectation-a*a == covariance, "Bernoulli shared-edge covariance")
    mean = amplitude*a*gamma1
    variance = amplitude**2*((a*(1-a)-2*covariance)*gamma2+covariance*degree2)
    # Direct shared-edge weight products supply a separate geometry check.
    incident = [[] for _ in degrees]
    for (i, j), weight in weights.items():
        incident[i].append(weight)
        incident[j].append(weight)
    shared = sum(sum(x*y for x, y in combinations(row, 2)) for row in incident)
    require(2*shared == degree2-2*gamma2, "shared-edge identity")
    require(variance == amplitude**2*(a*(1-a)*gamma2+2*covariance*shared),
            "independent variance assembly")
    return {"raw_mean": mean, "raw_variance": variance,
            "cut_mean": mean/amplitude, "cut_variance": variance/amplitude**2,
            "gamma1": gamma1, "gamma2": gamma2, "weighted_degree_square_sum": degree2,
            "shared_edge_weight_products": shared,
            "degree_counts": dict(sorted(Counter(degrees).items())),
            "distinct_edges": len(weights), "maximum_cut_bound": gamma1}


def rejection_sets(moments, quantile, amplitude):
    mu, var = moments["raw_mean"], moments["raw_variance"]
    low_z, high_z = quantile["lower"], quantile["upper"]
    low, high, accepted = [], [], []
    minimum_margin = None
    for cut in range(moments["maximum_cut_bound"]+1):
        difference = amplitude*cut-mu
        squared = difference*difference
        if squared > var*high_z**2:
            (low if difference < 0 else high).append(cut)
            margin = squared-var*high_z**2
        elif squared <= var*low_z**2:
            accepted.append(cut)
            margin = var*low_z**2-squared
        else:
            raise AssertionError("integer score lies inside quantile uncertainty")
        minimum_margin = margin if minimum_margin is None else min(minimum_margin, margin)
    require(minimum_margin > 0, "decision is on a boundary")
    return set(low), set(high), {"lower_tail_maximum_cut": max(low) if low else None,
        "upper_tail_minimum_cut": min(high) if high else None,
        "accepted_cut_minimum": min(accepted), "accepted_cut_maximum": max(accepted),
        "minimum_squared_score_margin": minimum_margin,
        "score_support_bound_used": [0, moments["maximum_cut_bound"]]}


def histogram_rows(hist):
    return [{"occupied": k, "cut": cut, "count": count}
            for (k, cut), count in sorted(hist.items())]


def verify_cardinality_moments(hist, n, moments):
    """Check each layer against hypergeometric moments, independent of cuts.

    For a uniformly selected k-set, edge disagreement has probability
    2*k*(n-k)/(n*(n-1)). Two distinct edges sharing one endpoint disagree
    together with half that probability. For disjoint edges the probability
    is 4*(k)_2*(n-k)_2/(n)_4. These are finite-population identities and use
    neither Bernoulli weighting nor the enumeration recurrence.
    """
    gamma1, gamma2 = moments["gamma1"], moments["gamma2"]
    shared = moments["shared_edge_weight_products"]
    disjoint = F(gamma1*gamma1-gamma2, 2)-shared
    checks = []
    for k in sorted({k for k, cut in hist}):
        count = comb(n, k)
        edge = F(2*k*(n-k), n*(n-1))
        both_shared = edge/2
        both_disjoint = F(4*k*(k-1)*(n-k)*(n-k-1), n*(n-1)*(n-2)*(n-3))
        expected_first = count*gamma1*edge
        expected_second = count*(gamma2*edge+2*shared*both_shared+2*disjoint*both_disjoint)
        observed_first = sum(value*cut for (j,cut),value in hist.items() if j==k)
        observed_second = sum(value*cut*cut for (j,cut),value in hist.items() if j==k)
        require(observed_first==expected_first and observed_second==expected_second,
                f"hypergeometric histogram moments for cardinality {k}")
        checks.append({"occupied": k, "support_count": count,
                       "cut_sum": observed_first, "cut_square_sum": observed_second})
    return checks


def summarize(case, rows, columns, p, amplitude, hist, quantiles, complete):
    n = rows*columns
    moments = exact_moments(rows, columns, p, amplitude)
    layer_moment_checks = verify_cardinality_moments(hist, n, moments)
    max_k = max(k for k, cut in hist)
    retained_mass = sum((F(count)*p**k*(1-p)**(n-k)
                         for (k, cut), count in hist.items()), F(0))
    omitted_mass = 1-retained_mass
    require(omitted_mass == 1-sum((comb(n, k)*p**k*(1-p)**(n-k)
                                  for k in range(max_k+1)), F(0)), "binomial retained mass")
    if complete:
        require(omitted_mass == 0, "complete law has missing mass")
        mean = sum((count*p**k*(1-p)**(n-k)*amplitude*cut
                    for (k, cut), count in hist.items()), F(0))
        second = sum((count*p**k*(1-p)**(n-k)*(amplitude*cut)**2
                      for (k, cut), count in hist.items()), F(0))
        require(mean == moments["raw_mean"] and second-mean*mean == moments["raw_variance"],
                "histogram raw moments")
    result = {"case_id": case, "rows": rows, "columns": columns,
              "occupied_probability": p, "occupied_value": amplitude,
              "window_height": 2, "window_width": 2,
              "enumeration_complete": complete, "maximum_occupied_enumerated": max_k,
              "enumerated_support_count": sum(hist.values()),
              "enumerated_cut_minimum": min(cut for k, cut in hist),
              "enumerated_cut_maximum": max(cut for k, cut in hist),
              "exact_cardinality_moment_checks": layer_moment_checks,
              "support_count_by_cardinality": {str(k): sum(v for (j,c),v in hist.items() if j==k)
                                               for k in range(max_k+1)},
              "moments": moments, "retained_probability_mass": retained_mass,
              "omitted_probability_mass": interval(omitted_mass, omitted_mass),
              "histogram": histogram_rows(hist), "rejections": []}
    for alpha in ALPHAS:
        low, high, thresholds = rejection_sets(moments, quantiles[alpha], amplitude)
        lower = sum((count*p**k*(1-p)**(n-k) for (k, cut), count in hist.items()
                     if cut in low), F(0))
        upper = sum((count*p**k*(1-p)**(n-k) for (k, cut), count in hist.items()
                     if cut in high), F(0))
        lower_max = lower+(omitted_mass if low else 0)
        upper_max = upper+(omitted_mass if high else 0)
        total_max = lower+upper+(omitted_mass if low or high else 0)
        result["rejections"].append({"alpha": alpha, "thresholds": thresholds,
            "retained_lower_tail_count": sum(v for (k, cut),v in hist.items() if cut in low),
            "retained_upper_tail_count": sum(v for (k, cut),v in hist.items() if cut in high),
            "retained_rejection_counts_by_cardinality": [
                {"occupied": k,
                 "lower_tail_count": sum(v for (j,c),v in hist.items() if j==k and c in low),
                 "upper_tail_count": sum(v for (j,c),v in hist.items() if j==k and c in high)}
                for k in range(max_k+1)],
            "lower_tail": interval(lower, lower_max), "upper_tail": interval(upper, upper_max),
            "two_sided": interval(lower+upper, total_max)})
    return result


def verify_small(executable):
    checks = []
    for rows, columns in ((2,2), (2,3), (3,3)):
        direct = literal_window_histogram(rows, columns)
        full = run_kernel(executable, rows, columns, "full", rows*columns)
        sparse = run_kernel(executable, rows, columns, "sparse", 4)
        require(full == direct, "C++ full versus literal window oracle")
        require(sparse == {key: value for key, value in direct.items() if key[0] <= 4},
                "C++ sparse versus literal window oracle")
        for p in (F(1,2), F(1,1024), F(1,3)):
            moments = exact_moments(rows, columns, p, 7)
            mu = sum((v*p**k*(1-p)**(rows*columns-k)*7*c
                      for (k,c),v in direct.items()), F(0))
            second = sum((v*p**k*(1-p)**(rows*columns-k)*(7*c)**2
                          for (k,c),v in direct.items()), F(0))
            require(mu == moments["raw_mean"] and second-mu*mu == moments["raw_variance"],
                    "small-grid exact Bernoulli moments")
        checks.append(f"{rows}x{columns} full and sparse histograms match literal-window enumeration; moments match for three laws")
    require(outward(F(1,3),3)=="0.333" and outward(F(1,3),3,True)=="0.334"
            and outward(F(-1,3),3)=="-0.334" and outward(F(-1,3),3,True)=="-0.333",
            "directed decimal rounding")
    return checks


def check_existing_moments(cases):
    """Optional read-only cross-check after independent moments are computed."""
    path = STUDY/"results/calibration.json"
    if not path.exists():
        return {"status": "not available"}
    data = json.loads(path.read_text())
    original = {row["case"]["id"]: row for row in data["cases"]}
    for case in cases:
        row = original[case["case_id"]]
        for key, source_key in (("raw_mean", "exact_mean"), ("raw_variance", "exact_variance")):
            value = row[source_key]
            require(case["moments"][key] == F(int(value["numerator"]), int(value["denominator"])),
                    "independent moments disagree with the original simulation oracle")
        require(case["moments"]["gamma1"] == row["weight_sum"] and
                case["moments"]["distinct_edges"] == row["edges"],
                "independent geometry disagrees with original simulation")
    return {"status": "independent raw moments and graph invariants agree for both cases",
            "path": "results/calibration.json", "sha256": sha256(path),
            "protocol_sha256": sha256(STUDY/"results/protocol.json")}


def write_outputs(payload, directory):
    directory.mkdir(parents=True, exist_ok=True)
    target = directory/"exact_calibration.json"
    target.write_text(json.dumps(json_safe(payload), indent=2, allow_nan=False)+"\n")
    rows = []
    for case in payload["cases"]:
        for entry in case["rejections"]:
            for tail in ("lower_tail", "upper_tail", "two_sided"):
                bound = entry[tail]
                rows.append({"case_id": case["case_id"], "alpha": str(entry["alpha"]),
                    "tail": tail, "exact_probability_lower": str(bound["lower"]),
                    "exact_probability_upper": str(bound["upper"]),
                    "decimal_lower": bound["decimal_lower"], "decimal_upper": bound["decimal_upper"],
                    "percent_lower": bound["percent_lower"], "percent_upper": bound["percent_upper"],
                    "lower_tail_maximum_cut": entry["thresholds"]["lower_tail_maximum_cut"],
                    "upper_tail_minimum_cut": entry["thresholds"]["upper_tail_minimum_cut"],
                    "omitted_mass": str(case["omitted_probability_mass"]["lower"])})
    with (directory/"exact_calibration.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="small independent enumeration and exact threshold checks only")
    parser.add_argument("--output-dir", type=Path, default=STUDY/"results")
    args = parser.parse_args()
    start = time.time()
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    version = subprocess.run(compiler+["--version"], check=True, capture_output=True, text=True).stdout.splitlines()[0]
    quantiles = certify_quantiles()
    metadata = {"python": sys.version, "platform": platform.platform(), "compiler": version,
        "compile_flags": ["-std=c++17", "-O3", "-Wall", "-Wextra", "-pedantic"],
        "source_sha256": {p.name: sha256(p) for p in (Path(__file__), HERE/"enumerate_cut_histogram.cpp")},
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start)),
        "arithmetic": "Exact Python integers/Fraction and C++ uint64_t histogram counts; no floating-point decisions"}
    with tempfile.TemporaryDirectory(prefix="image_randomness_exact_") as temporary:
        executable = Path(temporary)/"enumerate_cut_histogram"
        subprocess.run(compiler+metadata["compile_flags"]+[str(HERE/"enumerate_cut_histogram.cpp"), "-o", str(executable)], check=True)
        checks = verify_small(executable)
        binary_hist = run_kernel(executable, 4, 4, "full", 16)
        require(all(binary_hist[k,c] == binary_hist.get((16-k,c),0) for k,c in binary_hist),
                "binary support-complement symmetry")
        binary = summarize("random_q2_4", 4, 4, F(1,2), 1, binary_hist, quantiles, True)
        require(binary["moments"]["raw_mean"]==27 and binary["moments"]["raw_variance"]==F(39,2),
                "binary benchmark moments")
        expected_binary = [(18,36), (15,39), (12,42)]
        for row, expected in zip(binary["rejections"], expected_binary):
            require((row["thresholds"]["lower_tail_maximum_cut"], row["thresholds"]["upper_tail_minimum_cut"])==expected,
                    "binary integer Gaussian thresholds")
        sparse_moments = exact_moments(16,16,F(1,1024),7)
        for alpha, expected in zip(ALPHAS, (14,17,21)):
            low, high, thresholds = rejection_sets(sparse_moments, quantiles[alpha],7)
            require(not low and min(high)==expected, "sparse integer Gaussian thresholds")
        checks += ["binary 4x4 complete histogram, complement symmetry, exact first two moments and integer thresholds",
                   "sparse 16x16 geometry, Bernoulli moments and integer Gaussian thresholds",
                   "all three normal quantiles certified by rational CDF enclosures",
                   "all score decisions separated from quantile brackets by a positive rational margin",
                   "directed decimal rounding"]
        if args.verify_only:
            print(json.dumps({"status":"passed", "checks":checks}, indent=2))
            return
        sparse_hist = run_kernel(executable,16,16,"sparse",4)
        sparse = summarize("known_sparse_16",16,16,F(1,1024),7,sparse_hist,quantiles,False)
        require(sparse["enumerated_support_count"] == sum(comb(256,k) for k in range(5)), "sparse support total")
        checks += ["each binary and sparse cardinality layer has exactly the hypergeometric first and second cut moments",
                   "independent moments and graph invariants match the original simulation oracle"]
    metadata["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    metadata["elapsed_seconds"] = time.time()-start
    metadata["environment_sha256"] = hashlib.sha256(json.dumps({k:metadata[k] for k in ("python","platform","compiler","compile_flags")},sort_keys=True).encode()).hexdigest()
    payload = {"status":"completed", "metadata":metadata,
        "rule":"Strict abs(S-mu)>z_(1-alpha/2)*sqrt(V); equality is acceptance. S is the unnormalized weighted absolute-difference sum.",
        "method":"All six pairs in each complete 2x2 window. For binary occupied support A, C(A)=sum_{i in A} weighted_degree(i)-2 sum_{i<j in A} weight(i,j). Raw S=amplitude*C(A).",
        "sparse_bound":"Each retained support with k occupied pixels keeps weight (1/1024)^k*(1023/1024)^(256-k). No renormalization. The omitted mass is exactly Pr(Binomial(256,1/1024)>4). Add it once to the total rejection probability. The lower-tail rejection set is empty over the full score range, hence its probability is exactly zero.",
        "decimal_rounding":"Probability intervals use 15 decimal places; percentages use 9. Lower endpoints are rounded toward minus infinity and upper endpoints toward plus infinity, directly from rational values.",
        "quantile_certification":"Rational candidates enclose each quantile because the upper CDF bound at the lower candidate is below 1-alpha/2 and the lower CDF bound at the upper candidate is above it. CDF bounds use 120 alternating integral-series terms, 100-term Machin arctangent bounds for pi, and exact integer square-root enclosures at 80 decimal places.",
        "quantiles":{str(a):v for a,v in quantiles.items()}, "verification":checks,
        "original_simulation_source":check_existing_moments([binary,sparse]), "cases":[binary,sparse]}
    write_outputs(payload,args.output_dir)
    for case in payload["cases"]:
        print(case["case_id"], "supports",case["enumerated_support_count"])
        for row in case["rejections"]:
            print("alpha",str(row["alpha"]),"total percent",row["two_sided"]["percent_lower"],row["two_sided"]["percent_upper"])
    print("Saved", args.output_dir/"exact_calibration.json", "and", args.output_dir/"exact_calibration.csv")


if __name__ == "__main__":
    main()
