"""Independent checks of finite counts and score moments, no simulations.

The reference calculation explicitly enumerates windows and pair incidences.
Selected small images are checked by summing their complete probability law
and evaluating each window score directly, without an edge representation.
"""

from collections import Counter, defaultdict
import csv
from fractions import Fraction
from itertools import combinations, product
import json
from math import comb, prod
from pathlib import Path

from core import (grid_counts, pair_moments, window_moments, large_grid_ratio,
                  large_alphabet_large_grid_ratio)


def direct_pair_moments(q):
    weights = [q] + [2*(q-d) for d in range(1, q)]
    assert sum(weights) == q*q
    conditional_numerators = [
        sum(weights[y]*abs(x-y) for y in range(q)) for x in range(q)]
    first = Fraction(sum(weights[x]*conditional_numerators[x]
                         for x in range(q)), q**4)
    second = Fraction(sum(weights[x]*weights[y]*(x-y)**2
                          for x in range(q) for y in range(q)), q**4)
    shared_product = Fraction(sum(weights[x]*conditional_numerators[x]**2
                                  for x in range(q)), q**6)
    return first, second-first**2, shared_product-first**2


def direct_graph_counts(rows, columns, width):
    edges = Counter()
    for i in range(rows-width+1):
        for j in range(columns-width+1):
            window = [(i+di, j+dj) for di in range(width) for dj in range(width)]
            edges.update(combinations(window, 2))
    incident = defaultdict(list)
    for (first, second), multiplicity in edges.items():
        incident[first].append(multiplicity)
        incident[second].append(multiplicity)
    # Distinct edges in a simple graph share at most one endpoint.
    shared_weight = sum(a*b for values in incident.values()
                        for a, b in combinations(values, 2))
    return {"pair_occurrences": sum(edges.values()),
            "pair_squares": sum(value**2 for value in edges.values()),
            "degree_squares": sum(sum(values)**2 for values in incident.values()),
            "shared_edge_weight": shared_weight}


def direct_image_distribution(rows, columns, width, q):
    # This evaluates windows directly, never the weighted-edge formula.
    windows = [[(i+di)*columns+j+dj for di in range(width) for dj in range(width)]
               for i in range(rows-width+1) for j in range(columns-width+1)]
    pixel_weights = [q] + [2*(q-d) for d in range(1, q)]
    mass = first = second = 0
    for image in product(range(q), repeat=rows*columns):
        weight = prod(pixel_weights[x] for x in image)
        score = 0
        for window in windows:
            histogram = [0]*q
            for pixel in window:
                histogram[image[pixel]] += 1
            score += sum(histogram[x]*histogram[y]*(y-x)
                         for x in range(q) for y in range(x+1, q))
        mass += weight
        first += weight*score
        second += weight*score*score
    denominator = q**(2*rows*columns)
    assert mass == denominator
    normalizer = len(windows)*comb(width*width, 2)*(q-1)
    mean = Fraction(first, denominator*normalizer)
    variance = Fraction(second, denominator*normalizer**2)-mean**2
    return mean, variance


def main():
    results = {"arithmetic": "exact integer and rational", "status": "PASS"}
    q_values = (2, 3, 7, 16, 256)
    direct = {q: direct_pair_moments(q) for q in q_values}
    for q in q_values:
        assert direct[q] == pair_moments(q), q
    results["independent_pair_laws"] = list(q_values)

    rectangle_count = covariance_count = 0
    for width in range(2, 7):
        for rows in range(width, 11):
            for columns in range(width, 11):
                reference = direct_graph_counts(rows, columns, width)
                closed = grid_counts(rows, columns, width)
                for key, value in reference.items():
                    assert value == closed[key], (rows, columns, width, key)
                rectangle_count += 1
                for q in (2, 3, 7):
                    _, variance, covariance = direct[q]
                    reference_variance = (
                        variance*reference["pair_squares"]
                        + 2*covariance*reference["shared_edge_weight"]
                    ) / (reference["pair_occurrences"]*(q-1))**2
                    assert reference_variance == window_moments(
                        rows, columns, width, q)["variance"]
                    covariance_count += 1
                if width == 2:
                    assert reference["pair_squares"] == (
                        10*rows*columns-12*rows-12*columns+14)
                    assert reference["shared_edge_weight"] == (
                        62*rows*columns-96*rows-96*columns+148)
    results["enumerated_rectangles"] = rectangle_count
    results["rectangle_domain"] = "width 2..6; both dimensions width..10"
    results["independent_covariance_variances"] = covariance_count
    results["width_two_reductions"] = 81

    single_window_count = 0
    for width in range(2, 10):
        for q in q_values:
            _, variance, covariance = direct[q]
            actual = window_moments(width, width, width, q)
            expected = 1 + 2*(width*width-2)*covariance/variance
            assert actual["variance_ratio"] == expected
            single_window_count += 1
    results["single_window_endpoint_checks"] = single_window_count

    cases = [(3, 4, 3, 2), (4, 4, 3, 2), (2, 3, 2, 3),
             (3, 3, 2, 3), (3, 3, 3, 3)]
    checked = []
    for rows, columns, width, q in cases:
        mean, variance = direct_image_distribution(rows, columns, width, q)
        formula = window_moments(rows, columns, width, q)
        assert (mean, variance) == (formula["mean"], formula["variance"])
        checked.append({"rows": rows, "columns": columns, "width": width,
                        "q": q, "difference_images": q**(rows*columns),
                        "mean": str(mean), "variance": str(variance)})
    results["complete_distribution_checks"] = checked
    results["difference_images_enumerated"] = sum(item["difference_images"]
                                                for item in checked)

    # A separate limit calculation uses sums over ordered signed shifts.
    limit_checks = 0
    for width in range(2, 20):
        overlap = sum(width-abs(shift) for shift in range(1-width, width))
        square_overlap = sum((width-abs(shift))**2
                             for shift in range(1-width, width))
        assert overlap == width*width
        assert square_overlap == Fraction(width*(2*width*width+1), 3)
        pair_squares = Fraction(square_overlap**2-overlap**2, 2)
        degree_squares = (width*width-1)**2*overlap**2
        for q in q_values:
            _, variance, covariance = direct[q]
            ratio = ((variance-2*covariance)*pair_squares
                     + covariance*degree_squares) / (comb(width*width, 2)*variance)
            assert ratio == large_grid_ratio(width, q)
            limit_checks += 1
        limiting_ratio = (Fraction(2, 3)*pair_squares + Fraction(1, 6)*degree_squares
                          ) / comb(width*width, 2)
        assert limiting_ratio == large_alphabet_large_grid_ratio(width)
    results["independent_grid_limit_checks"] = limit_checks
    results["large_alphabet_limit_checks"] = 18

    destination = Path(__file__).resolve().parent / "generated"
    with (destination / "window_variance_ratios.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 16
    for row in rows:
        width, q = int(row["window_width"]), int(row["q"])
        exact = Fraction(int(row["ratio_numerator"]), int(row["ratio_denominator"]))
        if row["rows"] == "infinity":
            assert exact == large_grid_ratio(width, q)
            assert row["variance_numerator"] == ""
        else:
            formula = window_moments(int(row["rows"]), int(row["columns"]), width, q)
            assert exact == formula["variance_ratio"]
            assert Fraction(int(row["variance_numerator"]),
                            int(row["variance_denominator"])) == formula["variance"]
        assert row["ratio_decimal"] == format(float(exact), ".12f")
    results["generated_ratio_rows"] = len(rows)
    (destination / "window_checks.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
