"""Regenerate the exact finite-window numerical table using Python >=3.9."""

import csv
import json
from pathlib import Path
from core import window_moments, large_grid_ratio, large_alphabet_large_grid_ratio


def main():
    destination = Path(__file__).resolve().parent / "generated"
    destination.mkdir(exist_ok=True)
    with (destination / "window_variance_ratios.csv").open("w", newline="") as stream:
        fields = ["q", "window_width", "rows", "columns", "ratio_numerator",
                  "ratio_denominator", "ratio_decimal", "variance_numerator",
                  "variance_denominator", "variance_decimal"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for width in range(2, 6):
            for size in (16, 64, 256):
                result = window_moments(size, size, width, q=256)
                ratio, variance = result["variance_ratio"], result["variance"]
                writer.writerow({"q": 256, "window_width": width, "rows": size,
                                 "columns": size, "ratio_numerator": ratio.numerator,
                                 "ratio_denominator": ratio.denominator,
                                 "ratio_decimal": format(float(ratio), ".12f"),
                                 "variance_numerator": variance.numerator,
                                 "variance_denominator": variance.denominator,
                                 "variance_decimal": format(float(variance), ".16g")})
            ratio = large_grid_ratio(width, q=256)
            writer.writerow({"q": 256, "window_width": width, "rows": "infinity",
                             "columns": "infinity", "ratio_numerator": ratio.numerator,
                             "ratio_denominator": ratio.denominator,
                             "ratio_decimal": format(float(ratio), ".12f"),
                             "variance_numerator": "", "variance_denominator": "",
                             "variance_decimal": ""})
    with (destination / "window_large_alphabet_limits.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["window_width", "ratio_numerator", "ratio_denominator",
                         "ratio_decimal"])
        for width in range(2, 6):
            ratio = large_alphabet_large_grid_ratio(width)
            writer.writerow([width, ratio.numerator, ratio.denominator,
                             format(float(ratio), ".12f")])
    metadata = {
        "arithmetic": "Exact fractions; only decimal display uses binary floating point",
        "pixel_model": "Two mutually independent iid uniform images",
        "boundary_model": "All fully contained sliding square windows; no wrapping",
        "score": "Mean of all unordered within-window pair distances, normalized by q-1",
        "ratio": "Exact variance divided by pair-occurrence independence approximation",
        "grid_limit": "Fixed alphabet and window width; both dimensions tend to infinity",
        "dependencies": "Python standard library only",
    }
    (destination / "window_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print("Generated 12 finite-grid ratios, 4 grid limits, and 4 large-alphabet limits.")


if __name__ == "__main__":
    main()
