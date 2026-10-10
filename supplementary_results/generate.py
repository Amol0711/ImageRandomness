"""Export supplementary tables and plot inputs from archived numerical outputs.

Python 3.11+, standard library only. No random draws, network or source writes.
"""
from __future__ import annotations

import argparse
import csv
import html
import io
import math
from pathlib import Path

MODULE = Path(__file__).resolve().parent
ARCHIVE = MODULE.parent
INPUTS = {
    "expanded_rates": "core/expanded/expected_rejection_rates.csv",
    "expanded_moments": "core/expanded/expected_moments.csv",
    "baseline_rates": "core/baseline/expected/rejection_rates.csv",
    "natural_moments": "fixed_image_and_power/natural_images/outputs/natural_image_moments.csv",
    "natural_exact": "fixed_image_and_power/natural_images/outputs/natural_image_moments_exact.json",
}
FIXED = ("fixed_zero", "fixed_mid", "fixed_checker", "fixed_split", "fixed_ramp")
UACI = ("UACI_fixed_constant_128_64x64", "UACI_fixed_checkerboard_0_128_64x64", "random_random_64x64")
METHODS = ("fixed_interval", "independent_pair_gaussian", "within_block_only_gaussian", "full_covariance_gaussian", "independent_null_rank")
METHOD_LABELS = ("Tabulated", "Pair independence", "Within-window", "Full covariance", "Null rank")
FIXED_PLOTS = {
    "fixed_reference_pair.csv": "independent_pair_gaussian",
    "fixed_reference_within_window.csv": "within_block_only_gaussian",
    "fixed_reference_full_covariance.csv": "full_covariance_gaussian",
    "fixed_reference_rank.csv": "independent_null_rank",
}


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def csv_bytes(rows, fields=None):
    if fields is None:
        fields = list(rows[0])
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def markdown_table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"] + ["| " + " | ".join(map(str, row)) + " |" for row in rows]) + "\n"


def html_table(headers, rows):
    return "<table><thead><tr>" + "".join("<th>" + html.escape(h) + "</th>" for h in headers) + "</tr></thead><tbody>" + "".join("<tr>" + "".join("<td>" + html.escape(str(c)) + "</td>" for c in row) + "</tr>" for row in rows) + "</tbody></table>"


def build_outputs(archive_root=ARCHIVE):
    """Return deterministic relative-name -> bytes, without writing anything."""
    archive_root = Path(archive_root)
    expanded = read_rows(archive_root / INPUTS["expanded_rates"])
    moments = read_rows(archive_root / INPUTS["expanded_moments"])
    baseline = read_rows(archive_root / INPUTS["baseline_rates"])
    natural = read_rows(archive_root / INPUTS["natural_moments"])
    ei = {(r["design"], r["alpha"], r["method"]): r for r in expanded}
    mi = {r["design"]: r for r in moments}
    bi = {(r["design"], r["alpha"], r["method"]): r for r in baseline if r["metric"] == "uaci"}
    out = {}

    repeated = []
    repeated_display = []
    for repeats in (50, 100):
        for alpha in ("0.01", "0.05"):
            values = []
            for method in METHODS:
                row = ei[f"random_R{repeats}", alpha, method]
                repeated.append({"repeats": repeats, **row})
                values.append(f"{float(row['rejection_percent']):.2f}")
            repeated_display.append([str(repeats), f"{100 * float(alpha):.0f}", *values])
    out["repeated_image_calibration.csv"] = csv_bytes(repeated)
    repeated_headers = ["Independent whole-image repeats R", "Nominal level (%)", *[x + " (%)" for x in METHOD_LABELS]]
    repeated_note = (
        "Null rejection percentages for averages of R independent 256 by 256 BACI image-pair statistics, "
        "with independent uniform 8-bit pixels in each random image. Each R uses 5000 evaluation averages "
        "and a separate independent bank of 9999 averages. Both nominal levels use those same evaluations. "
        "Tabulated applies Zhang's decimal endpoints with acceptance strictly between them. The three "
        "Gaussian rules use pair-occurrence independence, within-window covariance, or full covariance. "
        "Exact moments do not make their finite-sample tails Gaussian. Null rank includes upper-tail ties; "
        "its finite-sample guarantee is marginal over bank and observation. The CSV retains counts and "
        "Wilson 95% intervals; the rank intervals condition on the realized bank. Rates in this table are "
        "percentages; CSV rejection_rate and Wilson endpoints without a percent suffix are fractions."
    )
    out["repeated_image_calibration.md"] = ("# Complete repeated-image BACI calibration\n\n" + repeated_note + "\n\n" + markdown_table(repeated_headers, repeated_display) + "\nBACI means blocked average changing intensity. See ../README.md for the data and method crosswalk and Zhang (2021) citation.\n").encode()

    # Retain every original natural-moment column and its decimal precision.
    out["natural_reference_moments.csv"] = csv_bytes(natural)
    baci_headers = ["Image", "Arrangement", "BACI mean (%)", "BACI variance x 10^7", "Full/Pair"]
    baci_display = [[r["image"].title(), r["arrangement"].title(), f"{100 * float(r['baci_mean']):.5f}", f"{1e7 * float(r['baci_variance']):.5f}", f"{float(r['baci_full_to_pair']):.5f}"] for r in natural]
    uaci_headers = ["Image", "UACI mean (%)", "Fixed variance x 10^7", "Pooling excess x 10^7", "Pool/Fixed"]
    uaci_display = [[r["image"].title(), f"{100 * float(r['uaci_mean']):.5f}", f"{1e7 * float(r['uaci_fixed_variance']):.5f}", f"{1e7 * float(r['uaci_pooled_excess']):.5f}", f"{float(r['uaci_pooled_to_fixed']):.5f}"] for r in natural if r["arrangement"] == "original"]
    natural_note = (
        "Exact conditional moments for fixed 256 by 256 grayscale crops and one predetermined "
        "histogram-preserving permutation of each crop. Comparison pixels are independent uniform "
        "8-bit values. No Monte Carlo estimates are used. Means are displayed as percentages; all "
        "displayed variances use squared fraction units multiplied by 10^7. Full/Pair divides the full "
        "BACI variance by its pair-occurrence independence approximation. Pool/Fixed divides pooled "
        "UACI variance by fixed-coordinate UACI variance. UACI denotes unified average changing intensity. "
        "Original and permuted arrangements have exactly equal UACI moments. CSV means are fractions, "
        "variances and excesses are squared fractions, and ratios are dimensionless."
    )
    out["natural_reference_moments.md"] = ("# Natural-reference moments\n\n" + natural_note + "\n\n## BACI and spatial arrangement\n\n" + markdown_table(baci_headers, baci_display) + "\n## UACI and histogram preservation\n\n" + markdown_table(uaci_headers, uaci_display) + "\nExact rational outputs, image provenance and permutation instructions remain in ../../fixed_image_and_power/natural_images/. See ../README.md.\n").encode()

    fixed_long = []
    for filename, method in FIXED_PLOTS.items():
        plot = []
        for x, design in enumerate(FIXED, 1):
            row = ei[design, "0.05", method]
            rate = float(row["rejection_percent"])
            minus = rate - float(row["wilson95_lower_percent"])
            plus = float(row["wilson95_upper_percent"]) - rate
            plot.append({"x": x, "rate": rate, "minus": minus, "plus": plus})
            fixed_long.append({"x": x, **row, "error_minus_percentage_points": minus, "error_plus_percentage_points": plus})
        out[filename] = csv_bytes(plot)
    out["fixed_reference_calibration.csv"] = csv_bytes(fixed_long)
    geometry, geometry_plot = [], []
    for x, design in enumerate(("fixed_checker", "fixed_split"), 1):
        row = mi[design]
        mean = float(row["mean_percent"])
        variance = float(row["full_variance"])
        geometry.append({"x": x, "design": design, "mean_percent": mean, "variance_squared_fraction": row["full_variance"], "variance_squared_percentage_points": 10000 * variance})
        geometry_plot.append({"x": x, "mean_percent": mean, "sd_percent": 100 * math.sqrt(variance)})
    out["fixed_reference_geometry.csv"] = csv_bytes(geometry)
    out["fixed_reference_geometry_plot.csv"] = csv_bytes(geometry_plot)

    uaci_long = []
    for alpha in ("0.01", "0.05"):
        plot = []
        for x, design in enumerate(UACI, 1):
            item = {"x": x, "design": design}
            for series, fixed_method in (("source", "pooled_pixel_gaussian"), ("corrected", "fixed_position_gaussian"), ("rank", "independent_rank")):
                method = "standard_gaussian" if design == "random_random_64x64" and series != "rank" else fixed_method
                row = bi[design, alpha, method]
                rate = 100 * float(row["rate"])
                minus = 100 * (float(row["rate"]) - float(row["wilson_low"]))
                plus = 100 * (float(row["wilson_high"]) - float(row["rate"]))
                item.update({series: rate, series + "_minus": minus, series + "_plus": plus})
                uaci_long.append({"x": x, "series": series, **row, "rejection_percent": rate, "error_minus_percentage_points": minus, "error_plus_percentage_points": plus, "interval_conditioning": "realized null bank" if series == "rank" else "fixed decision rule"})
            plot.append(item)
        out["uaci_nominal_" + alpha[-2:] + ".csv"] = csv_bytes(plot)
    out["uaci_calibration.csv"] = csv_bytes(uaci_long)

    body = "<h1>Supplementary numerical tables</h1><p>Generated from the archived outputs without new random draws. <a href='../README.md'>Methods, data mapping and reproduction instructions</a>.</p>"
    body += "<h2>Complete repeated-image BACI calibration</h2><p>" + html.escape(repeated_note) + "</p>" + html_table(repeated_headers, repeated_display)
    body += "<h2>Natural-reference moments</h2><p>" + html.escape(natural_note) + "</p><h3>BACI and spatial arrangement</h3>" + html_table(baci_headers, baci_display) + "<h3>UACI and histogram preservation</h3>" + html_table(uaci_headers, uaci_display)
    body += "<p>Exact rational outputs and third-party source credits are retained in <a href='../../fixed_image_and_power/natural_images/README.txt'>the natural-images component</a>.</p>"
    out["tables.html"] = ("<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Supplementary numerical tables</title><style>body{max-width:1100px;margin:2rem auto;padding:0 1rem;font:16px/1.5 system-ui,sans-serif;color:#172f3d}table{border-collapse:collapse;width:100%;margin:1.5rem 0;font-size:14px}th,td{padding:.55rem;text-align:right;border-bottom:1px solid #cbd5dd}th{background:#edf3f6}th:first-child,td:first-child{text-align:left}h2{margin-top:2.5rem}@media print{body{font-size:10pt;margin:0}table{font-size:9pt}tr{break-inside:avoid}}</style><body>" + body + "</body></html>\n").encode()
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, default=ARCHIVE)
    parser.add_argument("--output", type=Path, required=True, help="New, absent output directory")
    args = parser.parse_args()
    destination = args.output.resolve()
    if destination.exists():
        parser.error("Output directory already exists; choose a new directory.")
    payloads = build_outputs(args.archive_root.resolve())
    destination.mkdir(parents=True)
    for name, contents in payloads.items():
        (destination / name).write_bytes(contents)
    import check
    report = check.verify(args.archive_root.resolve(), destination)
    (destination / "verification.json").write_text(check.report_text(report), encoding="utf-8")
    print(check.report_text({"status": report["status"], "generated_files": len(payloads), "random_draws": 0}).strip())


if __name__ == "__main__":
    main()
