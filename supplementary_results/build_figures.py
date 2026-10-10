#!/usr/bin/env python3
"""Build the two standalone native TikZ/PGFPlots figures.

Use the supplied generated/ data or regenerate it as described in README.md.
Python uses only its standard
library. PDF generation additionally requires pdflatex with PGFPlots 1.18 or
newer, TikZ, Latin Modern, microtype, geometry, float and caption. A typical
TeX Live installation provides these through its recommended fonts, pictures
and LaTeX-extra packages. No network access is used.
"""

from pathlib import Path
import argparse
import re
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parent
FIGURES = ("fixed_reference_calibration", "uaci_pooling_calibration")
DATA = (
    "fixed_reference_pair.csv",
    "fixed_reference_within_window.csv",
    "fixed_reference_full_covariance.csv",
    "fixed_reference_rank.csv",
    "fixed_reference_geometry_plot.csv",
    "uaci_nominal_01.csv",
    "uaci_nominal_05.csv",
)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    engine = shutil.which("pdflatex")
    if engine is None:
        raise SystemExit(
            "pdflatex was not found. Install TeX Live with PGFPlots, TikZ, "
            "Latin Modern, microtype, geometry, float and caption, then retry."
        )
    missing = [name for name in DATA if not (ROOT / "generated" / name).is_file()]
    if missing:
        raise SystemExit(
            "Missing plotting inputs in generated/. Follow the data generation "
            "instructions in README.md before building figures.\n"
            + "\n".join(missing)
        )
    # Build outside shared workspaces so auxiliary writes are fully visible
    # when pdfTeX rereads named PGFPlots legends at the end of each pass.
    with tempfile.TemporaryDirectory(prefix="image_randomness_figures_") as folder:
        work = Path(folder)
        (work / "figures").mkdir()
        (work / "generated").mkdir()
        for name in DATA:
            shutil.copyfile(ROOT / "generated" / name, work / "generated" / name)
        for stem in FIGURES:
            shutil.copyfile(ROOT / "figures" / (stem + ".tex"), work / "figures" / (stem + ".tex"))
            for _ in range(2):
                result = subprocess.run(
                    [engine, "-interaction=nonstopmode", "-halt-on-error", stem + ".tex"],
                    cwd=work / "figures", text=True, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                if result.returncode:
                    raise RuntimeError("Figure compilation failed.\n" + result.stdout)
            log = (work / "figures" / (stem + ".log")).read_text(errors="replace")
            warnings = re.findall(
                r"^.*(?:undefined references|Citation .* undefined|Reference .* undefined|"
                r"multiply defined|Overfull|Underfull|Fatal error|LaTeX Warning|Package .* Warning).*$",
                log, re.M,
            )
            pages = re.search(r"Output written on " + re.escape(stem) + r"\.pdf \((\d+) pages?", log)
            if warnings or pages is None or int(pages.group(1)) != 1:
                raise RuntimeError(stem + " did not pass the one-page, warning-free build check.\n" + "\n".join(warnings))
        # Replace outputs only after both figures have passed their checks.
        for stem in FIGURES:
            shutil.copyfile(work / "figures" / (stem + ".pdf"), ROOT / "figures" / (stem + ".pdf"))
            print("Built figures/" + stem + ".pdf (one page)")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        sys.exit(str(error))
