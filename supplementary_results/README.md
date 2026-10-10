# Supplementary numerical results

Readable tables, standalone figures and plotting inputs for repeated-image BACI calibration, fixed-reference BACI patterns, natural-reference moments and UACI pooling. Every numerical export is derived from existing archived outputs. No new random observations are generated.

## Read the results

| Result | Readable output | Numerical export |
| --- | --- | --- |
| Complete repeated-image calibration, both nominal levels | [Table](generated/repeated_image_calibration.md) | [20 calibration rows](generated/repeated_image_calibration.csv) |
| Exact natural-reference moments | [Tables](generated/natural_reference_moments.md) | [Six original/permuted rows](generated/natural_reference_moments.csv) |
| Both numerical tables together | [HTML tables](generated/tables.html) | The two CSVs above |
| Fixed-reference BACI calibration and spatial arrangement | [Figure](figures/fixed_reference_calibration.pdf) | [20 plotted calibration points](generated/fixed_reference_calibration.csv), [two geometry pairs](generated/fixed_reference_geometry.csv) |
| UACI pooling and three null controls | [Figure](figures/uaci_pooling_calibration.pdf) | [18 plotted points](generated/uaci_calibration.csv) |

[DATA_MAP.md](DATA_MAP.md) maps every artifact and plotting series to its authoritative data file and method. The [verification record](generated/verification.json) records input/output hashes and numerical checks. Figures include their complete captions and are accompanied by editable LaTeX sources. No manuscript source tree is required to rebuild them.

[Verification and execution details](verification_and_execution.md) document the accompanying finite-law checks, exact calculations, archived observations and reproduction levels.

## Reproduce and check

Run these commands from the repository root. Python 3.11 or later is sufficient for numerical exports and checks; they use only the standard library. They do not access the network, draw random values or rewrite the underlying scientific outputs.

```sh
python -B supplementary_results/check.py
python -B supplementary_results/generate.py --output ../supplementary_results_rebuilt
python -B supplementary_results/check.py --generated ../supplementary_results_rebuilt
```

For a single verification command that also saves a regenerated copy outside the archive:

```sh
python -B supplementary_results/check.py --archive-root . --output ../supplementary_results_verified
```

That output directory must be absent and outside `--archive-root`. It receives the exports and `verification.json`; the supplied generated files are checked without modification.

Generation requires a new, absent output directory and refuses to overwrite supplied results. It creates the two CSV tables, two Markdown tables, one HTML document, the long-form plotting exports, seven compact plotting CSVs and `verification.json`. The checker compares every exported byte with a fresh in-memory derivation from the archived inputs. It also independently recomputes count/rate relations and Wilson intervals, checks natural-image decimal moments against their exact rational outputs, checks UACI invariance under the supplied permutation and checks squared-unit conversions. Verification does not rerun the original simulation campaigns.

The standalone figure builder uses the supplied `generated/` plotting files. After checking or generating the numerical inputs, run:

```sh
python -B supplementary_results/build_figures.py
```

Figure compilation additionally requires `pdflatex`, TikZ/PGFPlots 1.18, `geometry`, Latin Modern, `microtype`, `float` and `caption`. The builder compiles in a temporary directory and replaces the two PDFs in this module's `figures/` directory after successful compilation. Run it in a working copy to retain the supplied PDF bytes. Figures are separate from the standard-library numerical check. To rebuild all original summaries from saved observations, use the parent components and the archive's root verification instructions.

## Designs and interpretation

**Repeated-image BACI.** The complete table includes all five rules at nominal 1% and 5% for averages of 50 or 100 independent whole-image comparisons. Each random image is 256 by 256 with independent uniform 8-bit pixels. Each repetition count uses 5000 evaluation averages and a separate independent bank of 9999 averages. Both nominal levels are applied to the same evaluations. The tabulated rule accepts strictly between the decimal endpoints recorded in `core/expanded/config.json`. Pair independence, within-window covariance and full covariance are three Gaussian rules with a common exact mean and different variance assignments. Their Gaussian tails remain approximations.

**Fixed-reference BACI.** The reference images are 256 by 256 and contain constant 0, constant 128, a 0/128 checkerboard, a vertical split at column 128, or a column ramp equal to the zero-based column index. Comparison pixels are independent uniform 8-bit values. The figure shows nominal 5% rejection for 5000 evaluations per pattern and the four rules, plus exact means and variances for the checkerboard and split. Both arrangements contain 32768 pixels at each of 0 and 128. The complete two-level fixed-pattern calibration remains in `core/expanded/expected_rejection_rates.csv`; no observations have been discarded.

**Natural references.** Camera, Clock and Grass are centered 256 by 256 grayscale crops distributed with scikit-image 0.25.2. Their original integer pixels and one predetermined histogram-preserving permutation define fixed reference images. The permutation sorts row-major indices by SHA-256 of the ASCII prefix `fixed-image-permutation-v1`, a zero byte and the index encoded as four unsigned big-endian bytes; digest ties are broken by the index. It is a single deterministic control arrangement, not an average over random permutations. Independent uniform 8-bit comparison pixels define the conditional law. Exact finite sums produce all moments without simulation.

The [natural-images README](../fixed_image_and_power/natural_images/README.txt), [input manifest](../fixed_image_and_power/natural_images/input_manifest.json) and [third-party credits](../fixed_image_and_power/natural_images/THIRD_PARTY.txt) retain crop origins, hashes, sources and rights information. The [exact moment JSON](../fixed_image_and_power/natural_images/outputs/natural_image_moments_exact.json) preserves rational values before display rounding. Those provenance records remain authoritative.

**UACI controls.** The three 64 by 64 designs compare a constant-128 reference, a 0/128 checkerboard reference, or one random image with an independent random image. Each has 5000 evaluation scores and an independent bank of 9999 scores; the same scores are used at both nominal levels. The fixed-reference Gaussian comparisons use pooled or fixed-coordinate variances. These variances agree for the constant reference. Both plotted Gaussian series use the same random/random rule in the random control.

**Rank calibration and uncertainty.** The original calibration experiments use the conservative tie-inclusive rank rule, with p-value `(1 + number of bank deviations >= observed deviation)/(bank size + 1)`. Its finite-sample level guarantee is marginal over independent bank and observation draws. A reused bank can have a different conditional rejection probability. Wilson 95% intervals attached to rank rejection rates describe uncertainty conditional on that realized bank. The two figure exports do not implement or substitute the randomized boundary rule used in the separate power experiment. These diagnostics assess compatibility with the declared randomness model.

## Units and display precision

CSV means for natural-image BACI and UACI are fractions; variances and pooling excesses are squared fractions. Their Markdown/HTML tables display means in percent and variances multiplied by `10^7`, while retaining squared fraction units. Dimensionless variance ratios are not percentages. The natural tables round their displayed entries to five decimal places; complete archived decimal fields and exact rational outputs remain available.

Rejection rates in the repeated/fixed CSVs are fractions in `rejection_rate`, percentages in `rejection_percent`, and percentages in fields ending in `_percent`. UACI uses `rate` for fractions and `rejection_percent` for percentages. All plotted error-bar lengths are percentage points. Geometry exports distinguish `variance_squared_fraction` from `variance_squared_percentage_points`, which is 10000 times larger. The compact geometry plotting file retains `sd_percent`; the figure squares this field for the variance panel. Independent repetition scaling is already included in the archived repeated-image moments and must not be applied again.

## References and coverage

The tabulated BACI endpoint criterion and pair-occurrence comparator are attributed to Y. Zhang, “Statistical test criteria for sensitivity indexes of image cryptosystems,” *Information Sciences* 550 (2021), 313–328, [doi:10.1016/j.ins.2020.10.026](https://doi.org/10.1016/j.ins.2020.10.026). Figure captions retain this attribution.

This module supplies the named calibration displays and natural-image tables. Full entropy intervals and exact binary controls remain in `reference_calibration/entropy/`; larger-window outputs remain in `window_reference/`. The comparison of externally reported entropy scores remains in the accompanying supplement. Proofs, notation and source-formula correspondence remain in the accompanying article and supplement.
