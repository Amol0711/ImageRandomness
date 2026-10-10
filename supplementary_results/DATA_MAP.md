# Artifact-to-data map

All paths below are relative to the repository root. `generate.py` selects existing rows and changes presentation units only. It does not estimate new moments, choose new thresholds or generate new observations. Names of source methods in this map are literal identifiers in the authoritative CSVs.

| Output under `supplementary_results/generated/` | Authoritative input | Selection and transformation |
| --- | --- | --- |
| `repeated_image_calibration.csv` and `.md` | `core/expanded/expected_rejection_rates.csv` | `random_R50` and `random_R100`; both levels; all five methods. Add explicit `repeats`, retain all archived rate/count/Wilson fields. Markdown displays percentages to two decimals. |
| `natural_reference_moments.csv` and `.md` | `fixed_image_and_power/natural_images/outputs/natural_image_moments.csv` | All six rows and all source columns. Markdown multiplies means by 100 and squared-fraction variances/excesses by `10^7`; displays five decimals. UACI presentation uses one row per image because permutation pairs agree exactly. |
| `tables.html` | Same two inputs | The complete repeated-image and natural-reference tables, with the same definitions and rounding as Markdown. |
| `fixed_reference_calibration.csv` | `core/expanded/expected_rejection_rates.csv` | All five fixed designs, nominal 0.05, four methods. Retain counts and Wilson endpoints; add asymmetric error lengths in percentage points. |
| `fixed_reference_geometry.csv` and `fixed_reference_geometry_plot.csv` | `core/expanded/expected_moments.csv` | `fixed_checker`, `fixed_split`; exact-formula decimal means/variances. Geometry plot uses mean percent and 100 times the square root of the squared-fraction variance. |
| `uaci_calibration.csv` | `core/baseline/expected/rejection_rates.csv` | Metric `uaci`; constant, checkerboard and random/random designs; two levels and three plotted series, including the duplicated random/random Gaussian series. Retain counts/rates; derive error lengths from saved Wilson endpoints. |
| `verification.json` | Five named input files plus generated outputs | Input/output SHA-256; complete byte comparisons; count/rate/Wilson checks; natural decimal values against exact rational JSON; permutation and unit checks. No simulation or figure compilation. |

## Repeated-image method labels

| Readable column | Archive method identifier | Decision |
| --- | --- | --- |
| Tabulated | `fixed_interval` | Reject at or beyond Zhang's configured decimal endpoints. |
| Pair independence | `independent_pair_gaussian` | Gaussian interval from pair-occurrence independence variance. |
| Within-window | `within_block_only_gaussian` | Gaussian interval retaining covariance inside each window. |
| Full covariance | `full_covariance_gaussian` | Gaussian interval including pair reuse and shared-pixel covariance. |
| Null rank | `independent_null_rank` | Conservative independent-null rank rule with upper-tail ties included. |

The Gaussian rules reject strictly beyond their unrounded endpoints. The tabulated and pair-independence Gaussian rules are distinct comparisons. The bank contains 9999 whole-image averages and the evaluation sample 5000 averages for each R; each average contains R independent complete image-pair statistics.

## Fixed-reference figure inputs

`figures/fixed_reference_calibration.tex` reads these compact CSVs. Their common schema is `x,rate,minus,plus`, with rates in percent and errors in percentage points.

| Compact plotting CSV | Archive method |
| --- | --- |
| `fixed_reference_pair.csv` | `independent_pair_gaussian` |
| `fixed_reference_within_window.csv` | `within_block_only_gaussian` |
| `fixed_reference_full_covariance.csv` | `full_covariance_gaussian` |
| `fixed_reference_rank.csv` | `independent_null_rank` |

The x positions 1–5 map to `fixed_zero`, `fixed_mid`, `fixed_checker`, `fixed_split`, `fixed_ramp`. All four files select nominal 0.05. `minus = rejection_percent - wilson95_lower_percent` and `plus = wilson95_upper_percent - rejection_percent`. Exact means are fixed-design conditional means.

`fixed_reference_geometry_plot.csv` has schema `x,mean_percent,sd_percent`. Here x=1 is checkerboard and x=2 is split. The figure plots `sd_percent` squared for its variance panel. The five original pattern arrays are in `core/expanded/data/fixed_images/`; the figure's checkerboard thumbnail is an enlarged 8 by 8 crop.

The complete underlying integer observations are the five `core/expanded/data/fixed_*_raw.npz` archives. The repeated-image archives are `random_R50_raw.npz` and `random_R100_raw.npz`. Their first 9999 rows are the bank and their final 5000 rows are evaluation units. The core component documents exact centering and preserved rank ties.

## UACI figure inputs

`figures/uaci_pooling_calibration.tex` reads `uaci_nominal_01.csv` and `uaci_nominal_05.csv`. Both retain the compact schema

```text
x,design,source,source_minus,source_plus,corrected,corrected_minus,corrected_plus,rank,rank_minus,rank_plus
```

The names `source` and `corrected` identify established plotting columns; the readable legend states their scientific meaning.

| x | Design | `source` series | `corrected` series | `rank` series |
| --- | --- | --- | --- | --- |
| 1 | `UACI_fixed_constant_128_64x64` | `pooled_pixel_gaussian` | `fixed_position_gaussian` | `independent_rank` |
| 2 | `UACI_fixed_checkerboard_0_128_64x64` | `pooled_pixel_gaussian` | `fixed_position_gaussian` | `independent_rank` |
| 3 | `random_random_64x64` | `standard_gaussian` | `standard_gaussian` | `independent_rank` |

Each ordinate is `100*rate`; the lower and upper error lengths are `100*(rate-wilson_low)` and `100*(wilson_high-rate)`. The random/random Gaussian rule appears twice to preserve the common three-series legend; these are not independent experiments. The underlying saved scores are in `core/baseline/data/null_samples.npz`, using the named design plus `_uaci_bank` or `_uaci_fresh`. No individual image arrays are needed to reproduce the reported rates from these scores.

## Provenance and boundaries

Natural-image exact fractions are in `fixed_image_and_power/natural_images/outputs/natural_image_moments_exact.json`; the generator preserves original decimal output fields and the checker compares them with those fractions. The crop arrays, deterministic permutation, third-party credits and input hashes remain in that component. This module neither replaces source credits nor changes any supplied image pixels.

The figure PDFs are compiled presentations of these exports, not additional simulation runs. Their attribution, conditioning statements and units are included in the captions. Hashes in `generated/verification.json` cover numerical exports; figure compilation is checked separately and the repository-wide manifest covers the supplied PDFs and figure sources.
