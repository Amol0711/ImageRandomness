# Practical calibration and analytic power interpretation

This component combines deterministic Gaussian BACI intervals, central-shape
diagnostics from saved null observations and analytic interpretations of the
saved paired BACI/UACI/NPCR power experiment. It makes no new random draws.
The four existing components supply its exact moment routines and numerical
inputs.

Run from the archive root with the pinned Python dependencies installed.

```sh
python -B practical_calibration/reproduce.py --archive-root . --output ../practical_calibration_rebuild
```

Choose an absent output directory outside the archive. The launcher verifies
all supplied hashes before and after execution, regenerates all seven numerical
files in that directory and requires byte equality with the supplied results.
It records the comparisons in verification_report.json. The default root
verify_all.py command also runs this component.

## Reference intervals and saved null shape

The intervals concern two independent uniform 8-bit images, every overlapping
2 by 2 window, square sides 256, 512 and 1024, independent whole-image
replication counts 1 and 100, and two-sided nominal levels 0.05 and 0.01.
Exact rational moments are checked against independent degree-count and
finite-alphabet calculations and the callable core routine. NormalDist and
SciPy quantiles independently check all endpoints. Small-grid enumeration
checks the edge multiplicities. These intervals use a Gaussian approximation.

The intervals/generated directory contains the following files.

- baci_practical_intervals.csv has 12 design-level rows. Scores and variances
  use proportions and squared proportions; percent endpoints multiply scores
  by 100. Standard errors in percentage points also multiply proportion
  standard errors by 100. Independent replication divides variance once by R.
- baci_saved_null_shape.csv uses only the 5000 fresh evaluation observations
  for each of the three sides 16, 64 and 256. Calibration banks are excluded.
  Bias-corrected skewness and excess kurtosis are accompanied by Gaussian
  reference standard errors. Those errors are not estimates under the BACI law
  and the shape diagnostics do not certify small-tail accuracy.
- practical_calibration_checks.json records exact moment comparisons,
  endpoint checks, replication scaling, literal grid checks and input hashes.

To regenerate only these calculations, choose another absent output directory.

```sh
python -B practical_calibration/intervals/generate_practical_calibration.py --archive-root . --output ../baci_reference_rebuild --saved-null-shape
```

## Analytic interpretation of power

The power directory evaluates the saved partial-identity and horizontal-copy
designs. Its four outputs are in power/analytic_results. It reconstructs the
six saved numerical score thresholds and tie boundaries, reproduces 36
conditional rejection counts, and checks exact rational moments by finite-law
enumeration on small overlapping grids.

NPCR power under partial identity uses binomial probabilities and the saved
randomized boundary. BACI/UACI values use continuous Gaussian surrogates with
exact alternative moments at the saved numerical cutoffs. The horizontal-copy
calculation instead uses ideal Gaussian thresholds for the variance-only
UACI/NPCR response and evaluates the exact BACI mean shift. It does not assert
exact finite-grid size independence or a universal BACI power curve.
The detailed power/README.txt gives the output schemas, conditioning and
approximation conventions, including the Gaussian comparisons outside the
pointwise Monte Carlo Wilson intervals.

```sh
python -B practical_calibration/power/analytic_interpretation.py --archive-root . --output ../analytic_power_rebuild
```

Gaussian intervals, exact moments, exact binomial probabilities and conditional
Monte Carlo estimates retain separate numerical labels. Random/random variance
ratios do not provide universal fixed-reference correction factors. Confidence
intervals for a reused bank condition on that bank. The existing license status
and third-party notices at the archive root apply without modification.
