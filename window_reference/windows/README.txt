FINITE SLIDING-WINDOW CALIBRATION

Run from this directory using Python 3.9 or later.

    python generate.py
    python check.py

No third-party packages are needed. All reference calculations use exact
integers and fractions. Only displayed decimal columns use floating point.
The checker enumerates 255 finite rectangular designs, checks shared-edge
covariances independently, and sums the full probability laws of five small
image designs. These are deterministic numerical checks, not Monte Carlo
trials. The generated check record contains the exact counts and outcomes.

MODEL

The two input images have mutually independent uniform pixels taking q
intensity values. Their absolute differences are independent with the
triangular finite-alphabet law. A score averages every unordered pair of
distinct pixels in each fully contained sliding w-by-w window, then divides
by q-1. There is no periodic wrapping. The case w=2 is the usual BACI
construction. Cases w>2 are its complete-pair sliding-window extension.

The comparison variance treats every pair occurrence as independent,
including repeated appearances of the same pair. The ratio is exact variance
divided by this pair-occurrence independence approximation. It quantifies the omitted
dependence. It is not a rejection probability or a security measure.

OUTPUTS

generated/window_variance_ratios.csv gives q=256 ratios and exact normalized
score variances for w=2,3,4,5 at 16-by-16, 64-by-64 and 256-by-256 pixels,
together with the ratio limit as BOTH image dimensions grow. Numerator and
denominator columns preserve exact values. Infinite-grid rows do not report
a finite-grid variance.

generated/window_large_alphabet_limits.csv gives the q-to-infinity limit of
the large-grid ratio at each fixed window width.

generated/window_metadata.json records the model and arithmetic conventions.
generated/window_checks.json records the independent numerical checks.

INTERFACE

    from core import window_moments, large_grid_ratio
    result = window_moments(256, 256, 3, q=256)
    print(result["variance"])
    print(result["variance_ratio"])
    print(large_grid_ratio(3, q=256))

The default alphabet is q=256. Dimensions must be at least the window width,
and width and q must be integers at least two. Results use fractional units,
so multiplying a score by 100 multiplies its variance by 10000 and leaves
all variance ratios unchanged. Neither the finite calculations nor the
large-grid limits require a normal approximation.
