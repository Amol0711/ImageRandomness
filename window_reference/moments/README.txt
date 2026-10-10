EXACT FINITE-ALPHABET MOMENTS

Run from this directory using Python 3.10 or later.

    python verify_moment_identities.py
    python verify_residual_bound.py

No third-party packages are required. Each command regenerates its exact
CSV and JSON outputs under data/. The calculations are deterministic finite
sums and exact integer or fractional arithmetic, not Monte Carlo trials.

The first command compares reference moments with direct enumeration for
34 alphabet sizes, including q=256. It also checks all 1,014 ordered pairs
of fixed intensities for q=2,...,14, additional coordinate designs, and
53 conditional-covariance examples. Its numerical checks include the
distinction between 0/255 and 0/128 fixed-intensity patterns.

The second command forms raw conditional distance matrices and centers
their rows and columns. It checks the common-support residual lower bound
for 1,034 fixed-law designs and 18 triangular laws, as well as the uniform
residual reference value at 35 alphabet sizes.

moment_identities.py supplies reusable functions for reference moments,
conditional means, pooling excess, and shared-endpoint covariance.
Empirical variances use divisor n. Conditional pair-distance covariance
is reported in unnormalized intensity units. Exact moments do not by
themselves give exact Gaussian acceptance probabilities.
