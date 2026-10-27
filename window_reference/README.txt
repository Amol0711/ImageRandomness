FINITE-GRID REFERENCE CALIBRATION

This component provides deterministic numerical support for image-randomness
experiments. It contains exact reference calculations, finite-law enumeration,
and reproducible numerical checks. No new Monte Carlo simulations are
included.

REQUIREMENTS AND REPRODUCTION

Python 3.10 or later is sufficient. No third-party packages or network access
are required. From this directory, run

    python reproduce.py

This regenerates every numerical CSV and JSON file and runs all included
checks. Each component can also be run separately.

    python moments/verify_moment_identities.py
    python moments/verify_residual_bound.py
    python windows/generate.py
    python windows/check.py
    python normal_approximation/generate.py

CONTENTS

moments/ contains exact pair moments, fixed-coordinate pooling calculations,
and conditional covariance examples. Direct enumeration checks the numerical
formulas independently. Its residual calculations form the raw distance
matrix and center its rows and columns.

windows/ contains exact boundary-aware reference moments for every fully
contained sliding square window. Each window averages all unordered pair
distances. Window width two is the blocked average changing intensity (BACI)
construction. Larger widths use its complete-pair extension. Generated tables
cover widths two through five and image dimensions 16, 64, and 256. The checker
also enumerates 255 rectangular designs and the complete difference-image
laws of five small designs.

normal_approximation/ evaluates numerical local-dependence cumulative
distribution function (CDF) bounds for three parameter choices. All clipped
bounds equal one. These examples therefore give no nontrivial finite-sample
Gaussian accuracy or tail certification. They are bound evaluations, not
simulation estimates of normal-approximation error.

NUMERICAL CONVENTIONS

Pixel intensities run from zero to q-1. Random/random calculations assume two
images with mutually independent uniform pixels. Fixed/random calculations
condition on specified intensity values and use independent uniform comparison
pixels. Whole-image replication counts refer to independent image draws.

The window construction has no periodic wrapping. Exact reference variances
retain repeated-pair and shared-pixel dependence. Reported window variance
ratios are exact variance divided by the pair-occurrence independence approximation.
They are neither rejection probabilities nor security measurements.

Normalized scores are fractions. Multiplying a score by 100 multiplies its
variance by 10000 without changing its variance ratio. Conditional covariance
examples in moments/ use unnormalized intensity units. Empirical variances in
pooling calculations use divisor n.

Integer and fractional output fields retain exact values. Decimal fields use
floating-point display. Exact moments do not make a Gaussian decision rule an
exact finite-sample test. No exact tail claim is made from the bound examples.

All files are numerical code, numerical outputs, or usage documentation.
