ANALYTIC INTERPRETATION OF THE SAVED POWER EXPERIMENT

From the computational archive root, run

    python practical_calibration/power/analytic_interpretation.py --archive-root . --output-dir ../analytic_power_rebuild

Choose an absent output directory outside the supplied archive or input tree.
The script creates that directory and reads the saved design, calibration
bank, thresholds, null-validation sample, partial-identity samples and
conditional rejection summaries. It generates no random draws and changes
none of those inputs. To use another input directory or output location, pass
--power-dir PATH instead of --archive-root, and --output-dir PATH (also accepted
as --output PATH). NumPy and SciPy are required. Explicit input and output paths
keep regenerated output separate from the saved computational archive.

Outputs

partial_identity_power.csv contains exact binomial NPCR rejection probabilities,
moment-based Gaussian BACI/UACI approximations at the saved score cutoffs,
boundary rejection probabilities and the existing conditional estimates with
their Wilson confidence intervals. Values are fractions rather than percentages.

partial_identity_moments.json contains exact rational pixel and pair-distance
moments, plus normalized statistic means and variances for each registered
partial-identity strength. Independence across pixels is preserved under this
alternative. BACI retains all repeated-pair and shared-endpoint contributions.

horizontal_copy_gaussian_benchmark.csv contains the ideal-threshold Gaussian
variance-only benchmark for UACI/NPCR, the exact BACI mean decrease and its
standardization by the null standard deviation. It does not contain saved-bank
rank power predictions or a BACI alternative-variance approximation.

verification.json records checks and hashes of the read-only inputs. Every
saved cutoff is reconstructed from its numerical scores and tie variables;
all reused conditional rejection counts are reproduced. Exact rational
enumeration on small overlapping grids verifies the partial-identity moment
formula, and literal pair counts verify the horizontal-copy boundary factor.

Interpretation

For NPCR the number of unequal pixels is binomial under partial identity. Its
exact conditional rejection probability includes both the strict tail and the
saved boundary rejection probability multiplied by the boundary mass. Numerical
binomial evaluation is checked independently by cumulative-tail summation.

For BACI and UACI, exact alternative means and variances parameterize a continuous
Gaussian surrogate at the saved numerical cutoff. The surrogate assigns zero
boundary probability. These are approximations, not exact power probabilities.
The Gaussian values therefore retain the observed numerical threshold while
the saved tie probability has no effect in this continuous approximation.
These approximations describe detection mechanisms, not a globally certified
fit to the saved power curve. The verification output identifies four Gaussian
predictions outside their pointwise Monte Carlo Wilson intervals. Agreement at
strength 0.005 and significance 0.05 is specific to that highlighted comparison.

For horizontal copying, UACI and NPCR keep their null means and their variances
increase by the factor 1+theta. The ideal null Gaussian threshold then gives
2*Phi(-z/sqrt(1+theta)). Its apparent size independence is a property of that
approximation, not an exact finite-grid statement, particularly for NPCR.
The BACI standardized mean shift has an exact finite-boundary coefficient and
grows proportionally to theta*sqrt(rows*columns) when both dimensions grow.
This mean-shift scaling does not establish a universal finite-grid power curve.

All confidence intervals copied into the output condition on the realized
calibration bank. Marginal exactness of the original randomized rank calibration
does not turn those conditional probabilities into exactly alpha under every
realized bank.
