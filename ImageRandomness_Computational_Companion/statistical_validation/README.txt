Finite-grid calibration and paired power
Known null laws, dependence alternatives and window sensitivity

Purpose and inventory

This study evaluates Gaussian calibration under 27 specified independent
null configurations and paired detection under eight specified alternatives.
It also evaluates 220 exact finite-grid mean-sensitivity indices.

code/moment_engine.py          Exact breakpoint moment engine
code/alphabet_baseline.py      Exact and float64 alphabet moment baselines
code/statistical_core.py       Integer scores, laws, ranks and intervals
code/alternative_analysis.py   Exact alternative means and window sensitivity
code/run_calibration.py        Gaussian null-calibration experiment
code/run_power.py              Shared-field paired-power experiment
code/verify_statistics.py      Independent finite exact checks
code/audit_results.py          Independent reanalysis of saved statistics
code/calibration_predictions.py Exact-moment Gaussian predictions and score replay
code/npcr_copying_diagnostic.py Exact copying law and independent seed replay
results/protocol.json          Prespecified design, alternatives and seeds
results/calibration.json/csv   Full calibration summaries
results/calibration_scores.npz Raw calibration integer statistics
results/power.json             Full thresholds, rates, contrasts and metadata
results/power_rates.csv        Rejection rates and pointwise Wilson intervals
results/power_paired.csv       Paired differences and uncertainty intervals
results/power_*.npz            Bank, held-out null and alternative score arrays
results/sensitivity.json/csv   Exact mean-sensitivity calculations
results/verification.json      Independent implementation verification
results/audit.json             Saved-data audit and NPCR binomial checks
results/power_audit.json       Additional power-output integrity audit
results/calibration_predictions.json/csv Displayed null predictions and verified counts
results/npcr_copying_diagnostic.json Copying-law and saved-generator checks
analysis/exact_calibration.py   Rational finite-grid rejection certification
analysis/enumerate_cut_histogram.cpp Exact integer support histograms
results/exact_calibration.json/csv Exact probabilities and certified bounds

Readable statistical detail

STATISTICAL_DETAILS.txt records calibration tails, exact and certified
probabilities, conditional-bank diagnostics, paired contrasts and
window-sensitivity values. It maps results to the complete saved records.
The six-statistic comparison, including lag-three copying, is retained in
results/power_rates.csv.

Raw image fields are regenerated from recorded seed streams. The NPZ files
retain the integer statistics required to reconstruct the recorded decisions,
not the complete image fields.

Reproduction

Use Python 3.12.14, NumPy 2.3.5 and SciPy 1.17.0 to match the recorded
environment. Make a working copy before executing commands from this
directory. Verification, audit and simulation commands write result files.

    python3 code/verify_statistics.py
    python3 code/audit_results.py
    python3 code/calibration_predictions.py
    python3 code/npcr_copying_diagnostic.py

The second command recomputes decisions and intervals from saved integer
scores and tie seeds, without regenerating images. It also evaluates the
conditional binomial NPCR formula, including randomized boundary mass.
Its numerical probabilities use SciPy float64 evaluation.

The prediction script recomputes the exact raw mean, variance and
pair-occurrence variance for twelve specified null cases.
It verifies the archived rational moments and score hashes, reconstructs both
Gaussian rules and their Wilson intervals, and evaluates the normal-tail
prediction from their variance ratio. The predicted probabilities are Gaussian
approximations. Exact moment evaluation does not imply exact Gaussian tails.

The copying diagnostic derives NPCR probabilities from the zero-count
generating function using integer polynomial coefficients. The realized bank
threshold and its tie cutoff determine the conditional rejection probability.
The cutoff is represented as its exact binary rational value. Exhaustive
small-field checks verify the copying law and pairing geometry. An independent
column-index implementation replays the NPCR and UACI sequences for all
199,999 recorded fields and reconstructs every saved NPCR decision. All
original inputs and observations are preserved. Pointwise binomial diagnostics
describe the observed deviations and are not simultaneous tests across models.

For full regeneration, run the following after verification and then rerun
code/audit_results.py. The checksum argument prevents a modified design from
silently inheriting the recorded experiment's identity.

    python3 code/run_calibration.py --authorized-protocol-sha256 850a18ff561948380f0b28f755a3ee1a24c2793ea60febe1acf1cb8df209c4ac
    python3 code/run_power.py --authorized-protocol-sha256 850a18ff561948380f0b28f755a3ee1a24c2793ea60febe1acf1cb8df209c4ac
    python3 code/alternative_analysis.py --authorized-protocol-sha256 850a18ff561948380f0b28f755a3ee1a24c2793ea60febe1acf1cb8df209c4ac
    python3 code/audit_results.py

The protocol retains case definitions, sample sizes, discrepancy conventions,
uncertainty methods and independent seed streams. Simulation elapsed times
record workload and are not comparative timing measurements. The calibration
and power jobs may overlap in wall-clock time in the recorded run.

Calibration design

There are 20,000 independent fields per case except the 256x256 case, which
uses 5,000, totaling 525,000 fields. The Gaussian rule uses the exact mean and
variance of the specified independent null law. The diagnostic alternative
variance is sum_e c_e Var(h_e), rather than sum_e c_e^2 Var(h_e), and omits
shared-endpoint covariance. Results retain alpha 0.05, 0.01 and 0.001.

Wilson intervals are pointwise, not simultaneous equivalence statements.
The 0.001 level corresponds to about 20 expected rejections in 20,000 fields
and five in 5,000 fields. Exact moments remain valid for the stated known
nonuniform laws even when the resulting distribution is not well approximated
by a Gaussian distribution.

Exact finite-grid evaluation

The Gaussian rule is evaluated independently on the binary 4x4 grid by
complete enumeration of all 65,536 difference fields. For the sparse 16x16
law, every support with at most four occupied pixels is enumerated, totaling
177,589,057 supports. Each support retains its original probability, without
renormalization. The omitted mass is exactly
Pr(Binomial(256,1/1024)>4), approximately 6.38217463693e-6 in probability
units, or 0.000638217464 percentage points. Adding that mass once to the
retained rejection probability gives a certified upper bound. Lower-tail
rejection is impossible for this sparse law at the three evaluated levels.

All decision thresholds and probability bounds use integer or rational
arithmetic. Rational normal-CDF enclosures certify the quantile brackets;
every attainable integer score is separated from the uncertain boundary.
Displayed lower and upper interval endpoints are rounded outward. These are
exact or certified rejection probabilities of the existing Gaussian rules,
not new exact-level rules. The original Monte Carlo observations are retained.

Run from this directory with Python and a C++17 compiler (CXX, or c++)

    python3 analysis/exact_calibration.py --verify-only
    python3 analysis/exact_calibration.py --output-dir results/exact_calibration_reproduction

The C++ executable is built in a temporary directory and removed afterward.
The verify-only route checks small histograms, binary 4x4 enumeration, exact
moments and the decision thresholds. The full route adds the sparse-support
enumeration. Results include histograms, exact rational weights, quantile
certificates, source hashes and environment details. No simulation inputs or
recorded simulation outputs are modified.

Paired-power design

The grid is 64x64 with q=256. The six raw statistics are NPCR count, UACI sum
and complete-pair scores for 2x2, 3x3, 4x4 and 5x5 windows. All six statistics
observe the same fields within each model. An independent bank of 99,999
fields calibrates exact integer centered discrepancies. Independent uniform
tie variables resolve equal numerical scores. Their seeds are retained, and
ties are reused across the two alpha levels to preserve nested decisions.

The tests have exact common marginal nominal levels over calibration banks
and test observations. The realized conditional rejection probabilities vary
with the bank. A separate 20,000-field null sample estimates those conditional
probabilities. Beta-law intervals describe variation across fresh banks and
are distinct from confidence intervals for the fixed bank's conditional size.

Each nonzero alternative has 10,000 independent fields. Paired intervals use
two 97.5% Clopper-Pearson intervals for discordant probabilities, giving at
least 95% pointwise conditional coverage for each difference. This coverage
does not assert simultaneous control across all reported contrasts.

Copying uses disjoint donor/recipient pairs at a specified horizontal lag,
with independent Bernoulli activation per pair. It copies difference values,
equivalently both raw image intensities together. Copying only one input image
would be a different model. Incomplete terminal blocks remain unchanged.
This construction preserves each one-pixel law but changes joint laws and
additive-statistic variances. Partial identity independently replaces selected
differences by zero and changes their one-pixel distribution.

Interpretation

Mean-sensitivity is the exact mean decrease divided by the null standard
deviation. It does not determine power by itself. The 3*lag/2 and binary
2*lag ratios describe successive large-grid and large-lag limits of this
index. The finite search records its objective and width range explicitly.
The studies evaluate the declared statistical models and do not establish
cryptographic security of an encryption system.

Public export identity

The companion-root EXPORT_PROVENANCE.json distinguishes original experimental
digests from distributed-file digests. Descriptive metadata and local paths
are normalized; numerical observations and computational parameters retain
their original values. Public checksums must not be interpreted as original
experimental file checksums.
