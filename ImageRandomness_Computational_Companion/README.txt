ImageRandomness computational companion
Exact moments, finite-grid calibration and paired detection

Contents

moment_computation/ contains exact breakpoint algorithms, exact and float64
alphabet baselines, deterministic checks, timing and allocation records,
and replicated Monte Carlo moment errors.

statistical_validation/ contains finite-grid calibration studies, paired
rank-calibrated detection experiments, raw integer scores, exact alternative
means and saved-data verification. Each study has a README.txt and a detailed
numerical record with relative paths and units.

Requirements

The exact engines use Python's standard library. The recorded numerical
environment uses Python 3.12.14, NumPy 2.3.5 and SciPy 1.17.0 on Linux x86-64.
The exact-calibration enumeration additionally uses a C++17 compiler.
No external image dataset is needed for these two studies.

Verification

Make a working copy before running component commands. Some verifiers write
new verification outputs, whose elapsed times and checksums differ from the
recorded outputs. From this companion directory, run

    python3 verify_export.py
    python3 -B verify.py --output ../companion_verification

The first command verifies every public-export mapping and its numerical-payload
digest. The second creates an unused working directory, checks implementation
identities and saved records, and leaves the distributed files unchanged.
The component guides describe deterministic implementation checks and
saved-data audits. Full simulation and timing commands are separate from
those checks. Timing measurements must run sequentially and depend on the
host and system load.

Export provenance

EXPORT_PROVENANCE.json identifies original experimental file digests and
public-export digests separately. Descriptive metadata and local machine
paths were normalized for distribution. Numerical observations, seeds,
model parameters, execution statuses and mathematical outputs are unchanged.
An original experimental digest is a provenance identifier, not a claim that
the public file has those original bytes. The verification adapters validate
the public digest before comparing the preserved experimental identifier.
The mapping does not claim independent observation of the original execution.

SHA256SUMS.txt identifies the delivered files. Experimental records retain
separate timing batches and all incomplete or censored calls. No numerical
result has been inferred from an omitted timing observation.

Units

Integer NPZ arrays store raw statistics. CSV and JSON rejection rates,
confidence endpoints, paired differences and relative errors are fractions.
Multiply rates by 100 for percentages and paired differences by 100 for
percentage points. Timing fields ending in seconds use seconds; allocation
fields ending in bytes use bytes. One MiB is 2^20 bytes. Rational values use
numerator/denominator objects or ratio strings.

Exact moments and Gaussian tail approximations are different outputs.
Independent-null rank calibration has a marginal finite-sample interpretation.
Intervals based on a reused bank condition on that realized bank. Pointwise
intervals do not imply simultaneous coverage. Mean-sensitivity indices alone
do not determine rejection probabilities.
