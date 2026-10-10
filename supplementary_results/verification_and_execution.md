# Numerical verification and execution

This record describes the numerical study's verification and execution methods.
The module READMEs give the commands and machine-readable outputs. Reanalysis,
selected random-stream replay and complete regeneration are distinct operations.
A fresh run reports only the operations it actually performs.

## Exact identities and independent implementations

Direct sliding-window sums and weighted sums over distinct edges give identical
BACI statistics. Finite-support calculations give exact rational moments, while
an independent covariance enumeration groups edges by orientation and shared
endpoint. For the five 256 by 256 fixed patterns, the numerical study reported
a maximum relative difference of 2.05e-13 between the exact moments and the
floating-point projection implementation. This value records the original
comparison. Both implementations and the fixed inputs are retained in `core/`.

Exhaustive evaluation in the original calibration component covers 86
three-symbol fixed-reference cases. These comprise all 81 references on a 2 by 2
grid and five additional references on 2 by 3 and 3 by 2 grids, with 10,206
complete comparison images. The checks distinguish the pair-occurrence
independence approximation, the within-window covariance diagnostic and the
full-covariance calculation. Two-pixel UACI, the binary entropy law and rank
calculations with ties provide additional exact checks.

The larger-window component checks 255 rectangular designs with window widths
two through six and 765 variance calculations by independent covariance
expansion. Complete probability-weighted enumeration of five small-image laws
covers 109,727 difference images, including overlapping 3 by 3 windows.
Finite-power sums verify the factored pair-variance and shared-endpoint
covariance formulas. Direct residual calculations verify the common-support
lower bound used in the normal approximation.

The dedicated exact fixed-image algorithm agrees with independent enumeration
on 360 fixed-reference laws and 75,244 complete comparison images. Additional
checks cover negative shared-endpoint covariance, reflected intensities,
transposed images, independent replication and alphabets through size 1024.
A 2 by 256 strip at alphabet size 512 requires an accumulation beyond the
signed 64-bit range and verifies the arbitrary-precision total. The five
256 by 256 synthetic references also agree exactly with the independent
covariance implementation.

## Saved observations and random streams

Repeated-image and fixed-image BACI calculations at 256 by 256 retain integer
numerators before normalization and aggregation. The C++17 engine uses
`std::mt19937_64`, with separate seed assignments for each evaluation unit and
disjoint seed-input ranges for reference banks and evaluation samples. Integer
byte extraction supplies symbols without a floating-point transformation.
Generator inputs and output order are independent of thread scheduling.
Exact centering of integer numerators preserves rank ties. Code, seed
assignments and all whole-image integer numerators are supplied in `core/`.
These pseudorandom computations implement the independent sampling models.

The entropy simulation draws multinomial symbol counts under the uniform-pixel
block law. Natural-reference moments are retained as exact rational values
before display rounding. The power experiment uses a separate NumPy
implementation with independently seeded image and tie-breaking streams.
Its integer statistics and random tie values reproduce the rejection counts
and paired comparisons.

Wilson 95% intervals describe rejection fractions. When a bank is reused,
they condition on that realized bank. The independent-null rank guarantee is
marginal over the independent bank and test observation. The power study uses
randomized ties to match marginal levels; the original null-calibration rank
rule includes ties without randomization.

## Deterministic reference calculations

Aggregate BACI probabilities use coefficient recursion with the exact grid
moments. Rational binomial-tail sums determine the count cutoffs.
Finite-multinomial sums independently reproduce entropy moments and Gaussian
reference intervals. Exact binary tails give the rejection probabilities of
both the Gaussian rules and the integer-count alternatives. The numerical
files retain source inputs, recomputed moments, interval conventions and
thresholds separately.

Natural-reference calculations use the supplied metadata-free integer arrays.
`fixed_image_and_power/natural_images/input_manifest.json` records crop origins,
source provenance and checksums. `prepare_inputs.py` records the deterministic
permutation. It orders row-major indices by SHA-256 digests of the byte prefix
`fixed-image-permutation-v1`, a zero byte and the index as four big-endian bytes,
breaking digest ties by index. The same permutation is used for every crop and
preserves each intensity histogram.

## Recorded validation scope

The original archival validation reanalyzed every recorded observation and
replayed selected expanded trials across thread counts. It did not regenerate
the full expanded 256 by 256 campaign. Deterministic moment and interval
calculations, natural-reference outputs and the full power experiment were
regenerated from the supplied inputs during their recorded validation runs.
Historical component records retain those scopes. Root `verify_all.py` and the
component READMEs specify the operations performed by each new invocation.

The supplementary-results module exports displays and tables from these
existing results. It makes no new random draws and changes no scientific
algorithm, input array or archived numerical result.

## Locations

| Evidence | Repository location |
| --- | --- |
| Baseline and expanded calibration | `core/README.md`, `core/VALIDATION.json`, `core/expanded/README.md` |
| Larger windows, pair identities and residual bound | `window_reference/README.txt` |
| Exact fixed-reference algorithm checks | `fixed_image_and_power/algorithm_checks/algorithm_verification.json` |
| Natural-reference inputs and exact outputs | `fixed_image_and_power/natural_images/` |
| Saved power observations and paired comparisons | `fixed_image_and_power/power/` |
| Aggregate and entropy reference calculations | `reference_calibration/README.md` |
| Practical intervals and analytic power interpretation | `practical_calibration/README.md` |

The published numerical baseline is version 1.0.0 at commit
`70192958be5da28d19fda50a6524de52a9708d4d` of
<https://github.com/Amol0711/ImageRandomness>.
Version 1.0.1 adds this presentation module while retaining those numerical
inputs, algorithms and results.
