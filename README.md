# Image randomness computational archive

Numerical code, simulation outputs and deterministic reference calculations for
blocked average changing intensity (BACI), unified average changing intensity
(UACI), number of pixels change rate (NPCR), and local Shannon entropy (LSE).
Three external grayscale crops are retained as fixed computational reference
inputs, with their third-party provenance.

Package version **1.0.0** is recorded in `VERSION.txt` and `RELEASE.json`.
The file manifest identifies the contents of this package. A package version
is separate from any repository commit, published tag or DOI.

## Minimal check

Run from this archive root with Python 3.11 or later and NumPy installed.
NumPy 2.3.5 is the pinned version used for the supplied numerical archive.

```sh
python -B quick_check.py
```

The command verifies the archive manifest, compares the exact fixed-image
BACI and UACI routines with independent enumeration of all 729 comparison
images for a three-symbol 2 by 3 reference image, and checks whole-image
replication scaling. It also checks the random/random BACI moments against
all 256 pairs of binary 2 by 2 images. It prints a JSON report, makes no
random draws and writes no files. SciPy, mpmath, image libraries and a C++
compiler are unnecessary for this small check. This check does not replace
the full archive verification below.

For exact moments of a supplied integer reference image, use the dedicated
fixed-image implementation from the archive root.

```python
import sys
sys.path.insert(0, "fixed_image_and_power/fixed_algorithm")
from fixed_image_moments import fixed_baci_moments, fixed_uaci_moments

image = [[0, 1, 2], [2, 0, 1]]
baci = fixed_baci_moments(image, q=3)
uaci = fixed_uaci_moments(image, q=3)
print(baci.mean, baci.variance)
print(uaci.mean, uaci.variance)
```

Both routines return exact fractions by default. Exact moments and Gaussian
reference intervals are different outputs; the latter remain approximations.

## Setup and verification

The tested environment uses Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, mpmath
1.3.0 and GCC 13.3 with C++17/OpenMP support. Install the pinned Python packages.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -B verify_all.py --output ../reproduction_results
```

Choose an absent output directory outside this archive. This command verifies
all supplied hashes, runs the original reference and saved-array checks,
recomputes deterministic corrections, repeats independent finite-law checks,
reanalyzes the archived power arrays, and regenerates the practical reference
intervals and analytic power interpretation without new random draws.
The latter seven numerical outputs must match the supplied files byte for byte.
It copies writable computational modules into the chosen output directory
before invoking their launchers.
Supplied inputs, outputs and source files remain unchanged. No network access
is required once dependencies are installed.

Add `--engine` to compile the original C++ engine in a temporary directory and
replay selected expanded trials using one and three threads. Add
`--regenerate-addenda` to recompute the full high-precision entropy moments and
regenerate all 13 power-array archives. These options may be combined.

```sh
python -B verify_all.py --engine --output ../verified_with_replay
python -B verify_all.py --engine --regenerate-addenda --output ../regenerated_addenda
```

Neither command regenerates the full original baseline or expanded campaign.
The commands below do so explicitly, with new output directories.

```sh
python -B core/baseline/run.py simulate --output ../baseline_fresh
python -B core/expanded/run.py simulate --output ../expanded_fresh --threads 4
python -B core/expanded/run.py verify --data ../expanded_fresh
```

Paths above are interpreted from the current working directory, assumed to be
this archive root. The complete expanded campaign draws 2,324,845 whole-image
statistics and has substantially greater cost than selected replay or analysis.
The component READMEs describe all individual commands and output schemas.

## Components and verification scope

| Directory | Contents | Default verification | Additional regeneration |
| --- | --- | --- | --- |
| `core/` | Original uniform-pixel baseline, 256 by 256 expanded BACI data, calibration utilities and binary-entropy control | All archived summaries and deterministic tests | Full campaigns require the explicit simulate commands; `--engine` performs selected replay |
| `reference_calibration/` | Aggregate BACI thresholds, all 78 entropy reference intervals and exact binary controls | Saved entropy tables, exact controls and BACI checks | `--regenerate-addenda` recomputes complete high-precision entropy moments |
| `window_reference/` | Finite-grid larger-window moments, exact identities and numerical normal-approximation bounds | Complete deterministic regeneration and independent checks | No random simulation in this component |
| `fixed_image_and_power/` | Callable fixed-image moments, three reference crops and paired BACI/UACI/NPCR power experiments | Exact-moment regeneration and power-array reanalysis | `--regenerate-addenda` regenerates all 13 raw power archives |
| `practical_calibration/` | BACI Gaussian intervals, saved null shape diagnostics and analytic interpretation of the paired power experiment | Complete deterministic regeneration and byte comparison of seven numerical outputs | No new random draws in this component |

The original archive validation fully regenerated the baseline campaign. Its
expanded checks recomputed every reported summary from archived arrays and
replayed 28 selected trial IDs, yielding 1,240 whole-image numerator comparisons
across two thread counts. They did not regenerate the entire expanded campaign.
The fixed-image and power package also records complete regeneration of its
13 raw archives. A new invocation records only the operations it actually runs
in verification_report.json under the selected output directory. Historical
component records are retained as such.

The practical-calibration component can also be verified on its own. It reads
the existing exact routines, saved evaluation arrays and realized power bank.
Its Gaussian approximations, exact binomial probabilities and conditional
Monte Carlo estimates retain separate labels.

```sh
python -B practical_calibration/reproduce.py --archive-root . --output ../practical_calibration_rebuild
```

## Reproduction levels and random streams

Reanalysis recomputes summaries from archived numerical observations. Replay
recreates selected saved random draws. Full regeneration recreates every draw
in the specified component. These are distinct checks.

| Level | Command or option | Scope |
| --- | --- | --- |
| Minimal | `python -B quick_check.py` | Manifest verification and two small independent exact finite-law checks |
| Offline verification | `python -B verify_all.py --output ../reproduction_results` | Saved-array reanalysis, deterministic regeneration and component checks |
| Selected replay | Add `--engine` to `verify_all.py` | C++ engine compilation and selected trial replay across two thread counts |
| Full addendum regeneration | Add `--regenerate-addenda` to `verify_all.py` | Complete entropy moment sums and all 13 raw power archives |
| Full original campaigns | Explicit `core/baseline/run.py simulate` and `core/expanded/run.py simulate` commands above | All random draws in the selected original campaign |

The original baseline uses NumPy PCG64 with fixed seed spawning, dtypes, batch
sizes and draw ordering. The expanded engine uses standardized
std::mt19937_64 with explicit integer byte extraction and disjoint seed-input
ranges. Its per-trial outputs do not depend on thread scheduling. The power
component uses NumPy PCG64, SeedSequence stream identifiers and a registered
batch size. Component configurations and READMEs specify these schedules.
Use the pinned numerical versions and batch sizes for exact stream replay.
Compression-version differences may change NPZ bytes while integer arrays
remain identical in the original expanded component. The newer power checker
requires byte equality under its pinned environment.

Elapsed times and runtime version records may change between runs. The
fixed-image/power reproduction excludes only the declared execution metadata
and wall-clock timing from its cross-run equality checks; it compares all raw
power arrays and numerical outputs. No theoretical claim of independence is
inferred solely from distinct pseudorandom seeds.

## Callable routines and units

The dedicated exact fixed-image implementation is
`fixed_image_and_power/fixed_algorithm/fixed_image_moments.py`. Its
`fixed_baci_moments(image, q=256, repeats=1, exact=True)` and
`fixed_uaci_moments` routines use exact integer sums. Setting `exact=False`
converts only the final rational answers to floats. The BACI routine reuses
reflected intensity laws and evaluates projections one row at a time. Its
arithmetic operation count is O((MN + t^2)q), where t is the number of reflected
intensity classes. This is an arithmetic count, not an integer bit-complexity
bound. It implements every overlapping 2 by 2 window; larger square-window
reference calculations are in `window_reference/windows/`.

`core/calibration/` supplies the earlier general finite-law routines, Gaussian
intervals and independent-null rank calculations. Its similarly named
`fixed_baci_moments` defaults to `exact=False`, evaluates moments using
floating-point law arrays and stores a projection for every vertex and
alphabet value. Its optional exact mode is intended for small reference
examples. It remains a separate general implementation; use the dedicated
fixed-image module above for the newer exact algorithm. The two APIs have
different return fields. In contrast, `core/calibration.iid_baci_moments` and
`core/calibration.fixed_uaci_moments` return exact rational moments by default.
The component READMEs specify complete contracts and diagnostic fields.

For a user-specified image size, alphabet, repetition count and nominal level,
run the following from this archive root. Exact moments and the subsequent
Gaussian approximation are separate return values.

```python
import sys
sys.path.insert(0, "core")
from calibration import iid_baci_moments, gaussian_interval
moments = iid_baci_moments(256, 256, q=256, repeats=100, exact=True)
interval = gaussian_interval(moments.mean, moments.variance, alpha=0.05)
print(moments.mean, moments.variance, interval.lower, interval.upper)

sys.path.insert(0, "fixed_image_and_power/fixed_algorithm")
from fixed_image_moments import fixed_baci_moments, fixed_uaci_moments
fixed = [[0, 64], [128, 255]]
baci = fixed_baci_moments(fixed, q=256, repeats=1, exact=True)
uaci = fixed_uaci_moments(fixed, q=256, repeats=1, exact=True)
baci_interval = gaussian_interval(baci.mean, baci.variance, alpha=0.05)
uaci_interval = gaussian_interval(uaci.mean, uaci.variance, alpha=0.05)
```

The last two intervals use normal approximations, including for the tiny
illustrative image. Use independent-null rank calibration when a finite-sample
level guarantee is required. The examples demonstrate the callable interface.

Normalized BACI, UACI and NPCR scores are fractions; variances are squared
fractions. Entropy uses bits. Whole-image repeats divide variances once by the
number of independent repetitions. Gaussian intervals use exact moments but
remain approximations to finite-sample tails. Independent-null rank guarantees
are marginal over calibration and test draws. Reported Wilson intervals for a
reused bank condition on that realized bank. Randomized rank boundaries in the
power study and conservative tie-inclusive rank rules in the original null
experiments are separate procedures. No statistic certifies encryption security.

## Inputs, file integrity and release status

SHA256SUMS.txt lists the supplied files except itself. The top-level verification
checks this manifest before running a component. Array files contain numeric
data and are opened with allow_pickle=False. Source credits for the three
reference crops are in THIRD_PARTY.txt and the natural-images module.
Re-fetching upstream PNGs is optional and requires Pillow 12.3.0 and network
access; ordinary reproduction uses the supplied arrays.

LICENSE_STATUS.txt records the unassigned software and original-data license
status. The third-party image status is documented separately. Package metadata
does not assign a repository commit, published tag or DOI.
