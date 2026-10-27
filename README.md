# Image randomness computation

Executable algorithms, numerical reference calculations and recorded experiments
for weighted absolute differences on finite image grids. The archive includes
blocked average changing intensity (BACI), unified average changing intensity
(UACI), number of pixels change rate (NPCR), and local Shannon entropy.

## Components

| Directory | Computation and data |
| --- | --- |
| `ImageRandomness_Computational_Companion/moment_computation/` | Exact breakpoint algorithms, exact and floating-point alphabet baselines, operation checks and recorded timing/allocation experiments |
| `ImageRandomness_Computational_Companion/statistical_validation/` | Gaussian calibration, randomized-rank detection, exact finite-grid probabilities and saved-data checks |
| `core/` | Uniform-pixel simulation arrays, BACI/UACI calibration routines and exact binary-entropy controls |
| `reference_calibration/` | Aggregate BACI reference rules, finite-sum entropy moments and reference intervals |
| `window_reference/` | Finite-grid window moments, exact enumeration and normal-approximation bound calculations |
| `fixed_image_and_power/` | Exact fixed-image routines, three grayscale reference arrays and paired BACI/UACI/NPCR observations |
| `practical_calibration/` | Gaussian reference intervals, null-shape diagnostics and analytic power calculations |

The component guides specify the input models, definitions, units and
experimental scope. `PUBLIC_EXPORT.json` records the numerical source snapshot
and the hashes of retained files. The companion also records the relationship
between its public file hashes and the original experiment digests.

## Requirements

Use Python 3.12 with the pinned dependencies for matching the saved numerical
outputs. The exact moment engines themselves use Python's standard library.
NumPy supports simulations and array analysis; SciPy supplies probability and
interval calculations; mpmath supports high-precision finite entropy sums.
A C++17 compiler is needed only for selected simulation replay and exact
binary enumeration. Install the Python packages with

```sh
python -m pip install -r requirements.txt
```

The recorded environment used NumPy 2.3.5, SciPy 1.17.0 and mpmath 1.3.0.
Different numerical versions can change the last displayed digits or random
stream output. No external download is required to use the supplied data.

## Verify the package

From the extracted root, the minimal check verifies every file in the manifest
and compares exact moment routines with independent small-grid enumeration.
It makes no random draws and writes no files.

```sh
python -B quick_check.py
```

The combined check uses a new output directory outside the supplied archive.
It reconstructs saved-array summaries, regenerates deterministic calculations,
checks exact algorithms and verifies the recorded benchmark evidence. It does
not run new timing measurements or regenerate the large simulation campaigns.

```sh
python -B verify_all.py --output ../verification_results
```

The output directory must not already exist. Reports identify the checks
actually run. Runtime durations belong to that invocation; they do not replace
the supplied observations. All supplied input and result files are checked
again after verification.

The two more expensive options are explicit

```sh
python -B verify_all.py --engine --output ../verification_with_replay
python -B verify_all.py --regenerate-reference-power --output ../regenerated_results
```

`--engine` compiles the simulation engine in temporary storage and replays
selected expanded trials. `--regenerate-reference-power` recomputes the complete entropy
moment sums and the 13 fixed-image power arrays. Neither option reruns every
uniform-pixel baseline or expanded simulation. Those campaigns have separate
commands in `core/README.md`.

The weighted-graph companion has its own guide and verification command at
[ImageRandomness_Computational_Companion/README.txt](ImageRandomness_Computational_Companion/README.txt).
Run computational benchmarks sequentially, in a separate working copy. Their
wall-clock results depend on hardware and current load. Preserve the supplied
experimental records when reproducing them.

## Callable fixed-image moments

The dedicated exact routine for overlapping 2 by 2 windows returns rational
moments by default

```python
import sys
sys.path.insert(0, "fixed_image_and_power/fixed_algorithm")
from fixed_image_moments import fixed_baci_moments, fixed_uaci_moments

reference = [[0, 1, 2], [2, 0, 1]]
baci = fixed_baci_moments(reference, q=3)
uaci = fixed_uaci_moments(reference, q=3)
print(baci.mean, baci.variance)
print(uaci.mean, uaci.variance)
```

For arbitrary weighted finite graphs and large alphabets, use the breakpoint
moment engines documented in the computational companion. The separate
`core/calibration/` API includes a general floating-point implementation and
specifies its own defaults and return fields.

## Numerical conventions

Normalized BACI, UACI and NPCR scores are fractions; variances are squared
fractions. Entropy is measured in bits. Integer raw-array statistics retain
exact score ties. Multiply fractional rejection probabilities by 100 to obtain
percentages. Whole-image repeats divide the variance by the number of
independent repetitions.

Exact moments, Gaussian approximations and finite-sample rank procedures are
separate calculations. Randomized-rank guarantees average over the independent
calibration bank and observation. Conditional rejection rates and uncertainty
intervals refer to the stated realized bank. Alternative-response experiments
use their specified copying and partial-identity models.

## File integrity and input provenance

`SHA256SUMS.txt` covers every distributed file except itself. Component manifests
cover their own directory trees. Numerical arrays are loaded with
`allow_pickle=False` where NumPy archives are used. `PUBLIC_EXPORT.json`
distinguishes retained source digests from the distributed file digests.

The three grayscale reference arrays are fixed center crops of Camera, Clock
and Grass. Their third-party source credits, crop origins and hashes remain in
`THIRD_PARTY.txt` and `fixed_image_and_power/natural_images/`. These credits are
required input provenance. Runtime dependencies are not vendored.

`LICENSE_STATUS.txt` records the licensing status of the original software and
generated numerical data. Third-party input rights are documented separately.
