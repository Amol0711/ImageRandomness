# Image randomness simulations

Simulation code and synthetic numerical data for blocked average changing
intensity (BACI), unified average changing intensity (UACI), number of pixels
change rate (NPCR), and local Shannon entropy (LSE), computed with the plug-in entropy estimator.

All random inputs follow specified independent uniform-pixel models. The
archive contains numerical routines, synthetic fixed patterns, saved random
statistics, experiment configurations, usage documentation and reproducibility
checks.

## Setup

The validated environment uses Python 3.12, NumPy 2.3.5, SciPy 1.17.0, and
GCC 13.3 with C++17 and OpenMP support. The Python dependencies are pinned.
Analyzing the archived arrays needs only Python and these dependencies.
Replaying or generating the expanded BACI experiments also requires `g++`.

From the extracted repository directory run

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -B verify.py
python -B verify.py --engine
```

The first verification command checks file integrity, runs the deterministic
reference tests, regenerates the numerical summaries from the archived arrays,
and checks the exact binary-entropy control. `--engine` additionally compiles
the C++ engine in a temporary directory and checks deterministic statistics,
selected archived trials, and thread-count invariance. Neither command reruns
the full expanded simulation campaign or modifies the supplied data.

## Experiments

| Component | Designs | Stored data |
| --- | --- | --- |
| `baseline/` | BACI at 16×16, 64×64, and 256×256; a mean of 100 independent 16×16 BACI values; fixed-pattern UACI and random/random UACI and NPCR at 64×64 | 9,999 calibration and 5,000 evaluation scores per design and metric |
| `baseline/` entropy | 10,000 means of 30 independent 1,936-pixel, 256-symbol blocks | Mean entropy scores in bits |
| `expanded/` | Means of 50 or 100 independent 256×256 BACI values, plus five fixed-pattern 256×256 BACI designs | 2,324,845 integer whole-image numerators, trial totals, and five synthetic reference arrays |
| `controls/` | Exact binary two-pixel block entropy, averaged over 30 blocks | Enumerated probability mass function and exact rejection fractions |

The five expanded fixed patterns are constant zero, constant 128, a 0/128
checkerboard, a 0/128 vertical split, and a column ramp from 0 through 255.
Only these deterministic reference images are stored. Random images are
generated transiently and reduced to the archived numerical statistics.

## Reproduce numerical outputs

Use new output directories. The commands refuse to replace existing results.

```sh
python -B baseline/run.py analyze --output outputs/baseline
python -B expanded/run.py analyze --output outputs/expanded
python -B controls/binary_entropy.py --output outputs/binary-entropy
```

The component READMEs describe columns, array shapes, generator settings,
comparison rules, and complete regeneration commands. The supplied
configurations preserve the original numerical seeds, sample sizes, draw
ordering, and batch sizes. They are clean reproduction specifications, not
records of external preregistration. They require no network access once the
dependencies are installed.

## Numerical conventions

BACI, UACI, and NPCR are stored as fractions, not percentages. Entropy uses
base-2 logarithms. A repeated-image trial averages independent whole-image
statistics. Its variance is divided by the number of repetitions. Fixed-image
experiments condition on the complete reference pattern and randomize the
comparison pixels.

`calibration/` provides statistics, exact rational or floating-point moments,
Gaussian intervals, and independent-null rank p-values. The legacy identifiers edge_independence_gaussian,
block_independence_gaussian and pooled_pixel_gaussian identify Gaussian rules
using the pair-occurrence independence approximation, the within-window
covariance diagnostic and pooled-pixel variances, respectively.
The full covariance calculation retains all overlapping-window dependencies.
Some fixed numerical interval endpoints are retained as comparison parameters.

Gaussian intervals are approximate even when their moments are exact. Rank
tests include ties in the upper tail and use an independent calibration bank.
Their finite-sample validity is marginal over both the bank and test draw.
Reported Wilson intervals describe evaluation uncertainty conditional on the
saved bank when it is reused. The simulated null models do not measure cipher
security or detection power.

## Validation and scope

`SHA256SUMS.txt` verifies this component's distributed files. `VALIDATION.json`
records the original archive validation. It is a historical computational
record, not a report that every simulation was regenerated during this
consolidation. The original baseline campaign was fully regenerated. The
expanded campaign was reanalyzed from all 2,324,845 stored whole-image
numerators and selected trials were replayed. A complete expanded regeneration
command is provided in expanded/README.md and is distinct from verification.

Numerical arrays are loaded with allow_pickle=False. Shared license status and
third-party input notices are documented at the top level of this archive.
