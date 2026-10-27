FIXED-IMAGE MOMENTS AND PAIRED POWER

This numerical package provides exact fixed-image moments, grayscale reference
input calculations, and a paired power experiment for three image statistics.
It contains executable numerical code, fixed pixel arrays, simulated integer
statistics, and reproducible numerical outputs. Third-party input credits are
retained in natural_images/THIRD_PARTY.txt.

Quick start
Install the tested dependencies and run the complete offline reproduction.

    python -m pip install -r requirements.txt
    python reproduce.py

The second command needs no network connection. It verifies the supplied file
inventory, copies the computational modules into a new reproduced directory,
removes copied derived outputs, regenerates them, and compares the results
with the bundled reference files. It regenerates all 13 power-array archives
from their registered random streams. The three fixed NPY inputs are retained
and their hashes verified. All regenerated files and reproduction_report.json
are written below reproduced. If that directory already contains files, choose
a fresh location with --output-dir fresh_results.

For faster reanalysis using the archived power arrays, run

    python reproduce.py --quick --output-dir reanalysis

Quick mode recomputes exact moments, repeats independent checks, and rebuilds
all power summaries from raw arrays. It does not repeat random image generation.
Neither mode modifies the bundled reference outputs.

Modules
fixed_algorithm
    Callable fixed_baci_moments and fixed_uaci_moments routines, returning exact
    rational moments by default, plus five deterministic example patterns.
algorithm_checks
    Independent exhaustive uniform-symbol enumeration and edge-covariance
    checks of the callable routines, including overflow and boundary cases.
natural_images
    Three metadata-free 256 by 256 uint8 grayscale crops and a predetermined
    histogram-preserving permutation control. Exact numerical moments include
    independent checks and third-party provenance. The original images remain
    fixed reference inputs; random comparison pixels follow the declared law.
power
    A prespecified comparison of blocked average changing intensity (BACI),
    unified average changing intensity (UACI), and number of pixels change
    rate (NPCR). Contains 13 raw NPZ archives, independent calibration and
    validation samples, paired alternative replications, thresholds, interval
    calculations, and rejection-rate summaries.

Units and statistical interpretation
Means use fraction units, not percentages. Variances use squared fraction
units. BACI is evaluated on all overlapping 2 by 2 windows. Fixed-image
routines return exact moments and do not by themselves define a finite-sample
acceptance interval. The power study uses independent calibration with
randomized ties, with significance levels matched marginally over calibration
and test draws. A realized calibration bank has its own conditional rejection
probabilities, which are separately checked. Read the module READMEs for the
sampling models, fixed-image versus pooled-image distinction and diagnostics.

Reproduction guarantees
SHA256SUMS.json separately verifies every supplied source, input and output
file, excluding the inventory itself. Full reproduction checks byte identity
of five exact-moment/input-manifest files and all 17 deterministic power files,
including the 13 raw archives. The independent algorithm-verification JSON is
compared after removing only its wall-clock timing field.

Runtime information is not a scientific output. power/results/execution.json
records elapsed time, runtime versions and whether raw arrays were reused;
it may differ across executions. power/results/sha256.json incorporates that
execution record, so it changes correspondingly. Its internal file integrity
is checked on every run. These two files are excluded from cross-run byte
comparison. No numerical thresholds, estimates, uncertainty intervals, exact
moments or raw random outputs are excluded. Use the pinned NumPy and SciPy
versions and the registered batch size for byte-level reproduction.

Input provenance and optional refresh
The bundled NPY files contain pixel arrays with shape and dtype metadata only.
The upstream inputs are CC0 or public-domain images distributed by scikit-image
0.25.2. Source URLs, crop origins and checksums are recorded locally. Recreating
these arrays from upstream PNG bytes is optional and requires Pillow 12.3.0 and
a network connection. The optional command is

    python natural_images/prepare_inputs.py --fetch

The default reproduction never retrieves upstream files.

Directory layout
The four computational module directories form a self-contained component.
Keep their relative arrangement unchanged when running the reproduction script.
