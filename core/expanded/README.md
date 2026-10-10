# Expanded BACI simulations

This component contains seven independent-uniform simulation designs on 256 by
256 grids with pixel alphabet 0 through 255. BACI is the sum of the six absolute
pair differences in every overlapping 2 by 2 window of a difference image,
divided by `6*(rows-1)*(columns-1)*255`. A difference image has pixels `abs(U-V)`
for two independent uniform images, or `abs(fixed-U)` for a synthetic fixed image
and an independent uniform comparison image.

Each design has 9,999 bank trials followed by 5,000 evaluation trials. A trial
averages the specified number of independently generated whole-image values.
Bank and evaluation trials use distinct random streams. `config.json` specifies
all seeds, trial-index boundaries, patterns, nominal levels, and numerical
intervals. It is a reproduction configuration, with no registration or original
run-timing assertion.

| Design | Fixed image or model | Whole-image statistics per trial |
| --- | --- | --- |
| `random_R50` | Two independent uniform images | 50 |
| `random_R100` | Two independent uniform images | 100 |
| `fixed_zero` | Constant 0 | 1 |
| `fixed_mid` | Constant 128 | 1 |
| `fixed_checker` | `128*((row+column)%2)` | 1 |
| `fixed_split` | 0 for columns below 128, otherwise 128 | 1 |
| `fixed_ramp` | Pixel equals column index | 1 |

All indices are zero-based. Synthetic patterns are also stored as exactly
65,536 row-major unsigned bytes in `data/fixed_images/*.u8`. These contain no
headers or image metadata.

## Run

From the repository root, install the root requirements, then run:

```sh
python expanded/run.py analyze --output outputs/expanded_analysis
python expanded/run.py verify
python expanded/run.py check
```

`analyze` reads the archived integer observations and creates `moments.csv`,
`rejection_rates.csv`, and `analysis.json`. It recomputes the exact finite-law
moments and does not draw new random observations. `verify` also compares every
field against the two shipped expected CSV files, including all 60 rejection
counts and all seven moment rows. Floating comparisons permit only numerical
roundoff; integer counts must agree exactly. Add `--output DIR` to `verify` or
`check` to retain a neutral JSON verification record.

`check` needs `g++` with C++17 and OpenMP support. It compiles in a temporary
directory, checks exhaustive small-grid moments and statistics, compares C++
draws against an independent Python MT19937-64 implementation, and replays the
first and last bank and evaluation trial of every archived design. Replay uses
one and three threads and compares each per-image numerator exactly. Small
stream checks also verify independence from work partitioning and cross the
312-word PRNG state transition. Compiler binaries and temporary draws are
removed automatically.

To regenerate the entire campaign:

```sh
python expanded/run.py simulate --output outputs/expanded_fresh --threads 4
python expanded/run.py verify --data outputs/expanded_fresh
```

The full campaign draws 2,324,845 whole-image statistics. Its computational cost
is substantially larger than analysis or selected-trial replay. Thread count
does not change a trial's random stream. Fresh NPZ compressed file bytes may
differ across NumPy or compression versions; the integer arrays must match.
Every command that writes results requires a new destination and refuses any
existing directory. Destinations inside this component are also rejected to
protect shipped inputs. Failures leave only incomplete new destinations, which
must be removed or renamed explicitly before rerunning.

## Stored arrays

Each `data/*_raw.npz` contains only numeric arrays, loadable with
`numpy.load(path, allow_pickle=False)`.

| Array | Type and shape | Meaning |
| --- | --- | --- |
| `raw_numerators` | `uint32`, `(14999, repeats)` | Exact unnormalized six-pair sum for each image |
| `trial_totals` | `uint64`, `(14999,)` | Exact sum of image numerators within each trial |
| `bank_trials` | `uint64` scalar | 9999 |
| `test_trials` | `uint64` scalar | 5000 |
| `repeats` | `uint64` scalar | 1, 50, or 100 |
| `per_image_scale` | `uint64` scalar | `6*255*255*255`, equal to 99488250 |

A trial statistic is `trial_totals/(repeats*per_image_scale)`. Rows 0 through
9998 form the rank bank; rows 9999 through 14998 form the evaluation sample.
Integer totals are reconstructed from the saved per-image values during every
analysis. No raw uniform pixel arrays are stored; the seeds and engine reproduce
them when needed.

## Numerical methods and output fields

The three Gaussian comparisons use the exact design mean with, respectively,
independent pair variances, sums of within-window variances, and the full
covariance variance. The first two deliberately omit dependence terms to serve
as numerical comparisons. Exact moments do not make Gaussian interval coverage
exact. Repeated-image variances are divided by the number of independent images.
Fixed-image moments use integer finite-law sums and geometric edge counting in
`fixed_moments.py`; the small-grid checks compare them with complete enumeration
and the root calibration implementation.

The rank method uses absolute deviation from the exact mean, represented by
integer arithmetic. Ties are included in the upper tail. Its p-value is
`(1 + number of bank scores >= evaluation score)/(bank_size+1)`.
The rank calibration guarantee averages over both the bank and observation;
reported Wilson intervals condition on the one reused bank. Gaussian rules use
strict rejection beyond unrounded endpoints. The fixed numerical interval
comparison applies only to the two repeated-image designs and rejects at or
beyond its configured percentage endpoints.

`rejection_rates.csv` has one row per design, nominal level, and method. It
records evaluation size, rejection count, rate, 95% Wilson limits, corresponding
percentage fields, and the conditioning of those limits. `moments.csv` records
means, variances, variance ratios, empirical evaluation moments, standardized
mean errors, and descriptive fourth-moment plug-in uncertainty for empirical
variance ratios. `analysis.json` additionally preserves exact moment fractions,
integer rank-score centering, decision thresholds, paired disagreement counts,
and per-input hashes. These are simulation summaries, not image-encryption
performance or security measurements.

## Random streams

The engine initializes a separate standardized C++ `std::mt19937_64` generator
for each trial with `SplitMix64(seed_base + trial_id)`. All SplitMix arithmetic
is modulo 2 to the power 64. For random pairs, each 64-bit draw supplies four
low-to-high 16-bit chunks; the low byte is `U` and the high byte is `V`. For fixed
images, it supplies eight low-to-high comparison bytes. Unused high chunks are
discarded at each image boundary. Pixels follow row-major order and consecutive
images within a trial continue the same stream. The design seed-input ranges
are disjoint. Pseudorandom streams implement the independent-uniform reference
model; disjoint seeds are not a mathematical proof of independence.
