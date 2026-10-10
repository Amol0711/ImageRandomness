# Uniform-pixel baseline simulations

This component contains numerical score arrays, simulation settings, numerical routines, and reproducible summaries. All images are synthetic independent uniform pixels or explicitly defined deterministic patterns. Image arrays are generated in memory and discarded after their scores are computed. The archived data contain scores, not images.

## Run

Run these commands from the repository root with NumPy and SciPy installed.

```sh
python baseline/run.py verify
python baseline/run.py analyze --output results/baseline-analysis
python baseline/run.py simulate --output results/baseline-replay
```

Every output directory must be new. Outputs within `baseline/` are refused to protect source files and archived data. The scripts locate their own inputs, so their operation is independent of the current working directory. `analyze --data PATH` can analyze a directory produced by `simulate`.

A reduced run exercises every design with fewer evaluation units.

```sh
python baseline/run.py simulate --bank-count 64 --fresh-count 64 --entropy-trials 100 --output results/baseline-smoke
```

Reduced runs have different sample counts and rejection rates. Their summaries record the actual counts. The default simulation reproduces the full settings in `config.json`. Generation used Python 3.12.14, NumPy 2.3.5, and SciPy 1.17.0. Matching NumPy version, draw order, dtype, seed spawning, and batch sizes supports exact replay of the saved random streams. Floating-point summaries can differ slightly across numerical-library builds.

## Experimental settings

The alphabet is the integers 0 through 255. Each random image has mutually independent uniform pixels. Separate random images and repeated whole-image measurements are independent. Every baseline score design has 9,999 independent calibration-bank scores and 5,000 independent evaluation scores.

| Design | Shape | Score |
| --- | --- | --- |
| Random/random pair | 16 by 16 | BACI |
| Random/random pair | 64 by 64 | BACI |
| Random/random pair | 256 by 256 | BACI |
| Mean of 100 independent random/random pairs | 16 by 16 per image | BACI |
| Constant fixed image with value 128 and one random image | 64 by 64 | UACI |
| Checkerboard fixed image with values 0 and 128 and one random image | 64 by 64 | UACI |
| Random/random pair | 64 by 64 | UACI and NPCR |

For a difference image `D = abs(U - V)`, BACI is the mean of all six unordered absolute pixel-pair differences within each overlapping 2 by 2 block, normalized by 255 and averaged across blocks. UACI is the mean absolute pixel difference divided by 255. NPCR is the fraction of unequal pixel pairs. All three scores are stored as fractions, not percentages. UACI and NPCR control scores within the same array index share the same random image pair.

The entropy experiment has 10,000 trials. Each trial averages the base-2 plug-in entropies of 30 independent blocks containing 1,936 independent uniform pixels each. Multinomial count vectors generate exactly the required block histogram distribution without storing pixel arrangements. Entropy scores use bits.

## Random streams and replay

The pixel experiments use `numpy.random.default_rng`, its `PCG64` bit generator, and master seed `202610041850`. A `SeedSequence` spawns 14 children in the design order shown above. Child `2*i` produces design `i`'s bank and child `2*i+1` produces its evaluation sample, with `i` starting at zero. Pixels are generated as `int16`. The batch size is 32 evaluation units for 256 by 256 images and 64 for all other image designs. For each random/random batch, all first-image pixels are generated before all second-image pixels. In the 100-repeat design, one evaluation unit contains 100 complete independent image pairs.

Entropy uses a separate `default_rng` with seed `202610040187`, batches of 100 trials, and multinomial probabilities `1/256`. Batch settings and draw order are intentionally fixed. Changing pixel batch sizes changes how the random stream is assigned to paired images. A reduced pixel run that stops partway through a batch need not match the corresponding prefix of a full run. Counts of 64 align with the pixel batch sizes and support prefix comparisons.

## Data schema

`data/null_samples.npz` contains 16 one-dimensional `float64` arrays. Keys have the form `<design>_<metric>_<purpose>`, where purpose is `bank` or `fresh` and metric is `baci`, `uaci`, or `npcr`. Design identifiers are `BACI_16x16_R1`, `BACI_64x64_R1`, `BACI_256x256_R1`, `BACI_16x16_R100`, `UACI_fixed_constant_128_64x64`, `UACI_fixed_checkerboard_0_128_64x64`, and `random_random_64x64`. Every bank array has length 9,999 and every fresh array has length 5,000.

`data/entropy_null_samples.npz` contains one one-dimensional `float64` array called `scores`, with length 10,000. The arrays contain no object types and are loaded with `allow_pickle=False`.

`config.json` records numeric settings, generator settings, and software versions. `expected/` contains deterministic summaries recomputed from the archived score arrays. No generation times, execution paths, or machine-specific logs are recorded.

## Comparators and interpretation

Every Gaussian pixel-score comparator rejects when the absolute deviation from its null mean strictly exceeds its two-sided Gaussian threshold. The nominal levels are 0.01 and 0.05. The BACI routines evaluate the entire finite alphabet to obtain `A = Var(abs(D1-D2))` and `B = Var(E[abs(D1-D2) | D1])` for independent difference pixels. For an `m` by `n` image, `K = (m-1)(n-1)`, `S2 = 10*m*n-12*m-12*n+14`, and `W = 62*m*n-96*m-96*n+148`. Variances are divided by the number of independent whole-image repeats.

| Method identifier | Variance or decision rule |
| --- | --- |
| `edge_independence_gaussian` | BACI variance `A / (6*K*255^2)` |
| `block_independence_gaussian` | BACI variance `(A/6 + 2*B/3) / (K*255^2)` |
| `overlap_gaussian` | BACI variance `(A*S2 + 2*B*W) / (36*K^2*255^2)` |
| `fixed_position_gaussian` | Fixed-image UACI variance equal to the sum of position-specific pixel variances divided by squared image size |
| `pooled_pixel_gaussian` | Fixed-image UACI variance including the across-position variance of conditional pixel means divided by image size |
| `standard_gaussian` | Exact independent-pixel variance of random/random UACI or NPCR, with a Gaussian threshold |
| `independent_rank` | Rank p-value `(1 + number of bank deviations at least as large as the evaluation deviation) / (bank size + 1)` |

The rank comparator rejects when its p-value is at most the nominal level. The bank is independent of the evaluation sample and reused across evaluation units within each design. Ties are included in the upper tail. A `null` strict threshold in a very small reduced run means that the level is below the smallest achievable p-value, so rejection is impossible. Gaussian thresholds are approximations to the score distribution, even when their input moments are exact.

Entropy comparison settings are fixed numeric inputs. `fixed_interval` accepts scores in `[7.901901305, 7.903037329]`. `gaussian_block_mean` accepts the interval centered at `7.902469317` with half-width `norm.isf(0.025) * 0.008694225 / sqrt(30)`. These decimal constants have finite precision. Rejection occurs strictly outside the stated intervals.

All reported uncertainty intervals are 95% Wilson intervals for the evaluation rejection count. For the rank comparator they describe variation conditional on the realized reused bank. They do not include uncertainty from drawing another bank. Rank-test validity averaged over bank and evaluation randomness follows from exchangeability; it does not imply an identical conditional rejection probability for every bank.

## Generated summaries and validation

`results.json` includes `pixel_reference_moments`, per-design `metrics` and `tests`, `entropy`, and `npcr_exact`. It records actual bank and evaluation counts, means, variances, rejection counts, Wilson intervals, and thresholds. `moments.csv` provides the eight image-score moment summaries. `rejection_rates.csv` provides 52 pixel-score comparisons and two entropy comparisons. `npcr_exact.csv` contains four probabilities calculated by summing the binomial probability mass function for the 4,096-pixel NPCR control. Rank probabilities in that file are conditional on the archived bank.

`verify` recomputes summaries and compares them against `expected/results.json` with relative tolerance `2e-12` and absolute tolerance `2e-14` for floating-point values; integer counts and strings must match exactly. It also checks all 512 binary 3 by 3 arrays against a literal BACI implementation, independently enumerates all 256 binary 2 by 2 image pairs to check BACI moments, enumerates weighted graph counts at six grid sizes, checks constant and mixed fixed-image variance behavior, and checks rank ties and small-bank behavior.

The archived entropy arrays retain trial means only. Individual block entropies and block-level empirical variance cannot be recovered from these arrays; they are not included in the expected summaries.
